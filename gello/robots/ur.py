from typing import Dict
import os
import time
import socket
import inspect
import numpy as np
import multiprocessing as mp
import ctypes
from gello.robots.robot import Robot

GRIPPER_IP = "192.168.1.101"
INVERT_GRIPPER = True
ONROBOT_DEFAULT_GRIPPER_ID = 0

class SharedState:
    def __init__(self):
        self.target_joints = mp.Array(ctypes.c_double, 6)
        self.current_joints = mp.Array(ctypes.c_double, 6)
        self.target_gripper = mp.Value(ctypes.c_double, 0.0)
        self.current_gripper = mp.Value(ctypes.c_double, 0.0)
        self.running = mp.Value(ctypes.c_bool, True)
        self.robot_ready = mp.Value(ctypes.c_bool, False)

def process_gripper_safe(state: SharedState, ip: str, port: int, invert: bool):
    print(f"[Gripper] Safe Mode Started.")
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(2.0)
        sock.connect((ip, port))
        sock.setblocking(False)
        
        # 初始化
        try: 
            sock.sendall(b"ACK\n")
            time.sleep(0.1)
            sock.sendall(b"MOVE(100, 100)\n")
        except: pass
        
        # === V13 核心参数（与 JAKA 完全一致）===
        last_sent_width = -999.0
        smooth_val = 0.0
        # filter_alpha = 0.15 
        filter_alpha = 0.45  # 更快跟随目标，减少“标签看起来已到位但实际未到位”的错配
        
        MIN_PHYSICAL = 0.8 
        MAX_PHYSICAL = 109.0 
        
        MIN_SEND_DELTA_MM = 0.05
        KEEPALIVE_STEPS = 30  # ~1s @ 30Hz
        keepalive_count = 0

        while state.running.value:
            loop_start = time.perf_counter()
            
            raw_target = state.target_gripper.value
            smooth_val = smooth_val * (1.0 - filter_alpha) + raw_target * filter_alpha
            
            if invert:
                target_width = MAX_PHYSICAL - (smooth_val * (MAX_PHYSICAL - MIN_PHYSICAL))
            else:
                target_width = MIN_PHYSICAL + (smooth_val * (MAX_PHYSICAL - MIN_PHYSICAL))
            
            target_width = max(MIN_PHYSICAL, min(MAX_PHYSICAL, target_width))
            
            # if abs(target_width - last_sent_width) > 1.5:
            if abs(target_width - last_sent_width) > 0.2:
                try:
                    cmd = f"MOVE({target_width:.1f}, 400)\n".encode()
                    sock.send(cmd)
                    last_sent_width = target_width
                    try: sock.recv(128) 
                    except: pass
                except: pass
            
            # 用平滑后的值更新“当前夹爪状态”，避免把目标值当成传感器读数
            state.current_gripper.value = smooth_val
            
            elapsed = time.perf_counter() - loop_start
            sleep_time = 0.033 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

                
    except Exception as e:
        print(f"[Gripper] Err: {e}")
    finally:
        if sock: sock.close()


def process_gripper_onrobot_safe(
    state: SharedState,
    robot_ip: str,
    gripper_id: int,
    invert: bool
):
    print("[Gripper-OnRobot] Safe Mode Started.")
    gripper_ctor = None
    ctor_name = ""
    try:
        # Prefer RG2 since that is the known-good class for the user's setup.
        import onRobot.gripper as onrobot_gripper
        if hasattr(onrobot_gripper, "RG2"):
            gripper_ctor = onrobot_gripper.RG2
            ctor_name = "onRobot.gripper.RG2"
        elif hasattr(onrobot_gripper, "RG"):
            gripper_ctor = onrobot_gripper.RG
            ctor_name = "onRobot.gripper.RG"
    except Exception:
        pass

    if gripper_ctor is None:
        try:
            # 兼容旧路径
            from onRobot.onRobot.gripper import RG2
            gripper_ctor = RG2
            ctor_name = "onRobot.onRobot.gripper.RG2"
        except Exception:
            # Fallback: work even when onRobot package is missing
            # (e.g. running robot node in diffusion_policy env).
            import requests
            import xmlrpc.client

            class _RG2HttpFallback:
                def __init__(self, rg_id: int = 0, host: str = "192.168.0.99", port: int = 41414):
                    self.rg_id = int(rg_id)
                    self.base_url = f"http://{host}:{port}"

                def rg_grip(self, target_width: float = 100.0, target_force: float = 10.0) -> bool:
                    target_width = max(0.0, min(100.0, float(target_width)))
                    target_force = max(0.0, min(40.0, float(target_force)))
                    xml_request = f"""<?xml version="1.0"?>
<methodCall>
<methodName>rg_grip</methodName>
<params>
<param><value><int>{self.rg_id}</int></value></param>
<param><value><double>{target_width}</double></value></param>
<param><value><double>{target_force}</double></value></param>
</params>
</methodCall>"""
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        data=xml_request.replace("\r\n", "").encode(),
                        timeout=1.0,
                    )
                    return response.status_code == 200

                def get_rg_width(self) -> float:
                    xml_request = f"""<?xml version="1.0"?>
<methodCall>
<methodName>rg_get_width</methodName>
<params>
<param><value><int>{self.rg_id}</int></value></param>
</params>
</methodCall>"""
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        data=xml_request.replace("\r\n", "").encode(),
                        timeout=1.0,
                    )
                    response.raise_for_status()
                    xml_response = xmlrpc.client.loads(response.text)
                    return float(xml_response[0][0])

            gripper_ctor = _RG2HttpFallback
            ctor_name = "_RG2HttpFallback"
            print(
                "[Gripper-OnRobot] onRobot package not found. "
                "Using built-in HTTP fallback client."
            )

    last_err_print_t = 0.0
    try:
        def normalized_to_width(value: float) -> float:
            value = max(0.0, min(1.0, float(value)))
            if invert:
                width = MAX_PHYSICAL - (value * (MAX_PHYSICAL - MIN_PHYSICAL))
            else:
                width = MIN_PHYSICAL + (value * (MAX_PHYSICAL - MIN_PHYSICAL))
            return max(MIN_PHYSICAL, min(MAX_PHYSICAL, width))

        ctor_signature = None
        try:
            ctor_signature = str(inspect.signature(gripper_ctor))
        except Exception:
            ctor_signature = "<unknown>"

        ctor_attempts = [
            ("rg_id + host keywords", lambda: gripper_ctor(rg_id=gripper_id, host=robot_ip)),
            ("single positional gripper_id", lambda: gripper_ctor(gripper_id)),
            ("rg_id keyword", lambda: gripper_ctor(rg_id=gripper_id)),
            ("robot_ip + gripper_id", lambda: gripper_ctor(robot_ip, gripper_id)),
        ]

        rg2 = None
        last_ctor_error = None
        for attempt_name, attempt in ctor_attempts:
            try:
                rg2 = attempt()
                print(
                    f"[Gripper-OnRobot] Using ctor: {ctor_name} "
                    f"signature={ctor_signature} via {attempt_name}"
                )
                break
            except Exception as e:
                last_ctor_error = e

        if rg2 is None:
            print(
                f"[Gripper-OnRobot] Constructor failed for {ctor_name} "
                f"signature={ctor_signature}: {last_ctor_error}"
            )
            return

        if not hasattr(rg2, "rg_grip"):
            print(
                f"[Gripper-OnRobot] Object from {ctor_name} has no rg_grip method: "
                f"{type(rg2).__name__}"
            )
            return

        MIN_PHYSICAL = 0.0
        MAX_PHYSICAL = 100.0
        FORCE_N = 40.0
        filter_alpha = 0.45

        initial_target = max(0.0, min(1.0, float(state.target_gripper.value)))
        smooth_val = initial_target
        initial_width = normalized_to_width(initial_target)

        # 启动后先按当前目标值做一次初始化，避免主循环第一拍反向跳变
        try:
            rg2.rg_grip(float(initial_width), 30.0)
            state.current_gripper.value = smooth_val
            print(
                "[Gripper-OnRobot] Init command sent. "
                f"target={initial_target:.3f}, width={initial_width:.1f}, invert={invert}"
            )
        except Exception as e:
            print(f"[Gripper-OnRobot] Init open failed: {e}")

        last_sent_width = initial_width

        def width_mm_to_normalized(w_mm: float) -> float:
            w_mm = max(MIN_PHYSICAL, min(MAX_PHYSICAL, float(w_mm)))
            if invert:
                return (MAX_PHYSICAL - w_mm) / (MAX_PHYSICAL - MIN_PHYSICAL + 1e-9)
            return (w_mm - MIN_PHYSICAL) / (MAX_PHYSICAL - MIN_PHYSICAL + 1e-9)

        # 主臂仍可“继续合”，从臂已夹死物体时，避免反复发更小宽度导致卡死/持续顶力。
        # 注意：必须用 raw_target 映射的宽度判断“主臂意图”，不能用滤波后的 target_width，
        # 否则平滑后目标仍接近实测，永远判不到 wants_tighter。
        prev_measured_width = None
        stall_count = 0
        GAP_MM = 0.55
        NOT_CLOSING_MM = 0.65
        STALL_FRAMES = 2
        # 仅当夹爪已经较合（疑似夹到物体或接近闭合）时才做防顶死；全开→闭合过程避免误触发
        STALL_ENABLE_BELOW_MM = 62.0
        MIN_SEND_DELTA_MM = 0.05
        KEEPALIVE_STEPS = 30  # ~1s @ 30Hz
        keepalive_count = 0

        while state.running.value:
            loop_start = time.perf_counter()
            raw_target = float(state.target_gripper.value)
            raw_target = max(0.0, min(1.0, raw_target))
            leader_width = normalized_to_width(raw_target)
            smooth_val = smooth_val * (1.0 - filter_alpha) + raw_target * filter_alpha

            target_width = normalized_to_width(smooth_val)

            width = None
            try:
                if not hasattr(rg2, "get_rg_width"):
                    raise AttributeError("get_rg_width not available")
                width = float(rg2.get_rg_width())
                width = max(MIN_PHYSICAL, min(MAX_PHYSICAL, width))
            except Exception as e:
                now = time.time()
                if now - last_err_print_t > 1.0:
                    print(f"[Gripper-OnRobot] get_rg_width failed: {e}")
                    last_err_print_t = now

            if (
                width is not None
                and prev_measured_width is not None
                and width < STALL_ENABLE_BELOW_MM
            ):
                leader_wants_tighter = leader_width < width - GAP_MM
                finger_not_closing = width >= prev_measured_width - NOT_CLOSING_MM
                if leader_wants_tighter and finger_not_closing:
                    stall_count += 1
                else:
                    stall_count = 0
                if stall_count >= STALL_FRAMES:
                    smooth_val = width_mm_to_normalized(width)
                    target_width = width
                    stall_count = 0
            elif width is not None:
                stall_count = 0
            if width is not None:
                prev_measured_width = width

            keepalive_count += 1
            should_send = abs(target_width - last_sent_width) > MIN_SEND_DELTA_MM
            if not should_send and keepalive_count >= KEEPALIVE_STEPS:
                should_send = True

            if should_send:
                try:
                    force = 18.0 if (width is not None and abs(target_width - width) < 0.8) else FORCE_N
                    rg2.rg_grip(float(target_width), force)
                    last_sent_width = target_width
                    keepalive_count = 0
                except Exception as e:
                    now = time.time()
                    if now - last_err_print_t > 1.0:
                        print(f"[Gripper-OnRobot] rg_grip failed: {e}")
                        last_err_print_t = now

            if width is not None:
                if invert:
                    state.current_gripper.value = (MAX_PHYSICAL - width) / (
                        MAX_PHYSICAL - MIN_PHYSICAL + 1e-6
                    )
                else:
                    state.current_gripper.value = (width - MIN_PHYSICAL) / (
                        MAX_PHYSICAL - MIN_PHYSICAL + 1e-6
                    )
            else:
                state.current_gripper.value = smooth_val

            elapsed = time.perf_counter() - loop_start
            sleep_time = 0.033 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except Exception as e:
        print(f"[Gripper-OnRobot] Err: {e}")

class URRobot(Robot):
    """A class representing a UR robot with WSG50 gripper (JAKA style)."""

    def __init__(self, 
                 robot_ip: str = "192.168.1.10", 
                 no_gripper: bool = False,
                 gripper_backend: str = "onrobot",
                 invert_gripper: bool = INVERT_GRIPPER,
                 gripper_port: int = 1000,
                 gripper_ip: str = GRIPPER_IP,
                 onrobot_gripper_id: int = ONROBOT_DEFAULT_GRIPPER_ID):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        
        self._use_gripper = not no_gripper
        self._gripper_backend = gripper_backend.lower()
        self._invert_gripper = bool(invert_gripper)
        
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)
            
        self.state = SharedState()
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.state.target_gripper.value = 1.0 if not self._invert_gripper else 0.0
        self.state.current_gripper.value = self.state.target_gripper.value

        # 启动夹爪安全进程（后端可切换）
        self.p_gripper = None
        if self._use_gripper:
            if self._gripper_backend == "wsg50":
                self.p_gripper = mp.Process(
                    target=process_gripper_safe,
                    args=(self.state, gripper_ip, gripper_port, self._invert_gripper),
                    daemon=True
                )
            elif self._gripper_backend == "onrobot":
                self.p_gripper = mp.Process(
                    target=process_gripper_onrobot_safe,
                    args=(self.state, robot_ip, onrobot_gripper_id, self._invert_gripper),
                    daemon=True
                )
            else:
                raise ValueError(
                    f"Unsupported gripper_backend={gripper_backend}. Use onrobot or wsg50."
                )
            self.p_gripper.start()
            print(f"[Gripper] backend={self._gripper_backend} process started (safe mode)")
        else:
            print("[Gripper] Disabled (no_gripper=True)")

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()

    def num_dofs(self) -> int:
        return 7 if self._use_gripper else 6

    def _get_gripper_pos(self):
        """从共享状态读取（JAKA 风格）"""
        if not self._use_gripper:
            return 0.0
        return self.state.current_gripper.value

    def get_joint_state(self) -> np.ndarray:
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        
        # 通过共享状态控制夹爪（第7维）
        if self._use_gripper and len(joint_state) > 6:
            gripper_pos = joint_state[-1]
            self.state.target_gripper.value = float(gripper_pos)

        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([0.0])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

    def get_health(self) -> Dict[str, object]:
        """Return a read-only snapshot of RTDE and gripper process health."""

        errors: Dict[str, str] = {}

        def read_bool(interface, method_name: str, key: str) -> bool:
            try:
                return bool(getattr(interface, method_name)())
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
                return False

        def read_scalar(interface, method_name: str, key: str):
            try:
                value = getattr(interface, method_name)()
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, (bool, int, float, str)) or value is None:
                    return value
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return str(value)
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
                return None

        control = getattr(self, "robot", None)
        receiver = getattr(self, "r_inter", None)
        control_connected = (
            read_bool(control, "isConnected", "control_connected")
            if control is not None
            else False
        )
        receive_connected = (
            read_bool(receiver, "isConnected", "receive_connected")
            if receiver is not None
            else False
        )

        emergency_stopped = (
            read_bool(receiver, "isEmergencyStopped", "emergency_stopped")
            if receiver is not None
            else False
        )
        protective_stopped = (
            read_bool(receiver, "isProtectiveStopped", "protective_stopped")
            if receiver is not None
            else False
        )

        health: Dict[str, object] = {
            "robot_connected": control_connected and receive_connected,
            "control_connected": control_connected,
            "receive_connected": receive_connected,
            "robot_mode": (
                read_scalar(receiver, "getRobotMode", "robot_mode")
                if receiver is not None
                else None
            ),
            "runtime_state": (
                read_scalar(receiver, "getRuntimeState", "runtime_state")
                if receiver is not None
                else None
            ),
            "safety_mode": (
                read_scalar(receiver, "getSafetyMode", "safety_mode")
                if receiver is not None
                else None
            ),
            "emergency_stopped": emergency_stopped,
            "protective_stopped": protective_stopped,
            "operational": (
                control_connected
                and receive_connected
                and not emergency_stopped
                and not protective_stopped
            ),
            "gripper_process_alive": (
                bool(self.p_gripper.is_alive())
                if getattr(self, "p_gripper", None) is not None
                else None
            ),
        }
        if errors:
            health["errors"] = errors
        return health


def main():
    robot_ip = "192.168.1.101"
    ur = URRobot(robot_ip, no_gripper=False)   # ← 改成 False 即可开启 WSG50
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()
