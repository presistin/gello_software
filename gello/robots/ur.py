from typing import Dict
import os
import time
import socket
import numpy as np
import multiprocessing as mp
import ctypes
from gello.robots.robot import Robot

GRIPPER_IP = "192.168.1.20"
INVERT_GRIPPER = True 

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
        filter_alpha = 0.35 
        
        MIN_PHYSICAL = 0.8 
        MAX_PHYSICAL = 109.0 
        
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
            if abs(target_width - last_sent_width) > 0.5:
                try:
                    cmd = f"MOVE({target_width:.1f}, 400)\n".encode()
                    sock.send(cmd)
                    last_sent_width = target_width
                    try: sock.recv(128) 
                    except: pass
                except: pass
            
            state.current_gripper.value = raw_target
            
            elapsed = time.perf_counter() - loop_start
            if 0.033 - elapsed > 0:
                time.sleep(0.02 - elapsed)
                # time.sleep(0.033 - elapsed)

                
    except Exception as e:
        print(f"[Gripper] Err: {e}")
    finally:
        if sock: sock.close()

class URRobot(Robot):
    """A class representing a UR robot with WSG50 gripper (JAKA style)."""

    def __init__(self, 
                 robot_ip: str = "192.168.1.10", 
                 no_gripper: bool = False,
                 gripper_port: int = 1000,
                 gripper_ip: str = GRIPPER_IP):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        
        self._use_gripper = not no_gripper
        
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)
            
        self.state = SharedState()
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        # === JAKA 风格：只在需要时启动 WSG50 安全进程 ===
        self.p_gripper = None
        if self._use_gripper:
            self.p_gripper = mp.Process(
                target=process_gripper_safe,
                args=(self.state, gripper_ip, gripper_port, INVERT_GRIPPER),
                daemon=True
            )
            self.p_gripper.start()
            print("[Gripper] WSG50 process started (safe mode)")
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
        
        # === JAKA 风格：通过共享状态控制 WSG50 ===
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


def main():
    robot_ip = "192.168.1.101"
    ur = URRobot(robot_ip, no_gripper=False)   # ← 改成 False 即可开启 WSG50
    print(ur)
    ur.set_freedrive_mode(True)
    print(ur.get_observations())


if __name__ == "__main__":
    main()