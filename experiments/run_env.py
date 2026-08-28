import glob
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.utils.launch_utils import instantiate_from_dict
from gello.utils.serial_ports import resolve_gello_port, resolve_tactile_port
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.request_reply import ZMQCommunicationError
#添加相机类
from gello.zmq_core.camera_node import ZMQClientCamera

def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False

    # 默认 None 保持原来的无限阻塞行为；仅在显式传参时启用故障超时。
    robot_recv_timeout_ms: Optional[int] = None
    robot_send_timeout_ms: Optional[int] = None
    camera_recv_timeout_ms: Optional[int] = None
    camera_send_timeout_ms: Optional[int] = None

    # 默认关闭，不产生额外文件或控制循环 I/O。
    runtime_log_dir: Optional[str] = None
    runtime_log_flush_interval_s: float = 1.0

    # FIX004 staged rollout. legacy preserves the existing command path;
    # shadow computes/logs safety output; enforce sends the safety output.
    safety_mode: str = "legacy"
    safety_config: Optional[str] = None

    # 触觉（高速板 + M3025）：record_tactile 写入 zarr 供 DP 训练；use_tactile_grip_limit 采集防卡死
    record_tactile: bool = False
    """为 True 时每帧将 6 维触觉向量存入 replay_buffer.zarr（需重新采集）。"""
    use_tactile_grip_limit: bool = False
    tactile_port: Optional[str] = None
    tactile_force_limit_n: float = 2.0
    tactile_close_increases: bool = True
    tactile_baseline_sec: float = 1.5
    tactile_background_poll_hz: float = 25.0
    """后台触觉轮询频率；>0 时主循环不阻塞串口。"""

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


def main(args):
    from gello.safety import (
        SafetyFault,
        SafetySupervisor,
        load_safety_config,
        normalize_safety_mode,
    )

    safety_mode = normalize_safety_mode(args.safety_mode)
    safety_config = (
        load_safety_config(args.safety_config)
        if safety_mode != "legacy"
        else None
    )
    safety_supervisor = (
        SafetySupervisor(safety_config) if safety_config is not None else None
    )
    capture_timestamps = args.runtime_log_dir is not None or safety_supervisor is not None
    if safety_supervisor is not None and args.mock:
        raise ValueError("shadow/enforce safety modes require real robot and sensor timing")
    if safety_supervisor is not None and args.bimanual:
        raise ValueError(
            "FIX004 SafetySupervisor currently supports the configured 7D UR arm only"
        )
    if safety_supervisor is not None and not (
        args.record_tactile or args.use_tactile_grip_limit
    ):
        raise ValueError(
            "shadow/enforce safety modes require the M3025 runtime; add "
            "--record-tactile and/or --use-tactile-grip-limit"
        )
    if safety_supervisor is not None:
        print(
            f"[Safety] mode={safety_mode} config="
            f"{args.safety_config or 'config/safety.yaml'} "
            f"joint_step={safety_config.max_joint_step_rad:.3f} rad "
            f"grip_force={safety_config.tactile_grip_limit_n:.2f} N "
            f"ages(camera/tactile/robot)="
            f"{safety_config.max_camera_age_ms:.0f}/"
            f"{safety_config.max_tactile_age_ms:.0f}/"
            f"{safety_config.max_robot_age_ms:.0f} ms"
        )
        if safety_mode == "shadow" and args.runtime_log_dir is None:
            print(
                "[Safety] WARNING: shadow mode has no --runtime-log-dir; "
                "comparisons will not be persisted."
            )

    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        from gello.zmq_core.camera_node import ZMQClientCamera
        camera_clients = {

            # you can optionally add camera nodes here for imitation learning purposes
            "wrist": ZMQClientCamera(
                port=args.wrist_camera_port,
                host=args.hostname,
                recv_timeout_ms=args.camera_recv_timeout_ms,
                send_timeout_ms=args.camera_send_timeout_ms,
                capture_timestamps=capture_timestamps,
            ),
            "base": ZMQClientCamera(
                port=args.base_camera_port,
                host=args.hostname,
                recv_timeout_ms=args.camera_recv_timeout_ms,
                send_timeout_ms=args.camera_send_timeout_ms,
                capture_timestamps=capture_timestamps,
            ),
            "camera_2": ZMQClientCamera(
                port=5002,
                host=args.hostname,
                recv_timeout_ms=args.camera_recv_timeout_ms,
                send_timeout_ms=args.camera_send_timeout_ms,
                capture_timestamps=capture_timestamps,
            ),                 # 新增
            # "camera_3": ZMQClientCamera(port=5003, host=args.hostname),
        }
        robot_client = ZMQClientRobot(
            port=args.robot_port,
            host=args.hostname,
            recv_timeout_ms=args.robot_recv_timeout_ms,
            send_timeout_ms=args.robot_send_timeout_ms,
            capture_timestamps=capture_timestamps,
        )
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
        capture_timestamps=capture_timestamps,
    )

    resolved_gello_port: Optional[str] = None
    agent_cfg = {}
    if args.bimanual:
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBG6A-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": left,
                },
                "agent_right": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": right,
                },
            }
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "l",
                },
                "agent_right": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "r",
                },
            }
        elif args.agent == "spacemouse":
            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": left_path,
                    "verbose": args.verbose,
                },
                "agent_right": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": right_path,
                    "verbose": args.verbose,
                    "invert_button": True,
                },
            }
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        reset_joints_left = np.deg2rad([0, -90, -90, -90, 90, 0, 0])
        reset_joints_right = np.deg2rad([0, -90, 90, -90, -90, 0, 0])
        reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        
        # curr_joints = env.get_obs()["joint_positions"]
        curr_joints = env.get_obs()["joints"]
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
    else:
        if args.agent == "gello":
            tactile_exclude = [args.tactile_port] if args.tactile_port else None
            gello_port = resolve_gello_port(
                args.gello_port,
                exclude_ports=tactile_exclude,
            )
            resolved_gello_port = gello_port
            print(f"GELLO 串口: {gello_port}")
            agent_cfg = {
                "_target_": "gello.agents.gello_agent.GelloAgent",
                "port": gello_port,
                "start_joints": args.start_joints,
            }

            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    [90, -90, 90, -90, -90, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = np.array(args.start_joints)

            curr_joints = env.get_obs()["joints"]
            #curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                "robot_type": args.robot_type,
                "which_hand": "l",
            }
        elif args.agent == "spacemouse":
            agent_cfg = {
                "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                "robot_type": args.robot_type,
                "verbose": args.verbose,
            }
        elif args.agent == "dummy" or args.agent == "none":
            agent_cfg = {
                "_target_": "gello.agents.agent.DummyAgent",
                "num_dofs": robot_client.num_dofs(),
            }
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    agent = instantiate_from_dict(agent_cfg)
    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    #添加一行打印信息
    print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in obs.items()})

    joints = obs["joints"]
    # joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joints"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joints"]
    # joints = obs["joint_positions"]
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    # from gello.utils.control_utils import SaveInterface, run_control_loop

    # save_interface = None
    # if args.use_save_interface:

    #     save_interface = SaveInterface(
    #         data_dir=args.data_dir, agent_name=args.agent, expand_user=True
    #     )

    # run_control_loop(env, agent, save_interface, use_colors=True)
# 3.18.17.28
        # ──────────────────────────────────────────────────────────────
    # 进入主控制循环前，处理保存接口
    # ──────────────────────────────────────────────────────────────
    from gello.utils.control_utils import run_control_loop
    from gello.utils.control_utils import DiffusionPolicySaveInterface   # 确认这个 import 路径正确

    save_interface = None
    if args.use_save_interface:
        if not args.record_tactile:
            print(
                "提示: 未加 --record-tactile 时 zarr 不含触觉；"
                "训练 tactile 模型请开启 --record-tactile 并重新采集。"
            )
        print("[DEBUG] use_save_interface = True，正在创建 DiffusionPolicySaveInterface")
        save_interface = DiffusionPolicySaveInterface(
            data_dir=args.data_dir,              # ← 小写 args
            agent_name=args.agent,               # ← 小写 args，例如 "gello"
            expand_user=True
        )
        print("[DEBUG] DiffusionPolicySaveInterface 已成功初始化")
        print("=== DiffusionPolicySaveInterface 已启用 ===")
        print("按键控制：  S: 开始录制    Q: 停止录制并保存")
    else:
        print("[DEBUG] use_save_interface = False，不启用保存接口")

    print(f"[DEBUG] 最终 save_interface 类型: {type(save_interface).__name__ if save_interface else 'NoneType'}")

    tactile_runtime = None
    tactile_limiter = None
    need_tactile = (
        args.record_tactile
        or args.use_tactile_grip_limit
        or safety_supervisor is not None
    ) and not args.mock
    if need_tactile:
        gello_exclude = [resolved_gello_port] if resolved_gello_port else None
        tactile_port = resolve_tactile_port(
            args.tactile_port,
            exclude_ports=gello_exclude,
        )
        from gello.utils.tactile_runtime import TactileRuntime

        tactile_runtime = TactileRuntime(
            port=tactile_port,
            baseline_sec=args.tactile_baseline_sec,
            force_limit_n=args.tactile_force_limit_n,
            background_poll_hz=args.tactile_background_poll_hz,
        )
        if args.record_tactile:
            print("[Tactile] 已启用：每帧 tactile[6] 写入 zarr")
        if args.use_tactile_grip_limit:
            from gello.utils.tactile_gripper_limit import TactileGripperLimiter

            tactile_limiter = TactileGripperLimiter(
                runtime=tactile_runtime,
                force_limit_n=args.tactile_force_limit_n,
                close_increases_command=args.tactile_close_increases,
            )
    elif args.record_tactile:
        raise ValueError("record_tactile 需要真实硬件，请勿使用 --mock")

    runtime_logger = None
    if args.runtime_log_dir is not None:
        from gello.utils.runtime_logger import create_runtime_logger

        runtime_logger = create_runtime_logger(
            args.runtime_log_dir,
            run_kind="collection",
            metadata={
                "python_executable": sys.executable,
                "python_version": sys.version,
                "mode": "collection",
                "args": asdict(args),
                "safety_mode": safety_mode,
                "safety_config": (
                    safety_config.to_dict() if safety_config is not None else None
                ),
            },
            flush_interval_s=args.runtime_log_flush_interval_s,
        )
        print(f"[RuntimeLogger] {runtime_logger.path}")

    # 进入控制循环
    try:
        run_control_loop(
            env,
            agent,
            save_interface,
            use_colors=True,
            tactile_gripper_limiter=tactile_limiter,
            tactile_runtime=tactile_runtime if args.record_tactile else None,
            safety_tactile_runtime=tactile_runtime,
            safety_supervisor=safety_supervisor,
            safety_mode=safety_mode,
            runtime_logger=runtime_logger,
        )
    except SafetyFault as exc:
        if runtime_logger is not None:
            runtime_logger.log(
                "safety_fault",
                mode="collection",
                message=str(exc),
            )
        print(
            "\n[FAULT] Collection safety supervisor stopped command output; "
            "one current-state hold was sent and no further targets will be sent."
        )
        print(f"[FAULT] SafetyFault: {exc}")
        if save_interface is not None:
            save_interface.discard_current_episode_on_fault(reason="SafetyFault")
            save_interface.finalize_on_exit(reason="safety fault")
        raise SystemExit(3) from None
    except ZMQCommunicationError as exc:
        if runtime_logger is not None:
            runtime_logger.log(
                "communication_fault",
                mode="collection",
                error_type=type(exc).__name__,
                message=str(exc),
            )
        print(
            "\n[FAULT] Collection communication failed; "
            "no further robot targets will be sent."
        )
        print(f"[FAULT] {type(exc).__name__}: {exc}")
        if save_interface is not None:
            save_interface.discard_current_episode_on_fault(
                reason=type(exc).__name__
            )
            save_interface.finalize_on_exit(reason="communication fault")
        raise SystemExit(2) from None
    finally:
        if tactile_limiter is not None:
            tactile_limiter.close()
        if tactile_runtime is not None:
            tactile_runtime.close()
        if runtime_logger is not None:
            runtime_logger.close()
            if runtime_logger.dropped_events:
                print(
                    f"[RuntimeLogger] dropped_events={runtime_logger.dropped_events}"
                )

if __name__ == "__main__":
    main(tyro.cli(Args))
