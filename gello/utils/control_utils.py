import sys
sys.path.append("/home/zaq/diffusion_policy")

import datetime
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
from numcodecs import blosc
import threading
import queue
import traceback

# 导入官方 ReplayBuffer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from gello.data_utils.keyboard_interface import KBReset
from gello.agents.agent import Agent
from gello.env import RobotEnv


class DiffusionPolicySaveInterface:
    """
    专为 Diffusion Policy 设计的保存接口，使用累积 + 一次性 add_episode 方式。
    目标：生成与官方 RealEnv 高度兼容的 replay_buffer.zarr 格式。
    支持：UR6 + WSG50夹爪 + 3个D435相机，action 固定为7维（6位姿 + 1夹爪）。
    """

    def __init__(
        self,
        data_dir: str = "~/bc_data",
        agent_name: str = "gello_ur",
        expand_user: bool = True,
        zarr_path: str = None,
        image_resolution: tuple = (480, 640),
        obs_float32: bool = False,
    ):
        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name

        # zarr 路径
        if zarr_path is None:
            zarr_path = str(self.data_dir / agent_name / "replay_buffer.zarr")
        self.zarr_path = zarr_path

        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path,
            mode='a'
        )

        print(f"DiffusionPolicySaveInterface 初始化完成")
        print(f"  zarr 路径: {zarr_path}")
        print(f"  当前 episode 总数: {self.replay_buffer.n_episodes}")
        print("按键控制：")
        print("  S : 开始录制（清空缓冲）")
        print("  Q : 停止录制并保存到 zarr")
        print("  D : 删除当前录制 or 最近一条已保存 episode")

        # 累积缓冲
        self.current_episode = {}           # dict of list
        self.current_timestamps = []
        self.in_recording = False
        self.image_resolution = image_resolution
        self.obs_float32 = obs_float32

        # 备份设置
        self.backup_npz = False  # 采集时尽量避免额外IO导致卡顿
        self.current_backup_path = None
        self.backup_buffer = []
        # 保存可能很慢：压缩 + 磁盘IO。放到后台线程，避免阻塞采集循环。
        self._save_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
        self._stop_save_worker = threading.Event()
        self._save_failed = False
        self._save_error_message: Optional[str] = None
        self._save_worker = threading.Thread(
            target=self._save_worker_loop, daemon=True
        )
        self._save_worker.start()

        # 相机映射（根据你的 env.get_obs() 实际键名调整）
        self.camera_mapping = {
            "image_wrist": "camera_0",
            "image_base": "camera_1",
            "image_camera_2": "camera_2",
        }

    def _save_worker_loop(self) -> None:
        """Background thread: stack + write episode to zarr."""
        while not self._stop_save_worker.is_set():
            try:
                payload = self._save_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:
                self._save_queue.task_done()
                break

            episode_dict, backup_path = payload
            try:
                self._save_episode_to_zarr(episode_dict, backup_path=backup_path)
            except Exception as e:
                self._save_failed = True
                self._save_error_message = f"{type(e).__name__}: {e}"
                print("\n❌ [SaveThread] Episode save failed.")
                print(f"❌ [SaveThread] {self._save_error_message}")
                print(traceback.format_exc())
                self._stop_save_worker.set()
            finally:
                self._save_queue.task_done()

    def _save_episode_to_zarr(self, episode_dict: Dict[str, Any], backup_path=None) -> None:
        timestamps = episode_dict["timestamps"]
        episode_lists = episode_dict["episode_lists"]

        episode = {
            "timestamp": np.array(timestamps, dtype=np.float64),
            "stage": np.zeros(len(timestamps), dtype=np.int64),
        }

        # 转换 list -> np.array（耗时操作放后台线程）
        for key, lst in episode_lists.items():
            if key.startswith("camera_"):
                episode[key] = np.stack(lst, axis=0)  # (T, H, W, 3) uint8
            elif key == "gripper_position":
                episode[key] = np.stack(lst, axis=0).reshape(-1, 1)  # (T, 1)
            else:
                episode[key] = np.stack(lst, axis=0)  # robot_joint/action (T, 7)

        # 写盘与压缩会显著影响采集实时性；lz4 通常更快
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=self.zarr_path,
            mode='a'
        )
        replay_buffer.add_episode(episode, compressors="default")
        self.replay_buffer = replay_buffer
        print(f"✅ [SaveThread] Episode saved. total={replay_buffer.n_episodes}")

        # npz 备份（默认关闭，避免再次阻塞）
        if self.backup_npz and backup_path is not None and episode_dict.get("backup_buffer"):
            backup_buffer = episode_dict["backup_buffer"]
            if backup_buffer:
                np.savez_compressed(
                    backup_path / "episode.npz",
                    **{
                        k: np.stack([d[k] for d in backup_buffer], axis=0)
                        for k in backup_buffer[0]
                    },
                )
                print(f"✅ [SaveThread] npz backup saved: {backup_path / 'episode.npz'}")

    def _queue_episode_for_save(self) -> bool:
        if not self.current_timestamps:
            self._reset_episode_buffer()
            return False

        episode_lists = self.current_episode
        timestamps = self.current_timestamps
        backup_path = self.current_backup_path
        backup_buffer = self.backup_buffer if self.backup_npz else None

        self._reset_episode_buffer()

        payload = {
            "timestamps": timestamps,
            "episode_lists": episode_lists,
        }
        if backup_buffer is not None:
            payload["backup_buffer"] = backup_buffer

        if self._save_queue.full():
            print(
                "\n⚠️ [SaveThread] Save queue is full; waiting for disk writer. "
                "Collection will pause briefly."
            )
        self._save_queue.put((payload, backup_path), timeout=30.0)
        print("⏳ [SaveThread] Episode queued. (main loop continues)")
        return True

    def finalize_on_exit(self, reason: str = "exit") -> None:
        if self._save_failed:
            raise RuntimeError(
                "Background save thread failed before shutdown. "
                f"Last error: {self._save_error_message}"
            )

        if self.in_recording and self.current_timestamps:
            print(
                f"\n>>> Finalizing current episode before {reason} "
                f"(length: {len(self.current_timestamps)}) <<<"
            )
            self._queue_episode_for_save()
        elif self.in_recording:
            print(f"\n>>> Clearing empty recording state before {reason} <<<")
            self._reset_episode_buffer()

        print("⏳ Waiting for background saves to finish...")
        self._save_queue.join()
        print("✅ All pending episodes written to replay_buffer.zarr")

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        if self._save_failed:
            raise RuntimeError(
                "Background save thread failed; stopping collection to avoid silent data loss. "
                f"Last error: {self._save_error_message}"
            )

        state = self.kb_interface.update()

        # ============== 实时图像可视化 ==============
        for img_key in [k for k in obs if k.startswith("image_")]:
            if img_key in obs:
                img = obs[img_key]
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                win_name = f"{img_key} (D435)"
                cv2.imshow(win_name, img[..., ::-1] if len(img.shape) == 3 else img)
        cv2.waitKey(1)

        current_time = time.time()

        if state == "delete_last":
            if self.in_recording and self.current_timestamps:
                print(
                    f"\n🗑️ 丢弃当前未保存 episode (长度: {len(self.current_timestamps)})"
                )
                self._reset_episode_buffer()
                return None

            if self.in_recording:
                print("\n⚠️ 当前处于录制态，但还没有采到有效帧；已清空当前录制状态。")
                self._reset_episode_buffer()
                return None

            print("\n🗑️ 请求删除最近一条已保存 episode，等待后台写盘完成...")
            self._save_queue.join()
            if self.replay_buffer.n_episodes <= 0:
                print("⚠️ 当前没有可删除的已保存 episode。")
                return None
            self.replay_buffer.pop_episode()
            print(f"✅ 已删除最近一条 episode。剩余 total={self.replay_buffer.n_episodes}")
            return None

        # ============== 开始新 episode ==============
        if state == "start":
            if self.in_recording:
                print("警告：已经在录制中，强制重新开始一个新 episode")
            print("\n>>> 开始新 episode <<<")
            self._reset_episode_buffer()
            self.in_recording = True
            self.current_backup_path = (
                self.data_dir / self.agent_name / datetime.datetime.now().strftime("%m%d_%H%M%S")
            )
            self.current_backup_path.mkdir(parents=True, exist_ok=True)

        # ============== 录制中：保存每一帧 ==============
        elif state == "save" and self.in_recording:
            print(f"[录制中] 已累积 {len(self.current_timestamps) + 1} 帧", end="\r")

            # === 1. 处理图像 ===
            for obs_key, cam_key in self.camera_mapping.items():
                if obs_key in obs:
                    img = obs[obs_key]
                    if img.dtype in (np.float32, np.float64):
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    elif img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    self.current_episode.setdefault(cam_key, []).append(img.copy())
                else:
                    dummy = np.zeros((*self.image_resolution, 3), dtype=np.uint8)
                    self.current_episode.setdefault(cam_key, []).append(dummy)

            # === 2. 处理机器人状态（关键修改）===
            # robot_joint: 强制转为 7 维
            joints = obs.get("joints", np.zeros(6, dtype=np.float32)).copy()
            if len(joints) == 6:
                joints = np.concatenate([joints, np.zeros(1, dtype=np.float32)])  # pad 到 7 维

            # gripper_position: 强制为 shape (1,) 而不是 (1,1)
            gripper = obs.get("gripper", np.array([0.0], dtype=np.float32)).flatten()

            # action: 强制为 7 维（6位姿 + 1夹爪）
            act = action.copy()
            if len(act) == 6:
                act = np.concatenate([act, gripper])
            elif len(act) != 7:
                print(f"警告：action 维度异常 {len(act)}，已强制处理为7维")
                if len(act) > 7:
                    act = act[:7]
                else:
                    pad = np.zeros(7 - len(act), dtype=np.float32)
                    act = np.concatenate([act, pad])

            # 存入缓冲
            self.current_episode.setdefault("robot_joint", []).append(joints)
            self.current_episode.setdefault("gripper_position", []).append(gripper)   # (1,)
            self.current_episode.setdefault("action", []).append(act)                 # (7,)

            self.current_timestamps.append(current_time)

            # === 3. npz 备份（调试用）===
            if self.backup_npz:
                self.backup_buffer.append({
                    "camera_0": self.current_episode["camera_0"][-1],
                    "camera_1": self.current_episode["camera_1"][-1],
                    "camera_2": self.current_episode.get("camera_2", np.zeros((*self.image_resolution, 3), dtype=np.uint8)),
                    "robot_joint": joints.copy(),
                    "gripper_position": gripper.copy(),
                    "action": act.copy(),
                })

        # ============== 结束录制并保存 ==============
        elif state == "normal" and self.in_recording:
            if not self.current_timestamps:
                print("空 episode，不保存")
            else:
                print(f"\n>>> 结束录制，保存 episode (长度: {len(self.current_timestamps)}) <<<")
                self._queue_episode_for_save()
                return None

            # 若为空 episode，直接重置
            self._reset_episode_buffer()

        elif state == "quit":
            print("\n>>> 退出程序 <<<")
            self.finalize_on_exit(reason="quit")
            return "quit"

        return None

    def _reset_episode_buffer(self):
        self.current_episode = {}
        self.current_timestamps = []
        self.backup_buffer = []
        self.in_recording = False


# ============== run_control_loop 函数保持不变 ==============
def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[DiffusionPolicySaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    """Run the main control loop."""
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored
            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)
    start_time = time.time()
    obs = env.get_obs()

    try:
        while True:
            if print_timing:
                num = time.time() - start_time
                message = f"\rTime passed: {round(num, 2)}          "
                if colors_available:
                    print(colored(message, color="white", attrs=["bold"]), end="", flush=True)
                else:
                    print(message, end="", flush=True)

            action = agent.act(obs)

            if save_interface is not None:
                result = save_interface.update(obs, action)
                if result == "quit":
                    break

            obs = env.step(action)
    except KeyboardInterrupt:
        print("\n>>> KeyboardInterrupt detected, shutting down safely... <<<")
        if save_interface is not None:
            save_interface.finalize_on_exit(reason="KeyboardInterrupt")