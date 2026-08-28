"""
采集时用触觉限制从臂夹爪继续闭合，避免主臂继续握紧导致主从位姿失配卡死。

与 TactileRuntime 共用同一串口，勿重复打开 port。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gello.safety.safety_supervisor import limit_gripper_by_force
from gello.utils.tactile_runtime import TactileRuntime


class TactileGripperLimiter:
    def __init__(
        self,
        runtime: TactileRuntime,
        force_limit_n: Optional[float] = None,
        close_increases_command: bool = True,
    ):
        self.runtime = runtime
        self.force_limit_n = (
            force_limit_n if force_limit_n is not None else runtime.force_limit_n
        )
        self.close_increases_command = close_increases_command
        self._limited_last = False
        print(
            f"[TactileGripper] 限力已启用 | 阈值 {self.force_limit_n:.2f} N | "
            f"闭合方向: {'指令增大' if close_increases_command else '指令减小'}"
        )

    def apply_limit(self, action: np.ndarray, obs: dict) -> np.ndarray:
        out = np.array(action, dtype=np.float64, copy=True)
        if out.shape[0] < 7:
            return out

        leader_g = float(out[-1])
        follower_g = float(np.asarray(obs.get("gripper", [0.0])).flatten()[0])
        # 优先用主循环已读到的力值，避免每圈再读一遍串口
        if "tactile_max_force_n" in obs:
            force_n = float(obs["tactile_max_force_n"])
        else:
            force_n = self.runtime.get_max_force_n()

        out, limited = limit_gripper_by_force(
            out,
            current_gripper=follower_g,
            force_n=force_n,
            force_limit_n=self.force_limit_n,
            close_increases_command=self.close_increases_command,
        )

        if limited:
            if not self._limited_last:
                print(
                    f"\n[TactileGripper] 达到 {force_n:.2f} N >= {self.force_limit_n:.2f} N，"
                    f"从臂夹爪保持 {follower_g:.3f}（主臂 {leader_g:.3f}）"
                )
            self._limited_last = True
        else:
            self._limited_last = False
        return out

    def close(self) -> None:
        pass  # 串口由 TactileRuntime 统一关闭
