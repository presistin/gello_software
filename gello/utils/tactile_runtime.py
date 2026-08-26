"""
高速板触觉运行时：单串口共享给采集存盘与夹爪限力。

用法：run_env 创建一个 TactileRuntime，传给 run_control_loop 与 TactileGripperLimiter。
后台线程持续轮询时，主控制循环 read_sample() 为非阻塞取缓存。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from gello.sensors.tactile_hs_reader import (
    DEFAULT_M3025_SLOTS,
    HighSpeedTactileBoard,
    SensorSlot,
)
from gello.sensors.tactile_vector import TACTILE_DIM, distribution_to_vector6


@dataclass
class TactileSample:
    vector6: np.ndarray
    max_force_n: float


class TactileRuntime:
    def __init__(
        self,
        port: str,
        baseline_sec: float = 1.5,
        poll_hz: float = 20.0,
        adapt_alpha: float = 0.08,
        release_ratio: float = 0.4,
        force_limit_n: float = 0.5,
        sensors: Tuple[SensorSlot, ...] = DEFAULT_M3025_SLOTS,
        background_poll_hz: float = 25.0,
    ):
        self.sensors = sensors
        self.adapt_alpha = adapt_alpha
        self.release_ratio = release_ratio
        self.force_limit_n = force_limit_n
        self._board = HighSpeedTactileBoard(port)
        self._baselines: dict = {s.key: None for s in sensors}

        self._lock = threading.Lock()
        self._latest: Optional[TactileSample] = None
        self._stop_poll = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None

        print(f"[Tactile] 零点采集 {baseline_sec:.1f}s，请勿触碰传感器…")
        n = max(1, int(baseline_sec * poll_hz))
        acc = {s.key: np.zeros((s.num_points, 3), np.float32) for s in sensors}
        cnt = {s.key: 0 for s in sensors}
        for _ in range(n):
            for slot in sensors:
                pts = self._board.read_distribution(slot)
                if pts is not None:
                    acc[slot.key] += pts
                    cnt[slot.key] += 1
            time.sleep(1.0 / poll_hz)
        for slot in sensors:
            if cnt[slot.key] > 0:
                self._baselines[slot.key] = acc[slot.key] / cnt[slot.key]
                print(f"  [Tactile] {slot.name} 零点 OK")
            else:
                print(f"  [Tactile] 警告: {slot.name} 零点采集失败")
        print(f"[Tactile] 已连接 {port}，向量维度 {TACTILE_DIM}")

        if background_poll_hz > 0:
            self._start_background_poll(background_poll_hz)

    def _start_background_poll(self, hz: float) -> None:
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(hz,),
            daemon=True,
            name="TactilePoll",
        )
        self._poll_thread.start()
        print(f"[Tactile] 后台轮询已启动 (~{hz:.0f} Hz)，主循环不再阻塞串口")

    def _poll_loop(self, hz: float) -> None:
        period = 1.0 / hz
        while not self._stop_poll.is_set():
            t0 = time.time()
            sample = self._read_sample_blocking()
            with self._lock:
                self._latest = sample
            elapsed = time.time() - t0
            sleep_t = period - elapsed
            if sleep_t > 0:
                self._stop_poll.wait(sleep_t)

    def close(self) -> None:
        self._stop_poll.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
        self._board.close()

    def _maybe_adapt_baseline(self, slot: SensorSlot, pts: np.ndarray, delta: np.ndarray) -> None:
        release_thresh = self.force_limit_n * self.release_ratio
        if float(np.linalg.norm(delta, axis=1).max()) < release_thresh:
            base = self._baselines[slot.key]
            if base is not None:
                self._baselines[slot.key] = (
                    (1.0 - self.adapt_alpha) * base + self.adapt_alpha * pts
                ).astype(np.float32)

    def _read_sample_blocking(self) -> TactileSample:
        """读所有传感器一次，同时得到 6 维向量与 max‖ΔF‖。"""
        out = np.zeros(TACTILE_DIM, dtype=np.float32)
        peak = 0.0
        for i, slot in enumerate(self.sensors):
            pts = self._board.read_distribution(slot)
            base = self._baselines.get(slot.key)
            if pts is None or base is None:
                continue
            delta = pts - base
            self._maybe_adapt_baseline(slot, pts, delta)
            peak = max(peak, float(np.linalg.norm(delta, axis=1).max()))
            out[i * 3 : (i + 1) * 3] = distribution_to_vector6(pts, base)
        return TactileSample(vector6=out, max_force_n=peak)

    def read_sample(self) -> TactileSample:
        """主循环调用：有后台线程时取缓存（非阻塞），否则同步读一次。"""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            with self._lock:
                if self._latest is not None:
                    return TactileSample(
                        vector6=self._latest.vector6.copy(),
                        max_force_n=self._latest.max_force_n,
                    )
        return self._read_sample_blocking()

    def read_vector6(self) -> np.ndarray:
        return self.read_sample().vector6

    def get_max_force_n(self) -> float:
        """兼容旧调用；优先用缓存，避免重复读串口。"""
        return self.read_sample().max_force_n
