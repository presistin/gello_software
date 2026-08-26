"""M3025 分布力 → 6 维触觉向量（与 DP 训练/推理一致）。"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from gello.sensors.tactile_hs_reader import DEFAULT_M3025_SLOTS, SensorSlot

TACTILE_DIM = 6  # ring(Fx,Fy,Fz) + pinky(Fx,Fy,Fz)，单位 N


def distribution_to_vector6(
    points: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """
    对 77×3 分布力去零点后，各轴取最大值作为该传感器的 Fx,Fy,Fz 代表值。
    """
    delta = points - baseline
    return np.array(
        [float(delta[:, 0].max()), float(delta[:, 1].max()), float(delta[:, 2].max())],
        dtype=np.float32,
    )


def slots_to_vector6(
    per_slot_points: Tuple[np.ndarray, ...],
    baselines: dict,
    slots: Tuple[SensorSlot, ...] = DEFAULT_M3025_SLOTS,
) -> np.ndarray:
    """按 DEFAULT_M3025_SLOTS 顺序拼接为 6 维。"""
    parts = []
    for slot, pts in zip(slots, per_slot_points):
        base = baselines.get(slot.key)
        if base is None or pts is None:
            parts.append(np.zeros(3, dtype=np.float32))
        else:
            parts.append(distribution_to_vector6(pts, base))
    return np.concatenate(parts, axis=0).astype(np.float32)
