"""高速通信集成板 + M3025 分布力读取（供采集/夹爪限力共用）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import serial
except ImportError:
    serial = None  # type: ignore

BAUDRATE = 921600
FUNC_READ = 0x03
REQ_HEAD = b"\x55\xAA"
RESP_HEAD = b"\xAA\x55"
POINTS_PER_SENSOR = 77
BYTES_PER_SENSOR = 231


@dataclass(frozen=True)
class SensorSlot:
    key: str
    name: str
    addr: int
    num_points: int = POINTS_PER_SENSOR
    num_bytes: int = BYTES_PER_SENSOR


DEFAULT_M3025_SLOTS: Tuple[SensorSlot, ...] = (
    SensorSlot("ring_prox", "无名指近节", 0x2800),
    SensorSlot("pinky_prox", "小拇指近节", 0x3000),
)


def calc_lrc(data: bytes) -> int:
    s = sum(data) & 0xFF
    return ((~s) + 1) & 0xFF


def build_read_frame(reg_addr: int, read_len: int) -> bytes:
    body = (
        REQ_HEAD
        + b"\x00"
        + bytes([FUNC_READ])
        + reg_addr.to_bytes(2, "little")
        + read_len.to_bytes(2, "little")
    )
    return body + bytes([calc_lrc(body)])


def parse_response_frame(frame: bytes) -> Optional[Tuple[int, bytes]]:
    if len(frame) < 9 or frame[:2] != RESP_HEAD:
        return None
    if calc_lrc(frame[:-1]) != frame[-1] or frame[3] != FUNC_READ:
        return None
    return int.from_bytes(frame[4:6], "little"), frame[8:-1]


def parse_distribution(raw: bytes, num_points: int = POINTS_PER_SENSOR) -> np.ndarray:
    pts = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        o = i * 3
        b0, b1, b2 = raw[o], raw[o + 1], raw[o + 2]
        pts[i] = (
            (b0 - 256 if b0 > 127 else b0) * 0.1,
            (b1 - 256 if b1 > 127 else b1) * 0.1,
            b2 * 0.1,
        )
    return pts


class HighSpeedTactileBoard:
    def __init__(
        self,
        port: str,
        baud: int = BAUDRATE,
        timeout: float = 0.08,
        post_write_sleep: float = 0.005,
        read_deadline: float = 0.08,
    ):
        self._post_write_sleep = post_write_sleep
        self._read_deadline = read_deadline
        if serial is None:
            raise ImportError("pyserial required: pip install pyserial")
        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=0.5,
            inter_byte_timeout=0.001,
        )

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    def read_register(self, reg_addr: int, read_len: int) -> Optional[bytes]:
        self.ser.reset_input_buffer()
        self.ser.write(build_read_frame(reg_addr, read_len))
        time.sleep(self._post_write_sleep)
        deadline = time.time() + self._read_deadline
        buf = b""
        min_len = 8 + read_len + 1
        while time.time() < deadline:
            if self.ser.in_waiting:
                buf += self.ser.read(self.ser.in_waiting)
                if RESP_HEAD in buf:
                    buf = buf[buf.find(RESP_HEAD) :]
                if len(buf) >= min_len:
                    break
            time.sleep(0.001)
        if not buf:
            return None
        parsed = parse_response_frame(buf)
        if parsed is None:
            return None
        resp_addr, payload = parsed
        if resp_addr != reg_addr or len(payload) < read_len:
            return None
        return payload[:read_len]

    def read_distribution(self, slot: SensorSlot) -> Optional[np.ndarray]:
        raw = self.read_register(slot.addr, slot.num_bytes)
        if raw is None:
            return None
        try:
            return parse_distribution(raw, slot.num_points)
        except ValueError:
            return None

    def read_max_delta_force(
        self,
        slots: Tuple[SensorSlot, ...],
        baselines: dict,
    ) -> float:
        """各传感器 max‖F-baseline‖ 的最大值 (N)。"""
        peak = 0.0
        for slot in slots:
            pts = self.read_distribution(slot)
            if pts is None:
                continue
            base = baselines.get(slot.key)
            if base is not None and base.shape == pts.shape:
                delta = pts - base
            else:
                delta = pts
            m = float(np.linalg.norm(delta, axis=1).max())
            peak = max(peak, m)
        return peak
