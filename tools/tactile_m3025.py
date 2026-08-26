#!/usr/bin/env python3
"""
PX-6AX GEN3 高速通信集成板 + M3025 指腹传感器 — 分布力轮询读取与接触显示。

默认读取两个已标定槽位（用户实测）：
  - 无名指近节 @ 0x2800, 77 点, 231 字节
  - 小拇指近节 @ 0x3000, 77 点, 231 字节

协议与厂商 Hand_UI.py 一致：921600 8N1，请求 55 AA，应答 AA 55，LRC 校验。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("请先安装: pip install pyserial numpy", file=sys.stderr)
    sys.exit(1)

BAUDRATE = 921600
FUNC_READ = 0x03
REQ_HEAD = b"\x55\xAA"
RESP_HEAD = b"\xAA\x55"
POINTS_PER_SENSOR = 77
BYTES_PER_SENSOR = 231  # 77 * 3


@dataclass(frozen=True)
class SensorSlot:
    key: str
    name: str
    addr: int
    num_points: int = POINTS_PER_SENSOR
    num_bytes: int = BYTES_PER_SENSOR


DEFAULT_SENSORS: Tuple[SensorSlot, ...] = (
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
    """返回 (reg_addr, payload) 或 None。"""
    if len(frame) < 9 or frame[:2] != RESP_HEAD:
        return None
    body, lrc_recv = frame[:-1], frame[-1]
    if calc_lrc(body) != lrc_recv:
        return None
    if frame[3] != FUNC_READ:
        return None
    return int.from_bytes(frame[4:6], "little"), frame[8:-1]


def parse_distribution(raw: bytes, num_points: int = POINTS_PER_SENSOR) -> np.ndarray:
    """raw: 231 字节 -> (N, 3) float32，单位 N。"""
    need = num_points * 3
    if len(raw) < need:
        raise ValueError(f"数据不足: 需要 {need} 字节, 收到 {len(raw)}")
    pts = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        o = i * 3
        b0, b1, b2 = raw[o], raw[o + 1], raw[o + 2]
        x = (b0 - 256 if b0 > 127 else b0) * 0.1
        y = (b1 - 256 if b1 > 127 else b1) * 0.1
        z = b2 * 0.1
        pts[i] = (x, y, z)
    return pts


class HighSpeedTactileBoard:
    def __init__(self, port: str, baud: int = BAUDRATE, timeout: float = 0.2):
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
        time.sleep(0.02)

        deadline = time.time() + 0.25
        buf = b""
        min_len = 8 + read_len + 1
        while time.time() < deadline:
            n = self.ser.in_waiting
            if n:
                buf += self.ser.read(n)
                if RESP_HEAD in buf:
                    start = buf.find(RESP_HEAD)
                    buf = buf[start:]
                if len(buf) >= min_len:
                    break
            time.sleep(0.001)

        if not buf:
            return None
        parsed = parse_response_frame(buf)
        if parsed is None:
            return None
        resp_addr, payload = parsed
        if resp_addr != reg_addr:
            return None
        if len(payload) < read_len:
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


def list_serial_ports() -> List[str]:
    return [p.device for p in serial.tools.list_ports.comports()]


def format_active_points(
    points: np.ndarray, mask: np.ndarray, max_show: int = 8
) -> str:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return ""
    parts = []
    for i in idx[:max_show]:
        fx, fy, fz = points[i]
        parts.append(f"p{i:02d}({fx:.1f},{fy:.1f},{fz:.1f})")
    extra = f" +{len(idx) - max_show}点" if len(idx) > max_show else ""
    return " ".join(parts) + extra


class BaselineTracker:
    """软件去零点 + 松手后缓慢跟踪 baseline，抑制残留力显示。"""

    def __init__(
        self,
        num_points: int,
        adapt_alpha: float = 0.08,
        release_ratio: float = 0.4,
    ):
        self.baseline = np.zeros((num_points, 3), dtype=np.float32)
        self.ready = False
        self.adapt_alpha = adapt_alpha
        self.release_ratio = release_ratio

    def collect(self, board: HighSpeedTactileBoard, slot: SensorSlot, sec: float, hz: float) -> bool:
        n = max(1, int(sec * hz))
        acc = np.zeros_like(self.baseline)
        got = 0
        for _ in range(n):
            pts = board.read_distribution(slot)
            if pts is not None:
                acc += pts
                got += 1
            time.sleep(1.0 / hz)
        if got == 0:
            return False
        self.baseline = acc / got
        self.ready = True
        return True

    def apply(
        self, pts: np.ndarray, contact_thresh: float
    ) -> Tuple[np.ndarray, float, int]:
        """返回 (去零点后的点, max‖ΔF‖, 接触点数)。"""
        delta = pts - self.baseline
        mag = np.linalg.norm(delta, axis=1)
        max_d = float(mag.max()) if len(mag) else 0.0
        release_thresh = contact_thresh * self.release_ratio
        # 判定为「已松手」时，让 baseline 向当前读数靠拢，吃掉弹性滞后残留
        if max_d < release_thresh:
            self.baseline = (
                (1.0 - self.adapt_alpha) * self.baseline
                + self.adapt_alpha * pts
            )
        n_contact = int((mag > contact_thresh).sum())
        return delta, max_d, n_contact


def run_poll_loop(
    board: HighSpeedTactileBoard,
    sensors: Tuple[SensorSlot, ...],
    thresh_n: float,
    hz: float,
    show_points: bool,
    use_baseline: bool,
    baseline_sec: float,
    adapt_alpha: float,
    release_ratio: float,
) -> None:
    period = 1.0 / hz
    trackers = {
        s.key: BaselineTracker(s.num_points, adapt_alpha, release_ratio)
        for s in sensors
    }
    mode = "去零点+松手跟踪" if use_baseline else "原始读数"
    print(f"轮询 {len(sensors)} 个传感器 | {mode} | 接触阈值 {thresh_n:.2f} N | {hz:.0f} Hz")
    if use_baseline:
        print(f"请保持无接触 {baseline_sec:.1f}s 采集零点…")
        for slot in sensors:
            ok = trackers[slot.key].collect(board, slot, baseline_sec, hz)
            if not ok:
                print(f"  警告: {slot.name} 零点采集失败，将用零向量")
            else:
                print(f"  {slot.name} 零点已记录")
    print("按 Ctrl+C 退出\n")

    try:
        while True:
            t0 = time.time()
            lines = []
            for slot in sensors:
                pts = board.read_distribution(slot)
                if pts is None:
                    lines.append(f"{slot.name}: [读取失败]")
                    continue
                if use_baseline:
                    tr = trackers[slot.key]
                    display, max_n, n_contact = tr.apply(pts, thresh_n)
                    mask = np.linalg.norm(display, axis=1) > thresh_n
                else:
                    display = pts
                    mag = np.linalg.norm(pts, axis=1)
                    mask = mag > thresh_n
                    n_contact = int(mask.sum())
                    max_n = float(mag.max()) if len(mag) else 0.0
                line = (
                    f"{slot.name} @0x{slot.addr:04X} | "
                    f"接触 {n_contact}/{slot.num_points} | max {max_n:.2f} N"
                )
                if show_points and n_contact > 0:
                    line += " | " + format_active_points(display, mask)
                lines.append(line)

            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(time.strftime("%H:%M:%S") + "\n")
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\n已停止。")


def pick_port(cli_port: Optional[str]) -> str:
    ports = list_serial_ports()
    if not ports:
        print("未找到串口设备。", file=sys.stderr)
        sys.exit(1)
    if cli_port:
        return cli_port
    if len(ports) == 1:
        return ports[0]
    print("可用串口:")
    for i, p in enumerate(ports, 1):
        print(f"  {i}. {p}")
    try:
        choice = int(input(f"选择 (1-{len(ports)}): "))
        return ports[choice - 1]
    except (ValueError, IndexError, EOFError):
        return ports[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M3025 @ 高速通信集成板 — 分布力轮询与接触显示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/tactile_m3025.py
  python tools/tactile_m3025.py --port /dev/ttyUSB0 --thresh 0.5
  python tools/tactile_m3025.py --thresh 0.3 --hz 15 --show-points
  python tools/tactile_m3025.py --raw -t 0.5
  python tools/tactile_m3025.py --list-ports
        """,
    )
    parser.add_argument("--port", "-p", default=None, help="串口，如 /dev/ttyUSB0")
    parser.add_argument(
        "--thresh", "-t", type=float, default=0.5,
        help="接触判定阈值 ‖ΔF‖ (N)，配合去零点使用",
    )
    parser.add_argument(
        "--hz", type=float, default=10.0,
        help="轮询频率 (Hz)，两传感器串联读取，过高可能丢包",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="显示原始读数，不做软件去零点（易有松手残留）",
    )
    parser.add_argument(
        "--baseline-sec", type=float, default=1.5,
        help="启动时空载采集零点时长 (s)，默认开启去零点",
    )
    parser.add_argument(
        "--adapt-alpha", type=float, default=0.08,
        help="松手后 baseline 跟踪速率，越大残留消得越快（0.05~0.15）",
    )
    parser.add_argument(
        "--release-ratio", type=float, default=0.4,
        help="判定松手: max‖ΔF‖ < thresh×该比例 时更新 baseline",
    )
    parser.add_argument("--show-points", action="store_true", help="打印接触测点 Fx,Fy,Fz")
    parser.add_argument("--list-ports", action="store_true", help="列出串口后退出")
    parser.add_argument("--once", action="store_true", help="只读一帧，用于调试")
    args = parser.parse_args()

    if args.list_ports:
        for p in list_serial_ports():
            print(p)
        return

    port = pick_port(args.port)
    print(f"打开串口 {port} @ {BAUDRATE}")

    board = HighSpeedTactileBoard(port)
    try:
        if args.once:
            for slot in DEFAULT_SENSORS:
                pts = board.read_distribution(slot)
                if pts is None:
                    print(f"{slot.name}: 读取失败")
                    continue
                mag = np.linalg.norm(pts, axis=1)
                print(f"\n=== {slot.name} (0x{slot.addr:04X}) ===")
                print(f"max ‖F‖ = {mag.max():.3f} N, 接触(>{args.thresh}N) = {(mag > args.thresh).sum()}")
                for i in np.where(mag > args.thresh)[0][:12]:
                    print(f"  点{i:02d}: {pts[i,0]:+.2f} {pts[i,1]:+.2f} {pts[i,2]:+.2f} N")
            return

        run_poll_loop(
            board,
            DEFAULT_SENSORS,
            args.thresh,
            args.hz,
            args.show_points,
            use_baseline=not args.raw,
            baseline_sec=args.baseline_sec,
            adapt_alpha=args.adapt_alpha,
            release_ratio=args.release_ratio,
        )
    finally:
        board.close()


if __name__ == "__main__":
    main()
