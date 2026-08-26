"""串口自动选择：区分 GELLO (FTDI/Dynamixel) 与 Paxini 触觉适配器。"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Set


def port_aliases(port: str) -> Set[str]:
    """同一物理设备可能对应 by-id 路径与 /dev/ttyACM*。"""
    aliases = {port}
    try:
        real = os.path.realpath(port)
        aliases.add(real)
        for bid in glob.glob("/dev/serial/by-id/*"):
            try:
                if os.path.realpath(bid) == real:
                    aliases.add(bid)
            except OSError:
                continue
        for tty in glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"):
            try:
                if os.path.realpath(tty) == real:
                    aliases.add(tty)
            except OSError:
                continue
    except OSError:
        pass
    return aliases


def is_tactile_adapter_path(path: str) -> bool:
    low = path.lower()
    return any(
        key in low
        for key in ("paxini", "pxsr", "tactile", "adapter_ec")
    )


def resolve_gello_port(
    explicit: Optional[str] = None,
    exclude_ports: Optional[List[str]] = None,
) -> str:
    """
    自动选择 GELLO 串口：仅使用 PORT_CONFIG_MAP 中已配置的 FTDI by-id 路径，
    并排除触觉板占用的端口。
    """
    from gello.agents.gello_agent import PORT_CONFIG_MAP

    if explicit:
        if explicit not in PORT_CONFIG_MAP:
            known = "\n  ".join(sorted(PORT_CONFIG_MAP.keys()))
            raise ValueError(
                f"GELLO 端口 {explicit} 不在 PORT_CONFIG_MAP 中。\n"
                f"请用 --gello-port 指定下列之一，或把该端口配置加入 gello_agent.py：\n  {known}"
            )
        return explicit

    excluded: Set[str] = set()
    for p in exclude_ports or []:
        excluded |= port_aliases(p)

    candidates: List[str] = []
    for path in sorted(glob.glob("/dev/serial/by-id/*")):
        if path in excluded or is_tactile_adapter_path(path):
            continue
        if path in PORT_CONFIG_MAP:
            candidates.append(path)

    if not candidates:
        raise ValueError(
            "未找到 GELLO 串口。请连接 FTDI 转换器并用 "
            "--gello-port /dev/serial/by-id/usb-FTDI_... 指定 "
            "(见 gello/agents/gello_agent.py 中 PORT_CONFIG_MAP)。"
        )

    if len(candidates) == 1:
        return candidates[0]

    print("检测到多个 GELLO 串口，请选择：")
    for i, p in enumerate(candidates, 1):
        print(f"  {i}. {p}")
    try:
        choice = int(input(f"选择 (1-{len(candidates)}): ")) - 1
        if 0 <= choice < len(candidates):
            return candidates[choice]
    except (ValueError, EOFError):
        pass
    print(f"输入无效，使用 {candidates[0]}")
    return candidates[0]


def resolve_tactile_port(
    explicit: Optional[str] = None,
    exclude_ports: Optional[List[str]] = None,
) -> str:
    """自动选择触觉高速板串口，优先 Paxini / ttyACM。"""
    if explicit:
        return explicit

    excluded: Set[str] = set()
    for p in exclude_ports or []:
        excluded |= port_aliases(p)

    candidates: List[str] = []
    for path in sorted(glob.glob("/dev/serial/by-id/*")):
        if path in excluded:
            continue
        if is_tactile_adapter_path(path):
            candidates.append(path)

    for path in sorted(glob.glob("/dev/ttyACM*")):
        if path in excluded:
            continue
        if path not in candidates:
            candidates.append(path)

    if not candidates:
        raise ValueError(
            "未找到触觉串口。请用 --tactile-port /dev/ttyACM0 指定。"
        )
    if len(candidates) == 1:
        return candidates[0]
    print("检测到多个触觉候选串口：")
    for i, p in enumerate(candidates, 1):
        print(f"  {i}. {p}")
    try:
        choice = int(input(f"选择 (1-{len(candidates)}): ")) - 1
        if 0 <= choice < len(candidates):
            return candidates[choice]
    except (ValueError, EOFError):
        pass
    return candidates[0]
