"""Deterministic RealSense role-to-serial configuration.

The configuration path is optional on purpose.  When it is omitted callers can
keep the historical enumeration-order behaviour while a deployment is being
migrated and verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class CameraAssignment:
    role: str
    serial: str
    port: int
    width: int = 640
    height: int = 480
    fps: int = 30
    flip: bool = False
    configured: bool = False


_ALLOWED_CAMERA_KEYS = {
    "serial",
    "port",
    "width",
    "height",
    "fps",
    "flip",
    "required",
}


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - both supported envs have PyYAML
        raise RuntimeError(
            "Camera YAML was requested but PyYAML is not installed. "
            "Install PyYAML or omit --camera-config to retain legacy ordering."
        ) from exc

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Camera config must be a mapping: {path}")
    return payload


def resolve_camera_assignments(
    device_ids: Sequence[str],
    *,
    config_path: Optional[str] = None,
    start_port: int = 5000,
) -> List[CameraAssignment]:
    """Resolve physical serials to stable roles and ports.

    ``config_path=None`` is the backward-compatible path: devices retain the
    enumeration order returned by ``get_device_ids`` and receive consecutive
    ports beginning at ``start_port``.
    """

    discovered = [str(device_id) for device_id in device_ids]
    if len(discovered) != len(set(discovered)):
        raise RuntimeError(f"Duplicate RealSense serials detected: {discovered}")

    if config_path is None:
        return [
            CameraAssignment(
                role=f"camera_{idx}",
                serial=serial,
                port=start_port + idx,
                configured=False,
            )
            for idx, serial in enumerate(discovered)
        ]

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Camera config does not exist: {path}")
    payload = _load_yaml(path)
    cameras = payload.get("cameras")
    if not isinstance(cameras, Mapping) or not cameras:
        raise ValueError(f"Camera config needs a non-empty 'cameras' mapping: {path}")

    assignments: List[CameraAssignment] = []
    configured_serials = set()
    configured_ports = set()
    for idx, (role_value, raw_entry) in enumerate(cameras.items()):
        role = str(role_value).strip()
        if not role:
            raise ValueError(f"Camera role at index {idx} is empty: {path}")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Camera role '{role}' must map to an object: {path}")
        unknown = set(raw_entry) - _ALLOWED_CAMERA_KEYS
        if unknown:
            raise ValueError(
                f"Camera role '{role}' has unknown keys {sorted(unknown)}: {path}"
            )

        raw_serial = raw_entry.get("serial")
        serial = "" if raw_serial is None else str(raw_serial).strip()
        required = bool(raw_entry.get("required", True))
        if not serial or serial.startswith("<"):
            raise ValueError(
                f"Camera role '{role}' needs a real serial; placeholders are rejected: {path}"
            )
        if serial in configured_serials:
            raise ValueError(f"Camera serial '{serial}' is assigned more than once: {path}")

        port = int(raw_entry.get("port", start_port + idx))
        if not 1 <= port <= 65535:
            raise ValueError(f"Camera role '{role}' has invalid port {port}: {path}")
        if port in configured_ports:
            raise ValueError(f"Camera port {port} is assigned more than once: {path}")

        # Reserve configured identities even when an optional camera is absent,
        # so a typo cannot silently create duplicate role/port definitions.
        configured_serials.add(serial)
        configured_ports.add(port)

        if serial not in discovered:
            if required:
                raise RuntimeError(
                    f"Required RealSense role '{role}' serial '{serial}' is missing. "
                    f"Detected serials: {discovered}"
                )
            continue

        width = int(raw_entry.get("width", 640))
        height = int(raw_entry.get("height", 480))
        fps = int(raw_entry.get("fps", 30))
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError(
                f"Camera role '{role}' needs positive width/height/fps: {path}"
            )

        assignments.append(
            CameraAssignment(
                role=role,
                serial=serial,
                port=port,
                width=width,
                height=height,
                fps=fps,
                flip=bool(raw_entry.get("flip", False)),
                configured=True,
            )
        )

    if not assignments:
        raise RuntimeError(f"Camera config resolved no connected devices: {path}")
    return assignments


def format_camera_assignment(assignment: CameraAssignment) -> str:
    source = "serial-config" if assignment.configured else "legacy-enumeration"
    return (
        f"role={assignment.role} serial={assignment.serial} port={assignment.port} "
        f"mode={source} stream={assignment.width}x{assignment.height}@{assignment.fps} "
        f"flip={assignment.flip}"
    )
