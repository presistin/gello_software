"""One safety implementation shared by GELLO collection and DP deployment.

The supervisor deliberately separates observation freshness checks from action
filtering.  Deployment checks camera freshness immediately before inference;
checking it after a long diffusion inference would reject every otherwise valid
frame.  Robot and tactile freshness are checked again at action time.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


SAFETY_MODES = ("legacy", "shadow", "enforce")
_CLOSE_EPSILON = 1e-4


class SafetyFault(RuntimeError):
    """Raised after a deployment safety fault stops further command output."""


@dataclass(frozen=True)
class SafetyConfig:
    tactile_grip_limit_n: float = 0.5
    max_joint_step_rad: float = 0.16
    max_camera_age_ms: float = 100.0
    max_tactile_age_ms: float = 120.0
    max_robot_age_ms: float = 10.0
    gripper_close_increases: bool = True
    expected_tactile_sensor_count: int = 2
    required_cameras: Tuple[str, ...] = ("wrist", "base", "camera_2")

    def __post_init__(self) -> None:
        positive_fields = {
            "tactile_grip_limit_n": self.tactile_grip_limit_n,
            "max_joint_step_rad": self.max_joint_step_rad,
            "max_camera_age_ms": self.max_camera_age_ms,
            "max_tactile_age_ms": self.max_tactile_age_ms,
            "max_robot_age_ms": self.max_robot_age_ms,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"safety.{name} must be a finite positive number")
        if int(self.expected_tactile_sensor_count) <= 0:
            raise ValueError("safety.expected_tactile_sensor_count must be positive")
        if not self.required_cameras:
            raise ValueError("safety.required_cameras must not be empty")
        if any(not str(name).strip() for name in self.required_cameras):
            raise ValueError("safety.required_cameras contains an empty role")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["required_cameras"] = list(self.required_cameras)
        return payload


@dataclass(frozen=True)
class SafetyDecision:
    safe_action: np.ndarray
    events: Tuple[Dict[str, Any], ...]

    @property
    def faulted(self) -> bool:
        return any(event.get("severity") == "fault" for event in self.events)


def normalize_safety_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SAFETY_MODES:
        raise ValueError(
            f"Invalid safety mode {mode!r}; expected one of {SAFETY_MODES}"
        )
    return normalized


def default_safety_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "safety.yaml"


def load_safety_config(path: Optional[str] = None) -> SafetyConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is in both runtime envs
        raise RuntimeError("PyYAML is required to load the safety configuration") from exc

    config_path = (
        default_safety_config_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("safety"), Mapping):
        raise ValueError(f"{config_path} must contain a top-level 'safety' mapping")

    raw = dict(payload["safety"])
    allowed = {
        "tactile_grip_limit_n",
        "max_joint_step_rad",
        "max_camera_age_ms",
        "max_tactile_age_ms",
        "max_robot_age_ms",
        "gripper_close_increases",
        "expected_tactile_sensor_count",
        "required_cameras",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown safety configuration keys: {unknown}")

    required = allowed
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            f"Safety configuration must explicitly define measured thresholds; missing: {missing}"
        )
    cameras = raw.get("required_cameras")
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
        raise ValueError("safety.required_cameras must be a list of camera roles")
    raw["required_cameras"] = tuple(str(name) for name in cameras)
    raw["expected_tactile_sensor_count"] = int(
        raw["expected_tactile_sensor_count"]
    )
    if not isinstance(raw["gripper_close_increases"], bool):
        raise ValueError("safety.gripper_close_increases must be true or false")
    return SafetyConfig(**raw)


def _copy_action(action: Any) -> np.ndarray:
    out = np.asarray(action).copy()
    if not np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float64)
    return out


def apply_joint_and_gripper_limits(
    action: Any,
    current_state: Any,
    max_joint_step_rad: float,
) -> Tuple[np.ndarray, Tuple[Dict[str, Any], ...]]:
    """Apply the legacy deployment joint/gripper bounds as a pure function."""

    out = _copy_action(action)
    current = np.asarray(current_state).reshape(-1)
    events = []
    if out.ndim != 1 or out.shape[0] < 7 or current.shape[0] < 7:
        raise ValueError("action and current_state must each contain 7 dimensions")

    arm_before = out[:6].copy()
    requested_delta = arm_before - current[:6]
    delta = np.clip(
        requested_delta,
        -float(max_joint_step_rad),
        float(max_joint_step_rad),
    )
    out[:6] = current[:6] + delta
    clipped_indices = np.flatnonzero(
        np.abs(requested_delta) > float(max_joint_step_rad)
    ).tolist()
    if clipped_indices:
        events.append(
            {
                "code": "joint_step_clipped",
                "severity": "limit",
                "joint_indices": clipped_indices,
                "max_joint_step_rad": float(max_joint_step_rad),
            }
        )

    gripper_before = float(out[6])
    out[6] = float(np.clip(gripper_before, 0.0, 1.0))
    if gripper_before != float(out[6]):
        events.append(
            {
                "code": "gripper_range_clipped",
                "severity": "limit",
                "before": gripper_before,
                "after": float(out[6]),
            }
        )
    return out, tuple(events)


def limit_gripper_by_force(
    action: Any,
    current_gripper: float,
    force_n: float,
    force_limit_n: float,
    close_increases_command: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Freeze only further closing when the measured force reaches the limit."""

    out = _copy_action(action)
    if out.ndim != 1 or out.shape[0] < 7:
        return out, False
    target = float(out[-1])
    current = float(current_gripper)
    if close_increases_command:
        wants_close_more = target > current + _CLOSE_EPSILON
    else:
        wants_close_more = target < current - _CLOSE_EPSILON
    limited = float(force_n) >= float(force_limit_n) and wants_close_more
    if limited:
        out[-1] = current
    return out, limited


def _timing_age_ms(
    timing: Optional[Mapping[str, Any]], now_ns: int
) -> Optional[float]:
    if not isinstance(timing, Mapping):
        return None
    source_ns = timing.get("source_monotonic_ns")
    if isinstance(source_ns, (int, np.integer)):
        return max(0.0, (int(now_ns) - int(source_ns)) / 1e6)
    age_ms = timing.get("age_ms")
    if isinstance(age_ms, (int, float, np.integer, np.floating)):
        return float(age_ms)
    return None


def _fault(code: str, **fields: Any) -> Dict[str, Any]:
    return {"code": code, "severity": "fault", **fields}


def action_delta_metrics(
    action: Optional[Any], reference: Optional[Any]
) -> Optional[Dict[str, Any]]:
    """Describe an action jump without changing either input."""

    if action is None or reference is None:
        return None
    action_array = np.asarray(action, dtype=np.float64).reshape(-1)
    reference_array = np.asarray(reference, dtype=np.float64).reshape(-1)
    if (
        action_array.shape != reference_array.shape
        or action_array.size == 0
        or not np.all(np.isfinite(action_array))
        or not np.all(np.isfinite(reference_array))
    ):
        return None
    delta = action_array - reference_array
    return {
        "delta": delta.tolist(),
        "max_abs": float(np.max(np.abs(delta))),
        "l2": float(np.linalg.norm(delta)),
    }


def check_robot_health(
    ping_response: Any,
    require_health_snapshot: bool = True,
) -> Tuple[Dict[str, Any], ...]:
    """Classify an existing robot ping response without guessing mode codes."""

    events = []
    if not isinstance(ping_response, Mapping):
        return (_fault("robot_ping_invalid"),)
    if ping_response.get("server_alive") is not True:
        events.append(_fault("robot_server_unavailable"))

    health = ping_response.get("robot_health")
    if not isinstance(health, Mapping):
        severity = "fault" if require_health_snapshot else "warning"
        events.append(
            {
                "code": "robot_health_missing",
                "severity": severity,
            }
        )
        return tuple(events)

    if health.get("emergency_stopped") is True:
        events.append(_fault("robot_emergency_stopped"))
    if health.get("protective_stopped") is True:
        events.append(_fault("robot_protective_stopped"))
    if health.get("robot_connected") is not True:
        events.append(_fault("robot_disconnected"))
    if health.get("operational") is not True:
        events.append(_fault("robot_not_operational"))
    if health.get("gripper_process_alive") is False:
        events.append(_fault("gripper_process_dead"))
    errors = health.get("errors")
    if isinstance(errors, Mapping) and errors:
        events.append(
            _fault(
                "robot_health_read_error",
                health_errors=dict(errors),
            )
        )
    return tuple(events)


class SafetySupervisor:
    def __init__(self, config: SafetyConfig):
        self.config = config

    def check_observation_freshness(
        self,
        observation_timing: Optional[Mapping[str, Any]],
        tactile_sample: Optional[Any],
        now_ns: Optional[int] = None,
        check_cameras: bool = True,
        check_robot: bool = True,
        check_tactile: bool = True,
    ) -> Tuple[Dict[str, Any], ...]:
        """Return structured fault events without mutating observations."""

        evaluated_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        events = []

        if check_cameras:
            camera_timing = (
                observation_timing.get("cameras")
                if isinstance(observation_timing, Mapping)
                else None
            )
            for role in self.config.required_cameras:
                timing = (
                    camera_timing.get(role)
                    if isinstance(camera_timing, Mapping)
                    else None
                )
                age_ms = _timing_age_ms(timing, evaluated_ns)
                if age_ms is None:
                    events.append(_fault("camera_timing_missing", camera=role))
                elif age_ms > self.config.max_camera_age_ms:
                    events.append(
                        _fault(
                            "camera_stale",
                            camera=role,
                            age_ms=age_ms,
                            limit_ms=self.config.max_camera_age_ms,
                        )
                    )

        if check_robot:
            robot_timing = (
                observation_timing.get("robot")
                if isinstance(observation_timing, Mapping)
                else None
            )
            age_ms = _timing_age_ms(robot_timing, evaluated_ns)
            if age_ms is None:
                events.append(_fault("robot_timing_missing"))
            elif age_ms > self.config.max_robot_age_ms:
                events.append(
                    _fault(
                        "robot_stale",
                        age_ms=age_ms,
                        limit_ms=self.config.max_robot_age_ms,
                    )
                )

        if check_tactile:
            if tactile_sample is None:
                events.append(_fault("tactile_sample_missing"))
            else:
                source_ns = getattr(tactile_sample, "monotonic_ns", None)
                if not isinstance(source_ns, (int, np.integer)):
                    events.append(_fault("tactile_timing_missing"))
                else:
                    age_ms = max(0.0, (evaluated_ns - int(source_ns)) / 1e6)
                    if age_ms > self.config.max_tactile_age_ms:
                        events.append(
                            _fault(
                                "tactile_stale",
                                age_ms=age_ms,
                                limit_ms=self.config.max_tactile_age_ms,
                            )
                        )
                valid_count = getattr(tactile_sample, "valid_sensor_count", None)
                if valid_count != self.config.expected_tactile_sensor_count:
                    events.append(
                        _fault(
                            "tactile_sensor_count_invalid",
                            valid_sensor_count=valid_count,
                            expected_sensor_count=(
                                self.config.expected_tactile_sensor_count
                            ),
                        )
                    )
                force_n = getattr(tactile_sample, "max_force_n", None)
                if not isinstance(
                    force_n, (int, float, np.integer, np.floating)
                ) or not math.isfinite(float(force_n)) or float(force_n) < 0:
                    events.append(_fault("tactile_force_invalid", force_n=force_n))
        return tuple(events)

    def filter_action(
        self,
        action: Any,
        current_state: Any,
        tactile_sample: Optional[Any],
        observation_timing: Optional[Mapping[str, Any]] = None,
        now_ns: Optional[int] = None,
        check_cameras: bool = False,
        check_robot: bool = True,
        check_tactile: bool = True,
    ) -> SafetyDecision:
        """Return a safe command and all limit/fault events for one action."""

        current = np.asarray(current_state).reshape(-1)
        action_array = np.asarray(action)
        shape_valid = (
            current.shape[0] == 7
            and action_array.ndim == 1
            and action_array.shape[0] == 7
        )
        values_valid = shape_valid and bool(
            np.all(np.isfinite(current[:7]))
            and np.all(np.isfinite(action_array[:7]))
        )
        if not values_valid:
            if current.shape[0] >= 7 and np.all(np.isfinite(current[:7])):
                hold = current.copy()
            else:
                hold = np.zeros(7, dtype=np.float64)
            return SafetyDecision(
                safe_action=hold,
                events=(
                    _fault(
                        "action_or_state_invalid",
                        action_shape=list(action_array.shape),
                        current_shape=list(current.shape),
                    ),
                ),
            )

        freshness_events = self.check_observation_freshness(
            observation_timing=observation_timing,
            tactile_sample=tactile_sample,
            now_ns=now_ns,
            check_cameras=check_cameras,
            check_robot=check_robot,
            check_tactile=check_tactile,
        )
        if freshness_events:
            return SafetyDecision(
                safe_action=current[:7].copy(),
                events=freshness_events,
            )

        safe_action, bound_events = apply_joint_and_gripper_limits(
            action_array[:7],
            current[:7],
            self.config.max_joint_step_rad,
        )
        force_n = float(getattr(tactile_sample, "max_force_n", 0.0))
        force_limited_action, force_limited = limit_gripper_by_force(
            safe_action,
            current_gripper=float(current[6]),
            force_n=force_n,
            force_limit_n=self.config.tactile_grip_limit_n,
            close_increases_command=self.config.gripper_close_increases,
        )
        events = list(bound_events)
        if force_limited:
            events.append(
                {
                    "code": "tactile_gripper_frozen",
                    "severity": "limit",
                    "force_n": force_n,
                    "limit_n": self.config.tactile_grip_limit_n,
                    "held_gripper": float(current[6]),
                }
            )
        return SafetyDecision(
            safe_action=force_limited_action,
            events=tuple(events),
        )
