#!/usr/bin/env python3
"""Read-only summary for GELLO collection/deployment JSONL runtime logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _statistics(values: Sequence[float]) -> Dict[str, float]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def _append_number(series: Dict[str, List[float]], name: str, value: Any) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            series[name].append(number)


def _numeric_vector(value: Any) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    numbers = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            return []
        number = float(item)
        if not math.isfinite(number):
            return []
        numbers.append(number)
    return numbers


def summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    event_counts = Counter(str(record.get("event", "<missing>")) for record in records)
    monotonic_values = [
        record.get("monotonic_ns")
        for record in records
        if isinstance(record.get("monotonic_ns"), int)
    ]
    monotonic_violations = sum(
        current < previous
        for previous, current in zip(monotonic_values, monotonic_values[1:])
    )

    series: Dict[str, List[float]] = defaultdict(list)
    previous_event_time: Dict[str, int] = {}
    observation_timing_records = 0
    tactile_timing_records = 0
    safety_event_counts: Counter = Counter()
    safety_decision_records = 0
    safety_fault_records = 0
    legacy_safety_mismatch_records = 0
    sent_matches_legacy_records = 0
    sent_matches_safety_records = 0
    health_event_counts: Counter = Counter()
    health_check_records = 0
    health_fault_records = 0
    policy_inference_records = 0
    policy_action_trace_records = 0
    complete_policy_action_trace_records = 0
    chunk_end_records = 0
    chunk_fault_abort_records = 0
    chunks_with_abort_reason = 0
    chunks_completed_all_planned = 0
    single_step_chunks = 0
    multi_step_chunks = 0
    planned_chunk_steps = 0
    executed_chunk_steps = 0

    for record in records:
        event = str(record.get("event", "<missing>"))
        monotonic_ns = record.get("monotonic_ns")
        if isinstance(monotonic_ns, int) and event in {
            "idle_observation",
            "teleop_step",
            "policy_step",
        }:
            previous = previous_event_time.get(event)
            if previous is not None:
                _append_number(
                    series,
                    f"{event}.event_interval_ms",
                    (monotonic_ns - previous) / 1e6,
                )
            previous_event_time[event] = monotonic_ns

        for field in ("loop_duration_ms", "step_duration_ms", "duration_ms"):
            _append_number(series, f"{event}.{field}", record.get(field))

        if event == "policy_inference":
            policy_inference_records += 1
            _append_number(
                series,
                "deployment.requested_execution_count",
                record.get("requested_execution_count"),
            )
            for field_name in ("chunk_boundary", "first_action_vs_current"):
                metrics = record.get(field_name)
                if not isinstance(metrics, Mapping):
                    continue
                _append_number(
                    series,
                    f"deployment.{field_name}.max_abs",
                    metrics.get("max_abs"),
                )
                _append_number(
                    series,
                    f"deployment.{field_name}.l2",
                    metrics.get("l2"),
                )

        if event == "robot_health_check":
            health_check_records += 1
            if record.get("faulted") is True:
                health_fault_records += 1
            health_events = record.get("health_events")
            if isinstance(health_events, Sequence) and not isinstance(
                health_events, (str, bytes)
            ):
                for health_event in health_events:
                    if isinstance(health_event, Mapping):
                        health_event_counts[
                            str(health_event.get("code", "<missing>"))
                        ] += 1

        if event in {"policy_chunk_end", "policy_chunk_abort"}:
            if event == "policy_chunk_end":
                chunk_end_records += 1
            else:
                chunk_fault_abort_records += 1
            if record.get("abort_reason") is not None:
                chunks_with_abort_reason += 1
            planned = record.get("planned_steps")
            if isinstance(planned, int) and not isinstance(planned, bool):
                planned_chunk_steps += planned
                if planned == 1:
                    single_step_chunks += 1
                elif planned > 1:
                    multi_step_chunks += 1
            executed = record.get("executed_steps")
            if isinstance(executed, int) and not isinstance(executed, bool):
                executed_chunk_steps += executed
                if (
                    isinstance(planned, int)
                    and not isinstance(planned, bool)
                    and executed == planned
                    and record.get("abort_reason") is None
                ):
                    chunks_completed_all_planned += 1

        observation = record.get("observation_timing")
        if isinstance(observation, Mapping):
            observation_timing_records += 1
            _append_number(
                series,
                "observation.observation_span_ms",
                observation.get("observation_span_ms"),
            )
            robot = observation.get("robot")
            if isinstance(robot, Mapping):
                for field in ("age_ms", "client_receive_age_ms", "round_trip_ms"):
                    _append_number(series, f"robot.{field}", robot.get(field))
            cameras = observation.get("cameras")
            if isinstance(cameras, Mapping):
                for camera_name, timing in cameras.items():
                    if not isinstance(timing, Mapping):
                        continue
                    for field in ("age_ms", "client_receive_age_ms", "round_trip_ms"):
                        _append_number(
                            series,
                            f"camera.{camera_name}.{field}",
                            timing.get(field),
                        )

        tactile = record.get("tactile_timing")
        if isinstance(tactile, Mapping):
            tactile_timing_records += 1
            _append_number(series, "tactile.age_ms", tactile.get("age_ms"))
            _append_number(
                series,
                "tactile.valid_sensor_count",
                tactile.get("valid_sensor_count"),
            )

        safety_events = record.get("safety_events")
        if isinstance(safety_events, Sequence) and not isinstance(
            safety_events, (str, bytes)
        ):
            record_faulted = False
            for safety_event in safety_events:
                if not isinstance(safety_event, Mapping):
                    continue
                safety_event_counts[str(safety_event.get("code", "<missing>"))] += 1
                if safety_event.get("severity") == "fault":
                    record_faulted = True
            if record_faulted:
                safety_fault_records += 1

        legacy_action = _numeric_vector(record.get("legacy_action"))
        safety_action = _numeric_vector(record.get("safety_action"))
        sent_action = _numeric_vector(
            record.get("executed_action", record.get("sent_command"))
        )
        if event == "policy_step":
            policy_action_trace_records += 1
            raw_action = _numeric_vector(record.get("raw_policy_action"))
            if raw_action and safety_action and sent_action:
                complete_policy_action_trace_records += 1
        if (
            event in {"teleop_step", "policy_step"}
            and legacy_action
            and safety_action
            and len(legacy_action) == len(safety_action)
        ):
            safety_decision_records += 1
            max_abs = max(
                abs(legacy - safe)
                for legacy, safe in zip(legacy_action, safety_action)
            )
            _append_number(series, "safety.legacy_vs_safety_max_abs", max_abs)
            if max_abs > 1e-9:
                legacy_safety_mismatch_records += 1
            if sent_action and len(sent_action) == len(legacy_action):
                if max(
                    abs(sent - legacy)
                    for sent, legacy in zip(sent_action, legacy_action)
                ) <= 1e-9:
                    sent_matches_legacy_records += 1
                if max(
                    abs(sent - safe)
                    for sent, safe in zip(sent_action, safety_action)
                ) <= 1e-9:
                    sent_matches_safety_records += 1

    dropped_events = sum(
        int(record.get("dropped_events", 0))
        for record in records
        if record.get("event") == "run_end"
        and isinstance(record.get("dropped_events", 0), int)
    )
    return {
        "record_count": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "monotonic_non_decreasing": monotonic_violations == 0,
        "monotonic_violations": monotonic_violations,
        "dropped_events": dropped_events,
        "timestamp_coverage": {
            "observation_timing_records": observation_timing_records,
            "tactile_timing_records": tactile_timing_records,
        },
        "safety": {
            "event_counts": dict(sorted(safety_event_counts.items())),
            "decision_records": safety_decision_records,
            "fault_records": safety_fault_records,
            "legacy_safety_mismatch_records": legacy_safety_mismatch_records,
            "sent_matches_legacy_records": sent_matches_legacy_records,
            "sent_matches_safety_records": sent_matches_safety_records,
        },
        "deployment": {
            "policy_inference_records": policy_inference_records,
            "policy_action_trace_records": policy_action_trace_records,
            "complete_policy_action_trace_records": (
                complete_policy_action_trace_records
            ),
            "health_check_records": health_check_records,
            "health_fault_records": health_fault_records,
            "health_event_counts": dict(sorted(health_event_counts.items())),
            "chunk_end_records": chunk_end_records,
            "chunk_fault_abort_records": chunk_fault_abort_records,
            "chunks_with_abort_reason": chunks_with_abort_reason,
            "chunks_completed_all_planned": chunks_completed_all_planned,
            "single_step_chunks": single_step_chunks,
            "multi_step_chunks": multi_step_chunks,
            "planned_chunk_steps": planned_chunk_steps,
            "executed_chunk_steps": executed_chunk_steps,
        },
        "metrics": {
            name: _statistics(values)
            for name, values in sorted(series.items())
            if values
        },
        "notes": [
            "All metric values are milliseconds except tactile.valid_sensor_count.",
            "Camera source time is the RealSense read-completion time, not a hardware exposure timestamp.",
            "Sensor age compares monotonic clocks and is valid only when each ZMQ server and client run on the same host.",
            "This analyzer only reads JSONL logs; it does not modify datasets or robot state.",
            "In shadow mode, sent_matches_legacy should equal decision_records; safety differences are observational only.",
            "FIX005 single-step deployment should report multi_step_chunks=0 unless an explicit A/B run uses steps_per_inference>1.",
        ],
    }


def _read_jsonl(paths: Sequence[Path]) -> List[Mapping[str, Any]]:
    records: List[Mapping[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize one or more GELLO runtime JSONL logs without writing files."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Runtime .jsonl log path(s)")
    args = parser.parse_args()

    paths = [path.expanduser().resolve() for path in args.logs]
    summary = summarize_records(_read_jsonl(paths))
    summary["files"] = [str(path) for path in paths]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
