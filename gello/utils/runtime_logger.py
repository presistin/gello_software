"""Low-overhead, opt-in JSONL logging for collection and deployment runs."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


_STOP = object()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return repr(value)


class RuntimeLogger:
    """Write structured events on a background thread.

    Callers only construct this class when logging is explicitly requested.
    A full queue drops log events instead of delaying the robot control loop.
    """

    def __init__(
        self,
        path: Path,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        flush_interval_s: float = 1.0,
        queue_size: int = 4096,
    ) -> None:
        if flush_interval_s <= 0:
            raise ValueError("flush_interval_s must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = dict(metadata or {})
        self._flush_interval_s = float(flush_interval_s)
        self._queue: queue.Queue = queue.Queue(maxsize=int(queue_size))
        self._dropped_events = 0
        self._closed = False
        self._writer_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="gello-runtime-logger",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def log(self, event: str, **fields: Any) -> None:
        if self._closed:
            return
        record = {
            "event": str(event),
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_events += 1

    def _write_record(self, stream, record: Mapping[str, Any]) -> None:
        stream.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
            + "\n"
        )

    def _writer_loop(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                self._write_record(
                    stream,
                    {
                        "event": "run_start",
                        "wall_time_ns": time.time_ns(),
                        "monotonic_ns": time.monotonic_ns(),
                        "metadata": self._metadata,
                    },
                )
                last_flush = time.monotonic()
                while True:
                    timeout = max(
                        0.0,
                        self._flush_interval_s - (time.monotonic() - last_flush),
                    )
                    try:
                        item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        stream.flush()
                        last_flush = time.monotonic()
                        continue
                    if item is _STOP:
                        break
                    self._write_record(stream, item)
                    if time.monotonic() - last_flush >= self._flush_interval_s:
                        stream.flush()
                        last_flush = time.monotonic()
                self._write_record(
                    stream,
                    {
                        "event": "run_end",
                        "wall_time_ns": time.time_ns(),
                        "monotonic_ns": time.monotonic_ns(),
                        "dropped_events": self._dropped_events,
                    },
                )
                stream.flush()
        except BaseException as exc:  # surfaced from close(), never kills control
            self._writer_error = exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Preserve shutdown diagnostics even when the runtime queue was full.
        if self._thread.is_alive():
            try:
                self._queue.put(_STOP, timeout=2.0)
            except queue.Full:
                raise RuntimeError(
                    f"Runtime logger queue did not drain during close: {self.path}"
                )
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError(f"Runtime logger did not stop: {self.path}")
        if self._writer_error is not None:
            raise RuntimeError(f"Runtime logger failed: {self.path}") from self._writer_error

    def __enter__(self) -> "RuntimeLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def create_runtime_logger(
    base_dir: str,
    *,
    run_kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
    flush_interval_s: float = 1.0,
) -> RuntimeLogger:
    """Create a uniquely named runtime JSONL file below ``base_dir``."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_kind = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_kind)
    filename = f"{stamp}_{safe_kind}_pid{os.getpid()}.jsonl"
    return RuntimeLogger(
        Path(base_dir).expanduser() / filename,
        metadata=metadata,
        flush_interval_s=flush_interval_s,
    )
