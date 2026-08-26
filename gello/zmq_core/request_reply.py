"""Shared, backward-compatible helpers for synchronous ZMQ REQ/REP clients."""

from __future__ import annotations

import pickle
from typing import Any, Optional, Type

import zmq


REMOTE_ERROR_KEY = "__gello_remote_error__"


class ZMQCommunicationError(RuntimeError):
    """Base error for a failed GELLO ZMQ request."""


class ZMQRequestTimeout(ZMQCommunicationError):
    def __init__(
        self,
        *,
        endpoint: str,
        operation: str,
        phase: str,
        timeout_ms: Optional[int],
    ) -> None:
        timeout_text = "configured timeout" if timeout_ms is None else f"{timeout_ms} ms"
        super().__init__(
            f"ZMQ {phase} timeout during '{operation}' at {endpoint} "
            f"after {timeout_text}; the REQ socket was rebuilt"
        )
        self.endpoint = endpoint
        self.operation = operation
        self.phase = phase
        self.timeout_ms = timeout_ms


class RobotCommunicationTimeout(ZMQRequestTimeout):
    pass


class CameraCommunicationTimeout(ZMQRequestTimeout):
    pass


class ZMQRemoteError(ZMQCommunicationError):
    pass


class ZMQProtocolError(ZMQCommunicationError):
    pass


def serialize_remote_error(exc: Exception) -> dict:
    return {
        REMOTE_ERROR_KEY: {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    }


def raise_for_remote_error(result: Any, *, operation: str) -> None:
    if not isinstance(result, dict) or REMOTE_ERROR_KEY not in result:
        return
    payload = result[REMOTE_ERROR_KEY]
    if isinstance(payload, dict):
        error_type = payload.get("type", "RemoteError")
        message = payload.get("message", "")
    else:
        error_type = "RemoteError"
        message = str(payload)
    raise ZMQRemoteError(f"Remote {error_type} during '{operation}': {message}")


class RequestReplyClient:
    """Own a REQ socket and rebuild it after timeout/state-machine failure.

    Timeouts default to ``None`` so existing callers retain the historical
    blocking behaviour until they explicitly opt in.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        recv_timeout_ms: Optional[int] = None,
        send_timeout_ms: Optional[int] = None,
        linger_ms: Optional[int] = None,
        timeout_error_cls: Type[ZMQRequestTimeout] = ZMQRequestTimeout,
    ) -> None:
        self.endpoint = f"tcp://{host}:{port}"
        self.recv_timeout_ms = _validate_timeout("recv_timeout_ms", recv_timeout_ms)
        self.send_timeout_ms = _validate_timeout("send_timeout_ms", send_timeout_ms)
        self.linger_ms = _validate_timeout("linger_ms", linger_ms, allow_zero=True)
        self.timeout_error_cls = timeout_error_cls
        self.connection_ok: Optional[bool] = None
        self._closed = False
        self._context = zmq.Context()
        self._socket = self._create_socket()

    def _create_socket(self):
        socket = self._context.socket(zmq.REQ)
        if self.recv_timeout_ms is not None:
            socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        if self.send_timeout_ms is not None:
            socket.setsockopt(zmq.SNDTIMEO, self.send_timeout_ms)
        if self.linger_ms is not None:
            socket.setsockopt(zmq.LINGER, self.linger_ms)
        elif self.recv_timeout_ms is not None or self.send_timeout_ms is not None:
            socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        return socket

    def _reset_socket(self) -> None:
        if self._closed:
            return
        self._socket.close(linger=0)
        self._socket = self._create_socket()

    def request(self, payload: Any, *, operation: str) -> Any:
        if self._closed:
            raise ZMQCommunicationError(
                f"ZMQ request '{operation}' attempted after client close: {self.endpoint}"
            )
        try:
            self._socket.send(pickle.dumps(payload))
        except zmq.Again as exc:
            self.connection_ok = False
            self._reset_socket()
            raise self.timeout_error_cls(
                endpoint=self.endpoint,
                operation=operation,
                phase="send",
                timeout_ms=self.send_timeout_ms,
            ) from exc
        except zmq.ZMQError as exc:
            self.connection_ok = False
            self._reset_socket()
            raise ZMQCommunicationError(
                f"ZMQ send failed during '{operation}' at {self.endpoint}: {exc}"
            ) from exc

        try:
            reply = self._socket.recv()
        except zmq.Again as exc:
            self.connection_ok = False
            self._reset_socket()
            raise self.timeout_error_cls(
                endpoint=self.endpoint,
                operation=operation,
                phase="receive",
                timeout_ms=self.recv_timeout_ms,
            ) from exc
        except zmq.ZMQError as exc:
            self.connection_ok = False
            self._reset_socket()
            raise ZMQCommunicationError(
                f"ZMQ receive failed during '{operation}' at {self.endpoint}: {exc}"
            ) from exc

        try:
            result = pickle.loads(reply)
        except Exception as exc:
            self.connection_ok = False
            self._reset_socket()
            raise ZMQProtocolError(
                f"Invalid pickle reply during '{operation}' at {self.endpoint}: {exc}"
            ) from exc

        self.connection_ok = True
        raise_for_remote_error(result, operation=operation)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        self._context.term()


def _validate_timeout(
    name: str,
    value: Optional[int],
    *,
    allow_zero: bool = False,
) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} ms or None, got {value}")
    return value
