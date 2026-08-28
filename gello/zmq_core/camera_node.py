import pickle
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zmq

from gello.cameras.camera import CameraDriver
from gello.zmq_core.request_reply import (
    CameraCommunicationTimeout,
    RequestReplyClient,
    ZMQProtocolError,
    serialize_remote_error,
)

DEFAULT_CAMERA_PORT = 5000


class ZMQClientCamera(CameraDriver):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
        recv_timeout_ms: Optional[int] = None,
        send_timeout_ms: Optional[int] = None,
        linger_ms: Optional[int] = None,
        capture_timestamps: bool = False,
    ):
        self._client = RequestReplyClient(
            host=host,
            port=port,
            recv_timeout_ms=recv_timeout_ms,
            send_timeout_ms=send_timeout_ms,
            linger_ms=linger_ms,
            timeout_error_cls=CameraCommunicationTimeout,
        )
        self._capture_timestamps = bool(capture_timestamps)
        self._last_frame_timing: Optional[Dict[str, Any]] = None

    @property
    def connection_ok(self) -> Optional[bool]:
        return self._client.connection_ok

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        if not self._capture_timestamps:
            # Raw img_size payload retains the historical wire protocol.
            self._last_frame_timing = None
            return self._client.request(img_size, operation="read")

        result = self._client.request(
            {
                "__gello_camera_method__": "read_with_metadata",
                "img_size": img_size,
            },
            operation="read_with_metadata",
        )
        if not isinstance(result, dict) or not {"image", "depth", "timing"} <= set(result):
            raise ZMQProtocolError(
                "Camera read_with_metadata returned an invalid response at "
                f"{self._client.endpoint}"
            )
        timing = dict(result["timing"])
        request_timing = self._client.last_request_timing or {}
        timing.update(
            {
                "client_request_started_ns": request_timing.get("request_started_ns"),
                "client_reply_received_ns": request_timing.get("reply_received_ns"),
                "round_trip_ms": request_timing.get("round_trip_ms"),
            }
        )
        self._last_frame_timing = timing
        return result["image"], result["depth"]

    @property
    def last_frame_timing(self) -> Optional[Dict[str, Any]]:
        return None if self._last_frame_timing is None else dict(self._last_frame_timing)

    def ping(self):
        return self._client.request(
            {"__gello_camera_method__": "ping"},
            operation="ping",
        )

    def close(self) -> None:
        self._client.close()


class ZMQServerCamera:
    def __init__(
        self,
        camera: CameraDriver,
        port: int = DEFAULT_CAMERA_PORT,
        host: str = "127.0.0.1",
    ):
        self._camera = camera
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Camera Sever Binding to {addr}, Camera: {camera}"
        print(debug_message)
        self._timout_message = f"Timeout in Camera Server, Camera: {camera}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        try:
            while not self._stop_event.is_set():
                try:
                    message = self._socket.recv()
                    try:
                        request = pickle.loads(message)
                        if (
                            isinstance(request, dict)
                            and request.get("__gello_camera_method__") == "ping"
                        ):
                            result = {
                                "server_alive": True,
                                "server_monotonic_ns": time.monotonic_ns(),
                            }
                        elif (
                            isinstance(request, dict)
                            and request.get("__gello_camera_method__")
                            == "read_with_metadata"
                        ):
                            read_started_ns = time.monotonic_ns()
                            image, depth = self._camera.read(request.get("img_size"))
                            read_completed_ns = time.monotonic_ns()
                            result = {
                                "image": image,
                                "depth": depth,
                                "timing": {
                                    "read_started_ns": read_started_ns,
                                    "source_monotonic_ns": read_completed_ns,
                                    "source_event": "camera_read_completed",
                                    "server_reply_monotonic_ns": time.monotonic_ns(),
                                },
                            }
                        else:
                            # Raw img_size payload is retained for wire compatibility.
                            result = self._camera.read(request)
                    except Exception as exc:
                        print(f"Camera RPC failed: {type(exc).__name__}: {exc}")
                        result = serialize_remote_error(exc)
                    self._socket.send(pickle.dumps(result))
                except zmq.Again:
                    print(self._timout_message)
                    # Timeout occurred, check if the stop event is set
        finally:
            self._socket.close(linger=0)
            self._context.term()

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()
# if __name__ == "__main__":
#     import argparse
#     from gello.cameras.realsense_camera import RealSenseCamera

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=5000)
#     args = parser.parse_args()

#     print(f"Starting camera node on port {args.port}...")

#     camera = RealSenseCamera()
#     server = ZMQServerCamera(camera=camera, port=args.port)

#     server.serve()
# if __name__ == "__main__":
#     import argparse
#     from gello.cameras.realsense_camera import RealSenseCamera
#     from gello.zmq_core.camera_node import ZMQServerCamera

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=5000, help="ZMQ port for this camera node")
#     parser.add_argument("--serial", type=str, default=None, help="Optional serial number of RealSense camera")
#     args = parser.parse_args()

#     print(f"Starting camera node on port {args.port}...")

#     # 根据是否传入序列号选择相机
#     if args.serial is None:
#         camera = RealSenseCamera()
#     else:
#         camera = RealSenseCamera(device_id=args.serial)

#     server = ZMQServerCamera(camera=camera, port=args.port)
#     server.serve()
