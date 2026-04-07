import pickle
import threading
from typing import Optional, Tuple

import numpy as np
import zmq

from dataclasses import dataclass
import tyro
@dataclass
class CameraNodeArgs:
    port:int=5000
    serial:Optional[str]=None
from gello.cameras.camera import CameraDriver

DEFAULT_CAMERA_PORT = 5000


class ZMQClientCamera(CameraDriver):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_CAMERA_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # pack the image_size and send it to the server
        send_message = pickle.dumps(img_size)
        self._socket.send(send_message)
        state_dict = pickle.loads(self._socket.recv())
        return state_dict


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
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                img_size = pickle.loads(message)
                camera_read = self._camera.read(img_size)
                self._socket.send(pickle.dumps(camera_read))
            except zmq.Again:
                print(self._timout_message)
                # Timeout occurred, check if the stop event is set

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