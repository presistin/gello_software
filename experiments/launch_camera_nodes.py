from dataclasses import dataclass
from multiprocessing import Process
from typing import Optional

import tyro

from gello.cameras.camera_config import (
    CameraAssignment,
    format_camera_assignment,
    resolve_camera_assignments,
)
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.zmq_core.camera_node import ZMQServerCamera


@dataclass
class Args:
    # hostname: str = "127.0.0.1"
    # hostname: str = "128.32.175.167"
    hostname: str = "0.0.0.0"
    start_port: int = 5000
    camera_config: Optional[str] = None
    reset_devices: bool = True



def launch_server(assignment: CameraAssignment, args: Args):
    camera = RealSenseCamera(
        device_id=assignment.serial,
        flip=assignment.flip,
        width=assignment.width,
        height=assignment.height,
        fps=assignment.fps,
    )
    server = ZMQServerCamera(camera, port=assignment.port, host=args.hostname)
    print(f"Starting camera server: {format_camera_assignment(assignment)}")
    server.serve()


def main(args):
    ids = get_device_ids(reset_devices=args.reset_devices)
    assignments = resolve_camera_assignments(
        ids,
        config_path=args.camera_config,
        start_port=args.start_port,
    )
    if args.camera_config is None:
        print(
            "[CameraConfig] --camera-config not set; preserving legacy "
            "RealSense enumeration-order mapping."
        )
    camera_servers = []
    for assignment in assignments:
        # start a python process for each camera
        print(f"Launching camera: {format_camera_assignment(assignment)}")
        camera_servers.append(
            Process(target=launch_server, args=(assignment, args))
        )

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))
