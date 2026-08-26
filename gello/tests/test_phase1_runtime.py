import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from gello.cameras.camera_config import resolve_camera_assignments
from gello.utils.runtime_logger import RuntimeLogger
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera
from gello.zmq_core.request_reply import (
    CameraCommunicationTimeout,
    RobotCommunicationTimeout,
)
from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeRobot:
    def __init__(self) -> None:
        self.joints = np.zeros(7, dtype=np.float32)

    def num_dofs(self):
        return 7

    def get_joint_state(self):
        return self.joints.copy()

    def command_joint_state(self, joint_state):
        self.joints = np.asarray(joint_state, dtype=np.float32).copy()

    def get_observations(self):
        return {
            "joint_positions": self.joints.copy(),
            "gripper_position": np.float32(self.joints[-1]),
        }


class _FakeCamera:
    def read(self, img_size=None):
        shape = (4, 6) if img_size is None else tuple(img_size)
        return (
            np.zeros((*shape, 3), dtype=np.uint8),
            np.zeros(shape, dtype=np.uint16),
        )


class CameraConfigTests(unittest.TestCase):
    def test_default_is_legacy_enumeration_order(self):
        assignments = resolve_camera_assignments(["B", "A"], start_port=5100)
        self.assertEqual([item.serial for item in assignments], ["B", "A"])
        self.assertEqual([item.port for item in assignments], [5100, 5101])
        self.assertTrue(all(not item.configured for item in assignments))

    def test_config_binds_roles_to_serials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cameras.yaml"
            path.write_text(
                "cameras:\n"
                "  wrist:\n"
                "    serial: '111'\n"
                "    port: 5000\n"
                "  base:\n"
                "    serial: '222'\n"
                "    port: 5001\n",
                encoding="utf-8",
            )
            assignments = resolve_camera_assignments(
                ["222", "111"], config_path=str(path)
            )
        self.assertEqual(
            [(item.role, item.serial, item.port) for item in assignments],
            [("wrist", "111", 5000), ("base", "222", 5001)],
        )

    def test_required_missing_camera_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cameras.yaml"
            path.write_text(
                "cameras:\n  wrist:\n    serial: 'missing'\n    port: 5000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing"):
                resolve_camera_assignments(["present"], config_path=str(path))


class ZMQCompatibilityTests(unittest.TestCase):
    def test_robot_round_trip_with_default_client_arguments(self):
        port = _free_tcp_port()
        server = ZMQServerRobot(_FakeRobot(), port=port)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        client = ZMQClientRobot(port=port)
        try:
            self.assertEqual(client.num_dofs(), 7)
            command = np.linspace(0, 0.6, 7, dtype=np.float32)
            client.command_joint_state(command)
            np.testing.assert_allclose(client.get_joint_state(), command)
            self.assertTrue(client.ping()["server_alive"])
        finally:
            client.close()
            server.stop()
            thread.join(timeout=2.0)

    def test_camera_round_trip_preserves_raw_img_size_wire_shape(self):
        port = _free_tcp_port()
        server = ZMQServerCamera(_FakeCamera(), port=port)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        client = ZMQClientCamera(port=port)
        try:
            image, depth = client.read((3, 5))
            self.assertEqual(image.shape, (3, 5, 3))
            self.assertEqual(depth.shape, (3, 5))
            self.assertTrue(client.ping()["server_alive"])
        finally:
            client.close()
            server.stop()
            thread.join(timeout=2.0)

    def test_robot_timeout_is_classified_and_socket_recovers(self):
        port = _free_tcp_port()
        client = ZMQClientRobot(port=port, recv_timeout_ms=60)
        with self.assertRaises(RobotCommunicationTimeout):
            client.ping()
        self.assertFalse(client.connection_ok)

        server = ZMQServerRobot(_FakeRobot(), port=port)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    response = client.ping()
                    break
                except RobotCommunicationTimeout:
                    if time.monotonic() >= deadline:
                        raise
            self.assertTrue(response["server_alive"])
            self.assertTrue(client.connection_ok)
        finally:
            client.close()
            server.stop()
            thread.join(timeout=2.0)

    def test_camera_timeout_has_camera_specific_type(self):
        port = _free_tcp_port()
        client = ZMQClientCamera(port=port, recv_timeout_ms=30)
        try:
            with self.assertRaises(CameraCommunicationTimeout):
                client.read()
        finally:
            client.close()


class RuntimeLoggerTests(unittest.TestCase):
    def test_jsonl_lifecycle_and_numpy_serialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.jsonl"
            logger = RuntimeLogger(path, metadata={"mode": "test"})
            logger.log("step", action=np.array([1.0, 2.0], dtype=np.float32))
            logger.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["event"] for record in records], ["run_start", "step", "run_end"])
        self.assertEqual(records[1]["action"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
