import pickle
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import zmq

from gello.robots.robot import Robot
from gello.zmq_core.request_reply import (
    RequestReplyClient,
    RobotCommunicationTimeout,
    serialize_remote_error,
)

DEFAULT_ROBOT_PORT = 6000


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        try:
            while not self._stop_event.is_set():
                try:
                    # Wait for next request from client
                    message = self._socket.recv()
                    try:
                        request = pickle.loads(message)
                        result = self._dispatch(request)
                    except Exception as exc:
                        print(f"Robot RPC failed: {type(exc).__name__}: {exc}")
                        result = serialize_remote_error(exc)
                    self._socket.send(pickle.dumps(result))
                except zmq.Again:
                    # Timeout occurred - don't spam the console
                    pass
        finally:
            self._socket.close(linger=0)
            self._context.term()

    def _dispatch(self, request: Dict[str, Any]) -> Any:
        method = request.get("method")
        args = request.get("args", {})
        if method == "num_dofs":
            return self._robot.num_dofs()
        if method == "get_joint_state":
            return self._robot.get_joint_state()
        if method == "command_joint_state":
            return self._robot.command_joint_state(**args)
        if method == "get_observations":
            return self._robot.get_observations()
        if method == "ping":
            health_method = getattr(self._robot, "get_health", None)
            robot_health = health_method() if callable(health_method) else None
            return {
                "server_alive": True,
                "server_monotonic_ns": time.monotonic_ns(),
                "robot_health": robot_health,
            }
        raise NotImplementedError(f"Invalid method: {method}, args={args}")

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(
        self,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
        recv_timeout_ms: Optional[int] = None,
        send_timeout_ms: Optional[int] = None,
        linger_ms: Optional[int] = None,
    ):
        self._client = RequestReplyClient(
            host=host,
            port=port,
            recv_timeout_ms=recv_timeout_ms,
            send_timeout_ms=send_timeout_ms,
            linger_ms=linger_ms,
            timeout_error_cls=RobotCommunicationTimeout,
        )

    @property
    def connection_ok(self) -> Optional[bool]:
        return self._client.connection_ok

    def _request(self, method: str, args: Optional[Dict[str, Any]] = None) -> Any:
        request = {"method": method}
        if args:
            request["args"] = args
        return self._client.request(request, operation=method)

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        return self._request("num_dofs")

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        return self._request("get_joint_state")

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
        """
        return self._request(
            "command_joint_state",
            args={"joint_state": joint_state},
        )

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        return self._request("get_observations")

    def ping(self) -> Dict[str, Any]:
        return self._request("ping")

    def close(self) -> None:
        """Close the ZMQ socket and context."""
        self._client.close()
