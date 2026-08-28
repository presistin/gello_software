import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        capture_timestamps: bool = False,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._capture_timestamps = bool(capture_timestamps)
        self._last_observation_timing: Optional[Dict[str, Any]] = None

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observation_started_ns = (
            time.monotonic_ns() if self._capture_timestamps else None
        )
        observations = {}
        camera_timing: Dict[str, Any] = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            if self._capture_timestamps:
                timing = getattr(camera, "last_frame_timing", None)
                if timing is not None:
                    camera_timing[name] = timing
            # if name =="wrist":
            #     observations["image_wrist"]=image
            # elif name=="base":
            #     observations["image_base"]=image
            observations[f"image_{name}"] = image
            # observations[f"{name}_rgb"] = image
            # observations[f"{name}_depth"] = depth


        robot_obs = self._robot.get_observations()
        robot_timing = None
        if self._capture_timestamps:
            robot_timing = getattr(self._robot, "last_observation_timing", None)
        observations["joints"]=robot_obs["joint_positions"]
        observations["gripper"]=np.array(
            [robot_obs["gripper_position"]]
        )
        observations["joint_positions"] = observations["joints"]
        if self._capture_timestamps:
            observation_ready_ns = time.monotonic_ns()

            def add_age(timing):
                if timing is None:
                    return None
                enriched = dict(timing)
                source_ns = enriched.get("source_monotonic_ns")
                received_ns = enriched.get("client_reply_received_ns")
                if isinstance(source_ns, int):
                    enriched["age_ms"] = (
                        observation_ready_ns - source_ns
                    ) / 1e6
                if isinstance(received_ns, int):
                    enriched["client_receive_age_ms"] = (
                        observation_ready_ns - received_ns
                    ) / 1e6
                return enriched

            self._last_observation_timing = {
                "observation_started_ns": observation_started_ns,
                "observation_ready_ns": observation_ready_ns,
                "observation_span_ms": (
                    observation_ready_ns - observation_started_ns
                )
                / 1e6,
                "cameras": {
                    name: add_age(timing)
                    for name, timing in camera_timing.items()
                },
                "robot": add_age(robot_timing),
            }
        else:
            self._last_observation_timing = None
        return observations
        # assert "joint_positions" in robot_obs
        # assert "joint_velocities" in robot_obs
        # assert "ee_pos_quat" in robot_obs
        # observations["joint_positions"] = robot_obs["joint_positions"]
        # observations["joint_velocities"] = robot_obs["joint_velocities"]
        # observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        # observations["gripper_position"] = robot_obs["gripper_position"]
        # return observations

    def get_last_observation_timing(self) -> Optional[Dict[str, Any]]:
        if self._last_observation_timing is None:
            return None
        return deepcopy(self._last_observation_timing)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
