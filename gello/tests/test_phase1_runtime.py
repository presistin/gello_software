import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from gello.cameras.camera_config import resolve_camera_assignments
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.robots.ur import URRobot
from gello.safety import (
    SafetyConfig,
    SafetySupervisor,
    action_delta_metrics,
    apply_joint_and_gripper_limits,
    check_robot_health,
    default_safety_config_path,
    limit_gripper_by_force,
    load_safety_config,
)
from gello.utils.control_utils import run_control_loop
from gello.utils.tactile_runtime import TactileSample
from gello.utils.runtime_logger import RuntimeLogger
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera
from gello.zmq_core.request_reply import (
    CameraCommunicationTimeout,
    RobotCommunicationTimeout,
)
from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot
from tools.analyze_runtime_log import summarize_records


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

    def get_health(self):
        return {
            "robot_connected": True,
            "operational": True,
        }


class _FakeCamera:
    def read(self, img_size=None):
        shape = (4, 6) if img_size is None else tuple(img_size)
        return (
            np.zeros((*shape, 3), dtype=np.uint8),
            np.zeros(shape, dtype=np.uint16),
        )


class _FakeRTDEControl:
    def isConnected(self):
        return True


class _FakeRTDEReceive:
    def isConnected(self):
        return True

    def getRobotMode(self):
        return 7

    def getRuntimeState(self):
        return 2

    def getSafetyMode(self):
        return 1

    def isEmergencyStopped(self):
        return False

    def isProtectiveStopped(self):
        return False


class _FakeProcess:
    def is_alive(self):
        return True


class _SingleStepSafetyEnv:
    def __init__(self, current):
        self.current = np.asarray(current, dtype=np.float32)
        self.commands = []

    def get_obs(self):
        return {
            "joints": self.current.copy(),
            "gripper": np.array([self.current[6]], dtype=np.float32),
        }

    def get_last_observation_timing(self):
        now_ns = time.monotonic_ns()
        return {
            "cameras": {
                role: {"source_monotonic_ns": now_ns - 20_000_000}
                for role in ("wrist", "base", "camera_2")
            },
            "robot": {"source_monotonic_ns": now_ns - 1_000_000},
        }

    def step(self, action):
        self.commands.append(np.asarray(action).copy())
        raise KeyboardInterrupt

    def robot(self):
        return self

    def command_joint_state(self, action):
        self.commands.append(np.asarray(action).copy())


class _FixedAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def act(self, obs):
        return self.action.copy()


class _FixedTactileRuntime:
    def __init__(self, force_n):
        self.force_n = force_n

    def read_sample(self):
        return TactileSample(
            vector6=np.zeros(6, dtype=np.float32),
            max_force_n=self.force_n,
            monotonic_ns=time.monotonic_ns() - 20_000_000,
            valid_sensor_count=2,
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
            self.assertEqual(client.get_observations()["joint_positions"].shape, (7,))
            self.assertIsNone(client.last_observation_timing)
            health = client.ping()
            self.assertTrue(health["server_alive"])
            self.assertTrue(health["robot_connected"])
            self.assertTrue(health["robot_health"]["operational"])
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
            self.assertIsNone(client.last_frame_timing)
            self.assertTrue(client.ping()["server_alive"])
        finally:
            client.close()
            server.stop()
            thread.join(timeout=2.0)

    def test_opt_in_observation_timestamps_are_aggregated(self):
        robot_port = _free_tcp_port()
        camera_port = _free_tcp_port()
        robot_server = ZMQServerRobot(_FakeRobot(), port=robot_port)
        camera_server = ZMQServerCamera(_FakeCamera(), port=camera_port)
        robot_thread = threading.Thread(target=robot_server.serve, daemon=True)
        camera_thread = threading.Thread(target=camera_server.serve, daemon=True)
        robot_thread.start()
        camera_thread.start()
        robot_client = ZMQClientRobot(
            port=robot_port,
            capture_timestamps=True,
        )
        camera_client = ZMQClientCamera(
            port=camera_port,
            capture_timestamps=True,
        )
        env = RobotEnv(
            robot_client,
            camera_dict={"wrist": camera_client},
            capture_timestamps=True,
        )
        try:
            observation = env.get_obs()
            timing = env.get_last_observation_timing()
            self.assertEqual(observation["image_wrist"].shape, (4, 6, 3))
            self.assertIsNotNone(timing)
            self.assertGreaterEqual(timing["observation_span_ms"], 0.0)
            self.assertGreaterEqual(timing["cameras"]["wrist"]["age_ms"], 0.0)
            self.assertGreaterEqual(timing["robot"]["age_ms"], 0.0)
            self.assertGreaterEqual(
                timing["cameras"]["wrist"]["round_trip_ms"], 0.0
            )
            self.assertGreaterEqual(timing["robot"]["round_trip_ms"], 0.0)
        finally:
            camera_client.close()
            robot_client.close()
            camera_server.stop()
            robot_server.stop()
            camera_thread.join(timeout=2.0)
            robot_thread.join(timeout=2.0)

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


class URRobotHealthTests(unittest.TestCase):
    def test_print_robot_declares_simulated_operational_health(self):
        health = PrintRobot(7, dont_print=True).get_health()
        self.assertTrue(health["robot_connected"])
        self.assertTrue(health["operational"])
        self.assertTrue(health["simulated"])

    def test_health_snapshot_uses_read_only_rtde_status(self):
        robot = object.__new__(URRobot)
        robot.robot = _FakeRTDEControl()
        robot.r_inter = _FakeRTDEReceive()
        robot.p_gripper = _FakeProcess()

        health = robot.get_health()

        self.assertTrue(health["robot_connected"])
        self.assertTrue(health["operational"])
        self.assertEqual(health["robot_mode"], 7)
        self.assertEqual(health["runtime_state"], 2)
        self.assertEqual(health["safety_mode"], 1)
        self.assertFalse(health["emergency_stopped"])
        self.assertFalse(health["protective_stopped"])
        self.assertTrue(health["gripper_process_alive"])


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

    def test_runtime_analyzer_summarizes_sensor_freshness(self):
        records = [
            {
                "event": "teleop_step",
                "monotonic_ns": 1_000_000_000,
                "loop_duration_ms": 10.0,
                "observation_timing": {
                    "observation_span_ms": 8.0,
                    "cameras": {
                        "wrist": {
                            "age_ms": 7.0,
                            "client_receive_age_ms": 5.0,
                            "round_trip_ms": 2.0,
                        }
                    },
                    "robot": {
                        "age_ms": 1.0,
                        "client_receive_age_ms": 0.5,
                        "round_trip_ms": 0.8,
                    },
                },
                "tactile_timing": {
                    "age_ms": 20.0,
                    "valid_sensor_count": 2,
                },
                "legacy_action": [0.0, 0.0],
                "safety_action": [0.1, 0.0],
                "sent_command": [0.0, 0.0],
                "safety_events": [
                    {"code": "joint_step_clipped", "severity": "limit"}
                ],
            },
            {
                "event": "teleop_step",
                "monotonic_ns": 1_020_000_000,
                "loop_duration_ms": 12.0,
            },
            {
                "event": "idle_observation",
                "monotonic_ns": 1_025_000_000,
            },
            {
                "event": "idle_observation",
                "monotonic_ns": 1_035_000_000,
            },
            {
                "event": "policy_inference",
                "monotonic_ns": 1_036_000_000,
                "duration_ms": 1250.0,
                "requested_execution_count": 1,
                "chunk_boundary": {"max_abs": 0.04, "l2": 0.05},
                "first_action_vs_current": {"max_abs": 0.03, "l2": 0.04},
            },
            {
                "event": "robot_health_check",
                "monotonic_ns": 1_037_000_000,
                "faulted": False,
                "health_events": [],
            },
            {
                "event": "policy_step",
                "monotonic_ns": 1_038_000_000,
                "raw_policy_action": [0.0, 0.1],
                "safety_action": [0.0, 0.1],
                "executed_action": [0.0, 0.1],
            },
            {
                "event": "policy_chunk_end",
                "monotonic_ns": 1_039_000_000,
                "planned_steps": 1,
                "executed_steps": 1,
                "abort_reason": None,
            },
            {
                "event": "run_end",
                "monotonic_ns": 1_041_000_000,
                "dropped_events": 0,
            },
        ]

        summary = summarize_records(records)

        self.assertTrue(summary["monotonic_non_decreasing"])
        self.assertEqual(summary["timestamp_coverage"]["observation_timing_records"], 1)
        self.assertEqual(summary["metrics"]["teleop_step.loop_duration_ms"]["p50"], 11.0)
        self.assertEqual(summary["metrics"]["idle_observation.event_interval_ms"]["p50"], 10.0)
        self.assertEqual(summary["metrics"]["camera.wrist.age_ms"]["max"], 7.0)
        self.assertEqual(summary["metrics"]["tactile.valid_sensor_count"]["max"], 2.0)
        self.assertEqual(summary["safety"]["decision_records"], 1)
        self.assertEqual(summary["safety"]["legacy_safety_mismatch_records"], 1)
        self.assertEqual(summary["safety"]["sent_matches_legacy_records"], 1)
        self.assertEqual(
            summary["safety"]["event_counts"]["joint_step_clipped"], 1
        )
        self.assertEqual(summary["deployment"]["policy_inference_records"], 1)
        self.assertEqual(summary["deployment"]["health_check_records"], 1)
        self.assertEqual(summary["deployment"]["health_fault_records"], 0)
        self.assertEqual(summary["deployment"]["single_step_chunks"], 1)
        self.assertEqual(summary["deployment"]["multi_step_chunks"], 0)
        self.assertEqual(summary["deployment"]["executed_chunk_steps"], 1)
        self.assertEqual(
            summary["deployment"]["complete_policy_action_trace_records"], 1
        )
        self.assertEqual(
            summary["metrics"]["deployment.chunk_boundary.max_abs"]["max"],
            0.04,
        )


class SafetySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.config = SafetyConfig()
        self.supervisor = SafetySupervisor(self.config)
        self.now_ns = 2_000_000_000
        self.current = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.4], dtype=np.float32
        )
        self.sample = TactileSample(
            vector6=np.zeros(6, dtype=np.float32),
            max_force_n=0.2,
            monotonic_ns=self.now_ns - 40_000_000,
            valid_sensor_count=2,
        )
        self.timing = {
            "cameras": {
                role: {"source_monotonic_ns": self.now_ns - 50_000_000}
                for role in self.config.required_cameras
            },
            "robot": {"source_monotonic_ns": self.now_ns - 5_000_000},
        }

    def test_tracked_config_loads_explicit_measured_thresholds(self):
        config = load_safety_config(str(default_safety_config_path()))
        self.assertEqual(config.tactile_grip_limit_n, 0.5)
        self.assertEqual(config.max_joint_step_rad, 0.16)
        self.assertEqual(config.max_camera_age_ms, 100.0)
        self.assertEqual(config.max_tactile_age_ms, 120.0)
        self.assertEqual(config.max_robot_age_ms, 10.0)
        self.assertEqual(config.expected_tactile_sensor_count, 2)

    def test_action_delta_metrics_reports_boundary_without_mutation(self):
        action = np.array([0.2, -0.1, 0.4], dtype=np.float32)
        reference = np.array([0.1, -0.1, 0.1], dtype=np.float32)
        action_before = action.copy()
        metrics = action_delta_metrics(action, reference)

        np.testing.assert_array_equal(action, action_before)
        np.testing.assert_allclose(metrics["delta"], [0.1, 0.0, 0.3])
        self.assertAlmostEqual(metrics["max_abs"], 0.3, places=6)
        self.assertAlmostEqual(metrics["l2"], np.sqrt(0.1**2 + 0.3**2), places=6)

    def test_robot_health_classifier_accepts_healthy_snapshot(self):
        response = {
            "server_alive": True,
            "robot_health": {
                "robot_connected": True,
                "operational": True,
                "emergency_stopped": False,
                "protective_stopped": False,
                "gripper_process_alive": True,
            },
        }
        self.assertEqual(check_robot_health(response), ())

    def test_robot_health_classifier_fails_closed_and_reports_faults(self):
        missing = check_robot_health(
            {"server_alive": True, "robot_health": None}
        )
        self.assertEqual(missing[0]["code"], "robot_health_missing")
        self.assertEqual(missing[0]["severity"], "fault")

        faults = check_robot_health(
            {
                "server_alive": True,
                "robot_health": {
                    "robot_connected": False,
                    "operational": False,
                    "emergency_stopped": False,
                    "protective_stopped": True,
                    "gripper_process_alive": False,
                    "errors": {"runtime_state": "read failed"},
                },
            }
        )
        self.assertEqual(
            {event["code"] for event in faults},
            {
                "robot_protective_stopped",
                "robot_disconnected",
                "robot_not_operational",
                "gripper_process_dead",
                "robot_health_read_error",
            },
        )

    def test_joint_and_gripper_bounds_match_legacy_deployment_math(self):
        requested = self.current + np.array(
            [0.3, -0.3, 0.1, -0.1, 0.2, -0.2, 0.8], dtype=np.float32
        )
        safe, events = apply_joint_and_gripper_limits(
            requested, self.current, max_joint_step_rad=0.16
        )
        np.testing.assert_allclose(
            safe[:6] - self.current[:6],
            [0.16, -0.16, 0.1, -0.1, 0.16, -0.16],
            atol=1e-7,
        )
        self.assertEqual(float(safe[6]), 1.0)
        self.assertEqual(
            [event["code"] for event in events],
            ["joint_step_clipped", "gripper_range_clipped"],
        )

    def test_touch_freezes_only_further_close(self):
        close = self.current.copy()
        close[6] = 0.8
        frozen, limited = limit_gripper_by_force(
            close, self.current[6], 0.5, 0.5, True
        )
        self.assertTrue(limited)
        self.assertAlmostEqual(float(frozen[6]), float(self.current[6]))

        opening = self.current.copy()
        opening[6] = 0.1
        allowed, limited = limit_gripper_by_force(
            opening, self.current[6], 10.0, 0.5, True
        )
        self.assertFalse(limited)
        self.assertAlmostEqual(float(allowed[6]), 0.1)

    def test_same_inputs_produce_identical_collection_and_deployment_output(self):
        requested = self.current.copy()
        requested[0] += 0.4
        requested[6] = 0.9
        touch = TactileSample(
            vector6=np.zeros(6, dtype=np.float32),
            max_force_n=0.7,
            monotonic_ns=self.sample.monotonic_ns,
            valid_sensor_count=2,
        )
        collection = self.supervisor.filter_action(
            requested,
            self.current,
            touch,
            self.timing,
            now_ns=self.now_ns,
            check_cameras=True,
        )
        deployment = SafetySupervisor(self.config).filter_action(
            requested,
            self.current,
            touch,
            self.timing,
            now_ns=self.now_ns,
            check_cameras=True,
        )
        np.testing.assert_array_equal(
            collection.safe_action, deployment.safe_action
        )
        self.assertEqual(collection.events, deployment.events)
        self.assertAlmostEqual(
            float(collection.safe_action[6]), float(self.current[6])
        )

    def test_stale_or_incomplete_sensor_state_returns_current_hold(self):
        stale_timing = dict(self.timing)
        stale_timing["robot"] = {
            "source_monotonic_ns": self.now_ns - 11_000_000
        }
        incomplete = TactileSample(
            vector6=np.zeros(6, dtype=np.float32),
            max_force_n=0.0,
            monotonic_ns=self.now_ns - 121_000_000,
            valid_sensor_count=1,
        )
        decision = self.supervisor.filter_action(
            self.current + 0.1,
            self.current,
            incomplete,
            stale_timing,
            now_ns=self.now_ns,
            check_cameras=True,
        )
        self.assertTrue(decision.faulted)
        np.testing.assert_array_equal(decision.safe_action, self.current)
        self.assertEqual(
            {event["code"] for event in decision.events},
            {
                "robot_stale",
                "tactile_stale",
                "tactile_sensor_count_invalid",
            },
        )

    def test_camera_freshness_is_checked_at_observation_boundary(self):
        timing = {
            "cameras": dict(self.timing["cameras"]),
            "robot": self.timing["robot"],
        }
        timing["cameras"]["wrist"] = {
            "source_monotonic_ns": self.now_ns - 101_000_000
        }
        events = self.supervisor.check_observation_freshness(
            timing,
            self.sample,
            now_ns=self.now_ns,
            check_cameras=True,
        )
        self.assertEqual([event["code"] for event in events], ["camera_stale"])
        self.assertEqual(events[0]["camera"], "wrist")

    def test_collection_shadow_sends_legacy_and_enforce_sends_safe_action(self):
        requested = self.current.copy()
        requested[0] += 0.4
        requested[6] = 0.9
        tactile = _FixedTactileRuntime(force_n=0.7)

        shadow_env = _SingleStepSafetyEnv(self.current)
        run_control_loop(
            shadow_env,
            _FixedAgent(requested),
            print_timing=False,
            safety_tactile_runtime=tactile,
            safety_supervisor=self.supervisor,
            safety_mode="shadow",
        )
        np.testing.assert_array_equal(shadow_env.commands[0], requested)

        enforce_env = _SingleStepSafetyEnv(self.current)
        run_control_loop(
            enforce_env,
            _FixedAgent(requested),
            print_timing=False,
            safety_tactile_runtime=tactile,
            safety_supervisor=self.supervisor,
            safety_mode="enforce",
        )
        self.assertAlmostEqual(
            float(enforce_env.commands[0][0]),
            float(self.current[0] + self.config.max_joint_step_rad),
            places=6,
        )
        self.assertAlmostEqual(
            float(enforce_env.commands[0][6]), float(self.current[6]), places=6
        )


if __name__ == "__main__":
    unittest.main()
