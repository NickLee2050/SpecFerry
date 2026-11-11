"""Timeouts and hidden or unresponsive SDK children must never look like clean exits."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.deploy.execute.np101 import device


class DeviceRecoveryTests(unittest.TestCase):
    def test_long_device_deadlines_are_rejected_before_any_sdk_process(self):
        with patch.object(device.subprocess, "Popen") as spawn:
            for timeout in (0, -1, 301, 1800, float("nan")):
                with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "300"):
                    device.run_device(None, [], None, None, timeout)
            spawn.assert_not_called()

    def marker(self, root, member, namespace=None, boot_id=None):
        path = root / "np101-recovery-required.json"
        path.write_text(
            json.dumps(
                {
                    "pid_namespace": namespace or os.readlink("/proc/self/ns/pid"),
                    "boot_id": boot_id or device.host_boot_id(),
                    "members": [member],
                }
            )
        )
        return path

    def test_live_process_blocks_a_new_device_run(self):
        fields = Path("/proc/self/stat").read_text().rpartition(")")[2].split()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = self.marker(root, {"pid": os.getpid(), "start_ticks": fields[19]})
            with self.assertRaisesRegex(RuntimeError, "has not exited"):
                device.require_recovered_device(root)
            self.assertTrue(marker.exists())

    def test_a_different_pid_namespace_cannot_clear_the_recovery_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = self.marker(root, {"pid": 999999999, "start_ticks": "1"}, namespace="other")
            with self.assertRaisesRegex(RuntimeError, "PID namespace"):
                device.require_recovered_device(root)
            self.assertTrue(marker.exists())

    def test_process_exit_does_not_establish_driver_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = self.marker(root, {"pid": 999999999, "start_ticks": "1"})
            with self.assertRaisesRegex(RuntimeError, "process exit alone"):
                device.require_recovered_device(root)
            self.assertTrue(marker.exists())

    def test_reboot_preserves_gate_until_pinned_controls_pass(self):
        fields = Path("/proc/self/stat").read_text().rpartition(")")[2].split()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = self.marker(
                root,
                {"pid": os.getpid(), "start_ticks": fields[19]},
                namespace="old namespace",
                boot_id="previous boot",
            )
            with self.assertRaisesRegex(RuntimeError, "recovery is unverified"):
                device.require_recovered_device(root)
            self.assertTrue(marker.exists())

    def test_legacy_marker_without_boot_identity_still_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = self.marker(root, {"pid": 999999999, "start_ticks": "1"})
            record = json.loads(marker.read_text())
            del record["boot_id"]
            marker.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, "process exit alone"):
                device.require_recovered_device(root)
            self.assertTrue(marker.exists())

    def test_incomplete_marker_still_blocks_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "np101-recovery-required.json"
            marker.write_text("")
            with self.assertRaisesRegex(RuntimeError, "marker is incomplete"):
                device.require_recovered_device(root)
            self.assertTrue(marker.exists())

    def run_fixture(self, wait_results, members, **options):
        """Exercise timeout handling without loading the SDK or sending signals."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"fixture")
            process = Mock(pid=123456)
            process.wait.side_effect = wait_results
            with (
                patch.object(device.subprocess, "Popen", return_value=process) as spawn,
                patch.object(
                    device,
                    "process_group_members",
                    return_value=members,
                ),
                patch.object(device.os, "killpg") as terminate,
                patch.object(device, "RECOVERY_ROOT", root / "recovery"),
            ):
                result = device.run_device(
                    binary,
                    [],
                    root / "result",
                    root,
                    timeout=1,
                    **options,
                )
            process.spawn_environment = spawn.call_args.kwargs["env"]
            marker_path = root / "recovery/np101-recovery-required.json"
            marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
            return result, process, terminate, marker

    def test_timing_mode_is_explicit_and_summary_is_separate_from_call_tracing(self):
        for options, expected in (
            ({}, "off"),
            ({"sdk_timing": True}, "calls"),
            ({"sdk_summary": True}, "summary"),
        ):
            with (
                self.subTest(options=options),
                patch.dict(os.environ, SPECFERRY_SDK_TIMING="calls"),
            ):
                result, process, _, _ = self.run_fixture([0], [], **options)
                self.assertEqual(process.spawn_environment["SPECFERRY_SDK_TIMING"], expected)
                self.assertEqual(result["sdk_summary"], bool(options.get("sdk_summary")))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.run_fixture([0], [], sdk_timing=True, sdk_summary=True)

    def test_timeout_with_an_unresponsive_child_is_bounded_and_records_recovery(self):
        result, process, terminate, marker = self.run_fixture(
            [subprocess.TimeoutExpired("fixture", seconds) for seconds in (1, 5, 5)],
            [{"pid": 123457, "start_ticks": "99"}],
        )
        self.assertTrue(result["timeout"])
        self.assertIsNone(result["returncode"])
        self.assertFalse(result["process_group_exited"])
        self.assertTrue(result["device_recovery_required"])
        self.assertEqual(process.wait.call_count, 3)
        terminate.assert_any_call(123456, signal.SIGTERM)
        terminate.assert_any_call(123456, signal.SIGKILL)
        self.assertEqual(marker["boot_id"], device.host_boot_id())

    def test_profiler_is_explicit_and_cannot_instrument_recovery_controls(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, VIV_VX_PROFILE="1", VP_OUTPUT="inherited.vpd"),
        ):
            output = Path(directory) / "capture.vpd"
            result, process, _, _ = self.run_fixture([0], [])
            self.assertEqual(process.spawn_environment["VIV_VX_PROFILE"], "0")
            self.assertNotIn("VP_OUTPUT", process.spawn_environment)
            self.assertFalse(result["profiler"]["requested"])
            result, process, _, _ = self.run_fixture([0], [], profiler_output=output)
            self.assertEqual(process.spawn_environment["VIV_VX_PROFILE"], "1")
            self.assertEqual(process.spawn_environment["VP_OUTPUT"], str(output))
            self.assertEqual(result["profiler"]["output"], str(output))
            for option in ("recovery_control", "sdk_timing", "sdk_summary", "trace_driver"):
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, "separate"):
                    self.run_fixture([0], [], profiler_output=output, **{option: True})
            output.write_bytes(b"original evidence")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                self.run_fixture([0], [], profiler_output=output)
            self.assertEqual(output.read_bytes(), b"original evidence")

    def test_timeout_requires_recovery_even_if_signal_handler_exits_successfully(self):
        result, _, terminate, marker = self.run_fixture(
            [subprocess.TimeoutExpired("fixture", 1), 0], []
        )
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["process_group_exited"])
        self.assertTrue(result["device_recovery_required"])
        terminate.assert_called_once_with(123456, signal.SIGTERM)
        self.assertEqual(marker["members"], [])

    def test_keyboard_interrupt_stops_child_and_preserves_recovery_evidence(self):
        result, _, terminate, marker = self.run_fixture([KeyboardInterrupt(), 0], [])
        self.assertFalse(result["timeout"])
        self.assertTrue(result["interrupted_by_user"])
        self.assertTrue(result["device_recovery_required"])
        self.assertFalse(device.clean_execution(result))
        terminate.assert_called_once_with(123456, signal.SIGTERM)
        self.assertIsNotNone(marker)

    def test_signal_exit_requires_recovery_but_normal_api_error_does_not(self):
        for code, recovery in ((-signal.SIGSEGV, True), (3, False), (0, False)):
            with self.subTest(returncode=code):
                result, _, terminate, marker = self.run_fixture([code], [])
                self.assertFalse(result["timeout"])
                self.assertEqual(result["device_recovery_required"], recovery)
                self.assertEqual(marker is not None, recovery)
                terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
