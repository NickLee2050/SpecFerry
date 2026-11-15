"""Timeouts and hidden or unresponsive SDK children must never look like clean exits."""

import json
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.deploy.execute.np101 import device


class DeviceRecoveryTests(unittest.TestCase):
    def test_long_device_deadlines_are_rejected_before_any_sdk_process(self):
        with patch.object(device, "supervise") as spawn:
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

    def run_fixture(self, source="pass", *, timeout=1, **options):
        """Run only Python fixture children with an SDK-independent native guard."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "python"
            # Use an executable script so the guard can be staged beside it.
            binary.write_text(f"#!{sys.executable}\n{source}\n")
            binary.chmod(0o755)
            supervisor = Path(
                os.environ.get("SPECFERRY_TEST_SUPERVISOR", "build/host/bin/specferry_supervise")
            ).resolve()
            shutil.copy2(supervisor, root / "specferry_supervise")
            with patch.object(device, "RECOVERY_ROOT", root / "recovery"):
                result = device.run_device(binary, [], root / "result", root, timeout, **options)
            marker_path = root / "recovery/np101-recovery-required.json"
            marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
            return result, marker

    def test_timing_mode_is_explicit_and_summary_is_separate_from_call_tracing(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.run_fixture(sdk_timing=True, sdk_summary=True)
        for options, expected in (
            ({}, "off"),
            ({"sdk_timing": True}, "calls"),
            ({"sdk_summary": True}, "summary"),
        ):
            with (
                self.subTest(options=options),
                patch.dict(os.environ, SPECFERRY_SDK_TIMING="calls"),
            ):
                result, marker = self.run_fixture(
                    f"import os; assert os.environ['SPECFERRY_SDK_TIMING'] == {expected!r}",
                    **options,
                )
                self.assertEqual(result["sdk_summary"], bool(options.get("sdk_summary")))
                self.assertTrue(device.clean_execution(result))
                self.assertIsNone(marker)

    def test_timeout_stops_child_and_records_recovery(self):
        result, marker = self.run_fixture("import time; time.sleep(60)", timeout=0.15)
        self.assertTrue(result["timeout"])
        self.assertLess(result["returncode"], 0)
        self.assertTrue(result["process_group_exited"])
        self.assertTrue(result["device_recovery_required"])
        self.assertEqual(marker["members"], [])

    def test_blocking_health_sample_cannot_delay_execution_deadline(self):
        with patch.object(device, "HEALTH_SAMPLE_SECONDS", 0):
            result, marker = self.run_fixture(
                "import time; time.sleep(60)",
                timeout=0.15,
                abort_check=lambda: time.sleep(0.6),
            )
        self.assertTrue(result["timeout"])
        actions = result["supervision"]["actions"]
        delay = (
            actions[0]["monotonic_seconds"] - result["supervision"]["child_start_monotonic_seconds"]
        )
        self.assertLess(delay, 0.5)
        self.assertEqual(marker["members"], [])

    def test_health_failure_stops_child_and_preserves_gate(self):
        with patch.object(device, "HEALTH_SAMPLE_SECONDS", 0):
            result, marker = self.run_fixture(
                "import time; time.sleep(60)",
                abort_check=lambda: True,
            )
        self.assertFalse(result["timeout"])
        self.assertIn("kernel fault", result["health_monitor_failure"])
        self.assertTrue(result["process_group_exited"])
        self.assertIsNotNone(marker)

    def test_monitor_exception_stops_child_and_preserves_gate(self):
        def broken():
            raise OSError("lost journal")

        with patch.object(device, "HEALTH_SAMPLE_SECONDS", 0):
            result, marker = self.run_fixture("import time; time.sleep(60)", abort_check=broken)
        self.assertIn("lost journal", result["health_monitor_failure"])
        self.assertTrue(result["process_group_exited"])
        self.assertIsNotNone(marker)

    def test_keyboard_interrupt_stops_child_and_preserves_gate(self):
        def interrupted():
            raise KeyboardInterrupt()

        with patch.object(device, "HEALTH_SAMPLE_SECONDS", 0):
            result, marker = self.run_fixture(
                "import time; time.sleep(60)", abort_check=interrupted
            )
        self.assertTrue(result["interrupted_by_user"])
        self.assertFalse(device.clean_execution(result))
        self.assertEqual(marker["members"], [])

    def test_timeout_requires_recovery_even_if_signal_handler_exits_successfully(self):
        result, marker = self.run_fixture(
            "import signal,time,sys; signal.signal(signal.SIGTERM, lambda *args: sys.exit(0)); time.sleep(60)",
            timeout=0.3,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["timeout"])
        self.assertTrue(result["device_recovery_required"])
        self.assertEqual(marker["members"], [])

    def test_signal_exit_requires_recovery_but_normal_api_error_does_not(self):
        for source, code, recovery in (
            ("import os,signal; os.kill(os.getpid(),signal.SIGTERM)", -signal.SIGTERM, True),
            ("raise SystemExit(3)", 3, False),
            ("pass", 0, False),
        ):
            with self.subTest(returncode=code):
                result, marker = self.run_fixture(source)
                self.assertEqual(result["returncode"], code)
                self.assertEqual(result["device_recovery_required"], recovery)
                self.assertEqual(marker is not None, recovery)

    def test_profiler_is_explicit_and_cannot_instrument_recovery_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capture.vpd"
            result, _ = self.run_fixture(profiler_output=output)
            self.assertTrue(result["profiler"]["requested"])
            for option in ("recovery_control", "sdk_timing", "sdk_summary", "trace_driver"):
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, "separate"):
                    self.run_fixture(profiler_output=output, **{option: True})
            output.write_bytes(b"original evidence")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                self.run_fixture(profiler_output=output)
            self.assertEqual(output.read_bytes(), b"original evidence")


if __name__ == "__main__":
    unittest.main()
