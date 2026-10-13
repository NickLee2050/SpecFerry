"""Numerical regressions for the compact reporting and graph-cache checks."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.validation import device as device_runner
from specferry.validation import recovery_controls


class RecoveryControlTests(unittest.TestCase):
    def test_vendor_diagnostic_preserves_recovery_gate(self):
        for outcome in ("complete", "interrupted"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                marker = root / "np101-recovery-required.json"
                original = {"boot_id": "previous-boot", "reason": "previous device fault"}
                marker.write_text(json.dumps(original))
                options = SimpleNamespace(
                    prepare_only=False,
                    sudo_read_only=False,
                    cold_restarted=True,
                    timeout=None,
                )

                def run_control(binary, arguments, output, sdk, deadline, **kwargs):
                    self.assertTrue(marker.exists())
                    self.assertTrue(kwargs["recovery_control"])
                    with self.assertRaisesRegex(RuntimeError, "recovery"):
                        device_runner.require_recovered_device(root)
                    with self.assertRaisesRegex(RuntimeError, "another SpecFerry"):
                        with device_runner.device_lock(root, recovery_control=True):
                            self.fail("Recovery controls must retain the device lock")
                    if outcome == "interrupted":
                        raise RuntimeError("interrupted before diagnostic execution")
                    output.mkdir(parents=True)
                    return dict(
                        returncode=0,
                        timeout=False,
                        process_group_exited=True,
                        remaining_processes=[],
                        device_recovery_required=False,
                    )

                with (
                    patch.object(recovery_controls, "RECOVERY_ROOT", root),
                    patch.object(recovery_controls, "host_boot_id", return_value="new-boot"),
                    patch.object(recovery_controls, "vendor_control", return_value=Path("demo")),
                    patch.object(
                        recovery_controls, "snapshot", return_value=dict(faults=[], owners="")
                    ),
                    patch.object(recovery_controls, "run_device", side_effect=run_control),
                    patch.object(recovery_controls, "numerical_check", return_value=True),
                    patch.object(recovery_controls.time, "sleep"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if outcome == "interrupted":
                        with self.assertRaisesRegex(RuntimeError, "interrupted"):
                            recovery_controls.run_vendor_control(root / "output", options)
                    else:
                        self.assertEqual(
                            recovery_controls.run_vendor_control(root / "output", options),
                            0,
                        )
                self.assertTrue(marker.exists())
                self.assertEqual(
                    json.loads((root / "output/previous-recovery-marker.json").read_text()),
                    original,
                )

    def test_recovery_rejects_unconfirmed_or_same_boot_restart(self):
        for cold_restarted, previous_boot in ((False, "previous-boot"), (True, "new-boot")):
            with (
                self.subTest(cold_restarted=cold_restarted, previous_boot=previous_boot),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                marker = root / "np101-recovery-required.json"
                original = {"boot_id": previous_boot, "reason": "previous device fault"}
                marker.write_text(json.dumps(original))
                options = SimpleNamespace(
                    prepare_only=False,
                    sudo_read_only=False,
                    cold_restarted=cold_restarted,
                    timeout=None,
                )
                with (
                    patch.object(recovery_controls, "RECOVERY_ROOT", root),
                    patch.object(recovery_controls, "host_boot_id", return_value="new-boot"),
                    patch.object(recovery_controls, "vendor_control", return_value=Path("demo")),
                    patch.object(
                        recovery_controls, "snapshot", return_value=dict(faults=[], owners="")
                    ),
                    patch.object(recovery_controls, "run_device") as runner,
                    self.assertRaisesRegex(RuntimeError, "Confirm host and NP101 cold restart"),
                ):
                    recovery_controls.run_vendor_control(root / "output", options)
                runner.assert_not_called()
                self.assertEqual(json.loads(marker.read_text()), original)

    def test_running_child_is_stopped_when_health_monitor_reports_fault(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = Mock(pid=12345)
            process.wait.side_effect = [subprocess.TimeoutExpired("control", 5), -15]
            with (
                patch.object(device_runner, "RECOVERY_ROOT", root),
                patch.object(device_runner.subprocess, "Popen", return_value=process),
                patch.object(device_runner, "terminate_group") as terminate,
                patch.object(device_runner, "process_group_members", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                evidence = device_runner.run_device(
                    Path(sys.executable),
                    [],
                    root / "device",
                    root,
                    60,
                    abort_check=lambda: True,
                )
            terminate.assert_called_once()
            self.assertFalse(evidence["timeout"])
            self.assertTrue(evidence["device_recovery_required"])
            self.assertIn("kernel fault", evidence["health_monitor_failure"])
            self.assertTrue((root / "np101-recovery-required.json").exists())

    def test_kernel_page_fault_and_watchdog_are_fatal_but_hdma_wait_is_not(self):
        for line in (
            "kernel: BUG: unable to handle page fault for address: 0000000100000008",
            "kernel: Oops: 0000 [#1] SMP PTI",
            "kernel: [galcore]: NP[0] hang, automatic recovery.",
            "--------gckEDMA_StartTransfer---timeout---src:0x55c24e40",
        ):
            self.assertIsNotNone(recovery_controls.FAULT.search(line), line)
        self.assertIsNone(recovery_controls.FAULT.search("kernel: Waiting for HDMA Transfer..."))

    def test_delayed_fault_rejects_vendor_diagnostic_even_when_numbers_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = SimpleNamespace(
                prepare_only=False,
                sudo_read_only=False,
                cold_restarted=False,
                timeout=None,
            )
            evidence = dict(
                returncode=0,
                timeout=False,
                process_group_exited=True,
                remaining_processes=[],
                device_recovery_required=False,
            )
            healthy = dict(faults=[], owners="")
            faulty = dict(faults=["NP[0] hang"], owners="")

            def device_run(binary, arguments, output, sdk, deadline, **kwargs):
                output.mkdir(parents=True)
                return evidence

            with (
                patch.object(recovery_controls, "RECOVERY_ROOT", root),
                patch.object(recovery_controls, "vendor_control", return_value=Path("demo")),
                patch.object(recovery_controls, "snapshot", side_effect=[healthy, healthy, faulty]),
                patch.object(recovery_controls, "run_device", side_effect=device_run) as runner,
                patch.object(recovery_controls, "numerical_check", return_value=True),
                patch.object(recovery_controls.time, "sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(recovery_controls.run_vendor_control(root / "output", options), 1)
            self.assertEqual(runner.call_count, 1)
            self.assertTrue((root / "np101-recovery-required.json").exists())
            incident = json.loads((root / "output/vendor-conv/incident.json").read_text())
            self.assertEqual(incident["kernel_gate"]["faults"], faulty["faults"])

    def test_vendor_check_rejects_nan_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = Path(temporary)
            (device / "sdk.log").write_text("total=64 mismatches=0 max_abs_err=nan")
            self.assertFalse(recovery_controls.numerical_check(device))

    def test_prepare_only_never_checks_or_opens_device(self):
        with (
            patch.object(recovery_controls, "vendor_control", return_value=Path("demo")),
            patch.object(recovery_controls, "snapshot") as snapshot,
            patch.object(recovery_controls, "run_device") as runner,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                recovery_controls.run_vendor_control(
                    Path("unused"),
                    SimpleNamespace(prepare_only=True, timeout=None),
                ),
                0,
            )
        snapshot.assert_not_called()
        runner.assert_not_called()
