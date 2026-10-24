"""Numerical regressions for the compact reporting and graph-cache checks."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.validation import device as device_runner
from specferry.validation import recovery_controls


class RecoveryControlTests(unittest.TestCase):
    def test_same_boot_power_cycle_only_acknowledges_recorded_fault_occurrences(self):
        old = "2026-10-14T18:25:45.057746+0800 host kernel: EDMA timeout"
        new = "2026-10-15T08:55:00.000000+0800 host kernel: EDMA timeout"
        previous = dict(boot_id="same", kernel_gate=dict(faults=[old], owners=""))
        preflight = dict(boot_id="same", faults=[old], owners="")
        baseline, filtered = recovery_controls.power_cycle_baseline(previous, preflight)
        self.assertTrue(recovery_controls.healthy(filtered))
        self.assertEqual(filtered["historical_faults"], [old])
        for faults in ([old, new], [old, old], ["EDMA timeout"]):
            with self.subTest(faults=faults), self.assertRaises(RuntimeError):
                recovery_controls.power_cycle_baseline(previous, preflight | dict(faults=faults))
        self.assertEqual(
            recovery_controls.apply_fault_baseline(preflight | dict(boot_id="new"), baseline)[
                "faults"
            ],
            [old],
        )

    def test_multiline_edma_fault_keeps_its_journal_timestamp(self):
        prefix = "2026-10-14T18:25:45.057746+0800 host kernel:"
        detail = "--------gckEDMA_StartTransfer---timeout---len:0x4----"
        normal = "2026-10-14T18:25:44.000000+0800 host kernel: Waiting for HDMA Transfer..."
        direct = "2026-10-14T18:25:49.000000+0800 host kernel: NP[0] hang"
        log = f"{normal}\n{prefix} \n    {detail}\n{direct}\n"
        self.assertEqual(recovery_controls.kernel_faults(log), [prefix + " " + detail, direct])

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
            "kernel: ***   GPU DRV CONFIG   ***",
            "kernel: [galcore]: Stop driver to keep scene.",
            "--------gckEDMA_StartTransfer---timeout---src:0x55c24e40",
        ):
            self.assertIsNotNone(recovery_controls.FAULT.search(line), line)
        self.assertIsNone(recovery_controls.FAULT.search("kernel: Waiting for HDMA Transfer..."))
