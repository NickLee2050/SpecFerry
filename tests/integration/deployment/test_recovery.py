"""Numerical regressions for the compact reporting and graph-cache checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.deploy.execute.np101 import recovery_controls


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

    def test_kernel_page_fault_and_watchdog_are_fatal_but_hdma_wait_is_not(self):
        for line in (
            "kernel: BUG: unable to handle page fault for address: 0000000100000008",
            "kernel: Oops: 0000 [#1] SMP PTI",
            "kernel: [galcore]: NP[0] hang, automatic recovery.",
            "kernel: ***   GPU DRV CONFIG   ***",
            "kernel: [galcore]: Stop driver to keep scene.",
            "--------gckEDMA_StartTransfer---timeout---src:0x55c24e40",
            "[428293.680670] mmc0: Timeout waiting for hardware interrupt.",
            "[20055.063127] mmc0: Timeout waiting for hardware cmd interrupt.",
            "mmc1: Timeout waiting for hardware data interrupt.",
            "EXT4-fs error (device mmcblk0p2): failed to read inode",
            "Buffer I/O error on dev mmcblk0p2, logical block 42",
            "blk_update_request: I/O error, dev mmcblk0, sector 4096",
        ):
            self.assertIsNotNone(recovery_controls.FAULT.search(line), line)
        self.assertIsNone(recovery_controls.FAULT.search("kernel: Waiting for HDMA Transfer..."))


class EpExecutionGateTests(unittest.TestCase):
    def test_operator_acceptance_keeps_lock_until_health_and_numbers_pass(self):
        from tools.deploy.execute.np101 import runner

        clean = dict(boot_id="test-boot", faults=[], owners="")
        evidence = dict(
            returncode=0,
            timeout=False,
            process_group_exited=True,
            remaining_processes=[],
            device_recovery_required=False,
        )
        cases = (
            ("healthy", clean, clean, True),
            ("immediate-fault", clean | {"faults": ["NP hang"]}, clean, True),
            ("delayed-fault", clean, clean | {"faults": ["Oops"]}, True),
            ("unreleased", clean, clean | {"owners": "123"}, True),
            ("numerical-failure", clean, clean, False),
            ("observation-interrupted", clean, clean, True),
        )
        for name, immediate, postflight, numbers_pass in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                recovery = root / ".cache/runs"
                (root / "docs").mkdir()
                (root / "docs/np101-recovery-incidents.md").write_text("fixture ledger")
                output = root / "output"
                events = []

                def require_lock():
                    with self.assertRaisesRegex(RuntimeError, "another SpecFerry"):
                        with runner.device_lock(recovery, recovery_control=True):
                            pass

                def observe(seconds):
                    self.assertEqual(seconds, 45)
                    require_lock()
                    events.append("observe")
                    if name == "observation-interrupted":
                        raise KeyboardInterrupt("interrupted observation")

                def evaluate(observed):
                    self.assertEqual(observed, evidence)
                    require_lock()
                    events.append("evaluate")
                    return {"status": "numerical_pass" if numbers_pass else "numerical_mismatch"}

                def execute(*args, **kwargs):
                    require_lock()
                    self.assertTrue(callable(kwargs["abort_check"]))
                    self.assertTrue((recovery / "np101-recovery-required.json").is_file())
                    events.append("execute")
                    return evidence

                with (
                    patch.object(runner, "ROOT", root),
                    patch.object(runner, "RECOVERY_ROOT", recovery),
                    patch.object(runner, "snapshot", side_effect=[clean, immediate, postflight]),
                    patch.object(runner, "run_device", side_effect=execute),
                    patch.object(runner.time, "sleep", side_effect=observe),
                ):
                    status = runner.run_check(
                        Path("unused"),
                        [],
                        output,
                        SimpleNamespace(
                            timeout=None, sdk_timing=False, trace_driver=False, sudo_read_only=True
                        ),
                        evaluate=evaluate,
                    )
                    self.assertEqual(status, int(name != "healthy"))
                    marker = recovery / "np101-recovery-required.json"
                    self.assertEqual(marker.exists(), name != "healthy")
                    report = json.loads((output / "result.json").read_text())
                    self.assertEqual(report["passed"], name == "healthy")
                    self.assertEqual(events[:2], ["execute", "observe"])
                    if name == "healthy":
                        self.assertTrue((output / "completed-run-marker.json").is_file())
                    else:
                        # Process success, including automatic driver recovery, never opens the gate.
                        with self.assertRaises(RuntimeError):
                            with runner.device_lock(recovery):
                                self.fail("faulted device admitted another run")

    def test_preflight_failure_prevents_launch(self):
        from tools.deploy.execute.np101 import runner

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs").mkdir()
            (root / "docs/np101-recovery-incidents.md").write_text("fixture ledger")
            with (
                patch.object(runner, "ROOT", root),
                patch.object(runner, "RECOVERY_ROOT", root / ".cache/runs"),
                patch.object(
                    runner, "snapshot", return_value={"faults": ["EDMA timeout"], "owners": ""}
                ),
                patch.object(runner, "run_device") as execute,
                self.assertRaisesRegex(RuntimeError, "preflight failed"),
            ):
                runner.run_guarded(
                    Path("unused"),
                    [],
                    root / "output",
                    root,
                    120,
                    evaluate_result=lambda evidence: {"passed": True},
                )
            execute.assert_not_called()
