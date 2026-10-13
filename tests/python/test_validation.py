"""Numerical regressions for the compact reporting and graph-cache checks."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.opt.metrics import summarize
from specferry.validation.allocation import MIB, capacity_result, check_segment_budget
from specferry.validation.arrays import compare_arrays


class ValidationTests(unittest.TestCase):
    def test_capacity_is_not_integrity_and_signals_are_neither(self):
        evidence = {
            "returncode": 0,
            "timeout": False,
            "process_group_exited": True,
            "remaining_processes": [],
            "device_recovery_required": False,
        }
        report = {
            "status": "target_reached",
            "phase": "complete",
            "released": True,
            "target_bytes": 64 * MIB,
            "uploaded_bytes": 64 * MIB,
            "verified_bytes": 0,
            "storage": "constant",
            "dtype": "F16",
            "readback_mode": "none",
        }

        def result(r=report, e=evidence, mode="none"):
            return capacity_result(r, e, 64 * MIB, "constant", "F16", mode)

        self.assertTrue(result()["allocation_and_release_passed"])
        self.assertFalse(result()["readback_and_release_passed"])
        broken = report | {
            "status": "readback_mismatch",
            "readback_mode": "all",
            "readback_error": "byte_mismatch",
        }
        self.assertTrue(
            result(broken, evidence | {"returncode": 1}, "all")["allocation_and_release_passed"]
        )
        self.assertFalse(
            result(broken, evidence | {"returncode": 1}, "all")["readback_and_release_passed"]
        )
        # A complete scan can find corruption and still cover every retained block,
        # including blocks after the first error and after a final allocation refusal.
        scanned = broken | {
            "uploaded_bytes": 24 * MIB,
            "verified_bytes": 16 * MIB,
            "retained_tensors": 3,
            "scanned_bytes": 24 * MIB,
            "scanned_blocks": 3,
            "size_mismatch_blocks": 0,
            "mismatched_bytes": 24,
            "mismatched_blocks": 1,
            "mismatched_ranges": 3,
            "rejected_block_bytes": 8 * MIB,
            "rejection_phase": "allocate",
        }
        completed = result(scanned, evidence | {"returncode": 1}, "all")
        self.assertTrue(completed["full_scan_completed"])
        self.assertTrue(completed["next_block_rejected"])
        self.assertFalse(completed["target_reached"])
        self.assertFalse(completed["readback_and_release_passed"])
        self.assertEqual(completed["matching_bytes"], 24 * MIB - 24)
        for changed in (
            {"scanned_bytes": 16 * MIB},
            {"scanned_blocks": 2},
            {"size_mismatch_blocks": 1},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(
                    result(scanned | changed, evidence | {"returncode": 1}, "all")[
                        "full_scan_completed"
                    ]
                )
        self.assertFalse(result(e=evidence | {"returncode": -11})["allocation_and_release_passed"])
        self.assertFalse(result(r=report | {"released": False})["allocation_and_release_passed"])
        with self.assertRaises(ValueError):
            check_segment_budget(1024 * MIB, 1, "mutable")
        check_segment_budget(1024 * MIB, 1024 * MIB, "constant")

    def test_throughput_counts_only_intervals_after_first_prediction(self):
        run = {
            "tokens": [2, 3, 4],
            "prefill_seconds": 10,
            "first_token_seconds": 12,
            "token_seconds": [0.25, 0.75],
            "total_seconds": 13,
        }
        self.assertEqual(summarize([run])["decode_tokens_per_second"], 2)
        self.assertEqual(summarize([run])["ttft_seconds"], 12)
        self.assertIsNone(
            summarize([run | {"tokens": [2], "token_seconds": []}])["decode_tokens_per_second"]
        )
        for times in ([float("nan"), 1], [-1, 1], [100, 100], []):
            with self.assertRaises(ValueError):
                summarize([run | {"token_seconds": times}])

    def test_nonfinite_and_wrong_shape_cannot_pass(self):
        reference = np.ones(4)
        for actual in (np.zeros(4), np.ones(3), np.full(4, np.nan)):
            self.assertFalse(compare_arrays(actual, reference, 0.08, 0.02)["passed"])
