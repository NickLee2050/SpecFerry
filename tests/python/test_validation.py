"""Numerical regressions for the compact reporting and graph-cache checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.opt.graph_validation import evaluate
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

    def test_graph_cache_uses_valid_prefix_and_requires_exact_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture, device = root / "fixture", root / "device"
            fixture.mkdir()
            device.mkdir()
            metadata = {
                "layers": 1,
                "heads": 1,
                "width": 2,
                "capacity": 4,
                "requests": [{"tokens": [1, 2], "consumed": 2, "prompt_length": 1}] * 3,
            }
            for i in range(3):
                (device / f"measured.{i}.json").write_text(
                    json.dumps(
                        {"tokens": [1, 2], "consumed": 2, "launches": 2, "uploads": 2, "reads": 2}
                    )
                )
                for name in ("keys", "values"):
                    np.ones((1, 2, 2), dtype="<f2").tofile(fixture / f"expected.{i}.0.{name}.bin")
                    data = np.ones((1, 4, 2), dtype="<f2")
                    data[:, 2:] = 99 * i  # Stale suffix must not participate.
                    data.tofile(device / f"measured.{i}.layer.0.{name}.bin")
            checks, _ = evaluate(fixture, device, metadata, 1)
            self.assertTrue(all(checks.values()))
            path = device / "measured.2.layer.0.keys.bin"
            data = np.fromfile(path, dtype="<f2")
            data[0] += 0.001  # Within CPU tolerance, but invalid exact reset.
            data.tofile(path)
            checks, _ = evaluate(fixture, device, metadata, 1)
            self.assertTrue(checks["measured.2.layer.0.keys"])
            self.assertFalse(checks["measured.2.layer.0.keys.reset"])
            data[:] = 0
            data.tofile(path)
            self.assertFalse(evaluate(fixture, device, metadata, 1)[0]["measured.2.layer.0.keys"])

    def test_nonfinite_and_wrong_shape_cannot_pass(self):
        reference = np.ones(4)
        for actual in (np.zeros(4), np.ones(3), np.full(4, np.nan)):
            self.assertFalse(compare_arrays(actual, reference, 0.08, 0.02)["passed"])


if __name__ == "__main__":
    unittest.main()
