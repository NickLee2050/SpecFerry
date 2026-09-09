"""Probe comparisons must reject stale, nonfinite, truncated, and wrong-index outputs."""

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from scripts import check_np101_operators

from specferry.validation import capabilities
from specferry.validation.capabilities import compare_arrays, compare_outputs
from specferry.validation.operator_cases import catalog


class ComparisonTests(unittest.TestCase):
    def test_interrupted_run_stops_suite_even_with_matching_outputs_and_zero_exit(self):
        def interrupted_run(binary, arguments, output, sdk_lib, timeout):
            fixture = Path(arguments[0]).parent
            metadata = json.loads((fixture / "case.json").read_text())
            output.mkdir(parents=True)
            for name in metadata["outputs"]:
                for step in range(metadata["steps"]):
                    expected = fixture / f"{name}.expected.{step}.bin"
                    (output / f"{name}.{step}.bin").write_bytes(expected.read_bytes())
            return {
                "returncode": 0,
                "timeout": True,
                "process_group_exited": True,
                "device_recovery_required": True,
                "driver_activity_observed": False,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"fixture")
            output = root / "results"
            arguments = [
                "check_np101_operators",
                "--diagnostic",
                "--binary",
                str(binary),
                "--case",
                "matmul_fp16_small",
                "--case",
                "gather_small",
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(check_np101_operators, "device_lock", return_value=nullcontext()),
                patch.object(capabilities, "run_device", side_effect=interrupted_run) as run,
            ):
                self.assertEqual(check_np101_operators.main(), 2)
            run.assert_called_once()
            report = json.loads((output / "op-capabilities.json").read_text())
            self.assertFalse(report["numeric_pass"])
            self.assertIn("gather_small", report["pending_cases"])
            result = report["cases"][0]
            self.assertTrue(all(item["passed"] for item in result["comparisons"].values()))
            self.assertEqual(result["status"], "failed")
            self.assertIn("device_recovery_required", result["blockers"])

    def test_prepare_only_does_not_acquire_the_device_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            arguments = [
                "check_np101_operators",
                "--prepare-only",
                "--case",
                "argmax_small",
                "--output",
                str(output),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(
                    check_np101_operators, "device_lock", side_effect=RuntimeError("device busy")
                ) as lock,
            ):
                self.assertEqual(check_np101_operators.main(), 0)
            lock.assert_not_called()
            report = json.loads((output / "op-capabilities.json").read_text())
            self.assertEqual(report["status"], "prepared")
            self.assertEqual(len(report["cases"]), 1)
            self.assertTrue((output / "argmax_small/fixture/graph.txt").is_file())

    def test_nonfinite_wrong_shape_and_error_outside_budget_fail(self):
        expected = np.array([1.0, 2.0], dtype=np.float32)
        for actual in (
            np.array([np.nan, 2]),
            np.array([1, np.inf]),
            np.array([1]),
            np.array([1.1, 2]),
        ):
            self.assertFalse(compare_arrays(actual, expected, 0.01, 0.02)["passed"])

    def test_integer_ids_require_exact_agreement(self):
        self.assertFalse(compare_arrays(np.array([248044]), np.array([248045]), 0, 0)["passed"])

    def test_missing_truncated_and_stale_step_outputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = catalog()["matmul_fp16_small"].build()
            metadata = fixture.write(root / "fixture")
            execution = root / "execution"
            execution.mkdir()
            result = compare_outputs(metadata, root / "fixture", execution)
            self.assertTrue(all(not item["passed"] for item in result.values()))
            first = (root / "fixture/y.expected.0.bin").read_bytes()
            (execution / "y.0.bin").write_bytes(first[:-1])
            (execution / "y.1.bin").write_bytes(first)
            result = compare_outputs(metadata, root / "fixture", execution)
            self.assertFalse(result["y.0"]["passed"])
            self.assertFalse(result["y.1"]["passed"])

    def test_all_small_fixtures_have_finite_fixed_byte_expectations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in catalog().values():
                if case.scale != "small":
                    continue
                with self.subTest(case=case.name):
                    fixture = case.build()
                    metadata = fixture.write(root / case.name)
                    self.assertGreaterEqual(metadata["steps"], 2)
                    self.assertTrue(metadata["outputs"])
                    self.assertEqual(metadata["provenance"]["seed"], 101)
                    json.dumps(metadata, allow_nan=False)
                    for values in fixture.expected.values():
                        for value in values:
                            self.assertTrue(np.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
