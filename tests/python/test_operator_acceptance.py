"""Protect completion, readback, and memory-budget gates from false acceptance."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.qwen3_5.operator_cases import catalog
from specferry.models.qwen3_5.precision import TOLERANCES
from specferry.validation import capabilities


class AcceptanceTests(unittest.TestCase):
    def test_final_readback_checks_every_cycle_and_the_last_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = catalog()["matmul_fp16_small"].build().write(root / "fixture")
            metadata.update(cycles=2, readback_mode="final")
            output = root / "output"
            output.mkdir()
            correct = (root / "fixture/y.expected.1.bin").read_bytes()
            (output / "cycle.0.y.1.bin").write_bytes(correct)
            results = capabilities.compare_outputs(metadata, root / "fixture", output, TOLERANCES)
            self.assertEqual(set(results), {"cycle.0.y.1", "cycle.1.y.1"})
            self.assertTrue(results["cycle.0.y.1"]["passed"])
            self.assertFalse(results["cycle.1.y.1"]["passed"])
            (output / "cycle.1.y.1.bin").write_bytes(
                (root / "fixture/y.expected.0.bin").read_bytes()
            )
            results = capabilities.compare_outputs(metadata, root / "fixture", output, TOLERANCES)
            self.assertFalse(results["cycle.1.y.1"]["passed"])

    def test_budget_blocks_device_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(capabilities, "run_device") as run:
                result = capabilities.check_case(
                    catalog()["weights_mixed_fp16"],
                    root / "case",
                    root / "binary",
                    root,
                    timeout=1,
                    max_payload_bytes=1,
                )
            run.assert_not_called()
            self.assertEqual(result["status"], "blocked")
            self.assertIn("tensor_payload_budget_exceeded", result["blockers"])

    def run_with_report(self, native_report, log=""):
        def fake_device(binary, arguments, output, sdk_lib, timeout, **options):
            output.mkdir(parents=True)
            fixture = Path(arguments[0]).parent
            metadata = json.loads((fixture / "case.json").read_text())
            for name in metadata["outputs"]:
                for step in range(metadata["steps"]):
                    (output / f"{name}.{step}.bin").write_bytes(
                        (fixture / f"{name}.expected.{step}.bin").read_bytes()
                    )
            (output / "execution.json").write_text(json.dumps(native_report))
            (output / "sdk.log").write_text(log)
            return {
                "returncode": 0,
                "device_recovery_required": False,
                "driver_activity_observed": True,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(capabilities, "run_device", side_effect=fake_device):
                return capabilities.check_case(
                    catalog()["weights_mutable_fp16"], root / "case", root / "binary", root, 1
                )

    def test_matching_outputs_without_completed_release_cannot_pass(self):
        result = self.run_with_report({"status": "running", "phase": "release"})
        self.assertEqual(result["acceptance"]["numerical"], "passed")
        self.assertEqual(result["acceptance"]["lifecycle"], "failed")
        self.assertEqual(result["status"], "failed")

    def test_shader_failure_and_driver_calls_do_not_prove_hardware_execution(self):
        result = self.run_with_report(
            {
                "status": "executed",
                "completed_steps": 3,
                "completed_cycles": 1,
                "readback_mode": "each-step",
            },
            "ERROR: Failed to compile vx shader.\n",
        )
        self.assertEqual(result["status"], "numerical_pass")
        self.assertIn("sdk_kernel_compilation_failed", result["blockers"])
        self.assertEqual(result["acceptance"]["hardware"], "unverified")
        self.assertFalse(result["hardware_execution_proven"])

    def test_matching_outputs_do_not_prove_device_residency(self):
        result = self.run_with_report(
            {
                "status": "executed",
                "completed_steps": 3,
                "completed_cycles": 1,
                "readback_mode": "each-step",
            },
        )
        self.assertEqual(result["status"], "numerical_pass")
        self.assertEqual(result["acceptance"]["device_residency"], "unverified")


if __name__ == "__main__":
    unittest.main()
