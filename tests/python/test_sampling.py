"""A successful SDK call must not hide invalid categorical samples."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.validation.sampling import evaluate


class SamplingReportTests(unittest.TestCase):
    def test_rejects_invalid_and_distribution_ignoring_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "execution.json").write_text('{"status":"executed","released":true}')
            evidence = {
                "returncode": 0,
                "process_group_exited": True,
                "device_recovery_required": False,
            }
            count = 4096
            uniform = np.tile(np.arange(8, dtype="<i4"), count // 8)
            weighted = np.repeat(
                np.arange(8, dtype="<i4"), [114, 228, 341, 455, 569, 683, 796, 910]
            )
            for name, array in {
                "uniform.0": uniform,
                "uniform.1": uniform[::-1],
                "uniform.2": uniform,
                "counter": np.roll(uniform, 1),
                "fresh": uniform,
                "weighted": weighted,
                "last": np.full(count, 7, dtype="<i4"),
            }.items():
                array.tofile(directory / f"{name}.bin")

            def check(change=None):
                return evaluate(directory, "F16_TO_F32", 8, count, evidence | (change or {}))

            self.assertEqual(check()["status"], "numerical_pass")
            self.assertEqual(check({"returncode": -11})["status"], "failed")
            self.assertEqual(check({"process_group_exited": False})["status"], "failed")
            self.assertEqual(check({"device_recovery_required": True})["status"], "failed")
            for value in (8, 0):
                np.full(count, value, dtype="<i4").tofile(directory / "weighted.bin")
                self.assertEqual(check()["status"], "failed")
            (directory / "weighted.bin").write_bytes(b"")
            self.assertFalse(check()["checks"]["weighted.range"]["passed"])


if __name__ == "__main__":
    unittest.main()
