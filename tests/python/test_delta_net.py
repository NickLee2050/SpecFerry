"""Guard the layer acceptance contract without requiring weights or a device."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.qwen3_5.precision import TOLERANCES
from specferry.models.qwen3_5.validation_delta_net import OUTPUTS, captured_inputs, evaluate


class DeltaNetAcceptanceTests(unittest.TestCase):
    def test_trace_rejects_wrong_precision_shape_and_nonfinite_values(self):
        key = "token.0001.layer.0.linear_attn.in_proj_qkv.input.0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            valid = np.ones((1, 1, 1024), dtype=np.float16)
            np.savez(path, **{key: valid})
            values, names = captured_inputs(path, 2)
            self.assertEqual(names, [key, key])
            self.assertEqual(len(values), 2)
            for invalid in (valid.astype(np.float32), valid.reshape(1, -1), valid * np.nan):
                with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                    np.savez(path, **{key: invalid})
                    with self.assertRaises(ValueError):
                        captured_inputs(path, 2)

    def test_state_reset_final_readback_and_release_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, device = root / "fixture", root / "device"
            fixture.mkdir()
            device.mkdir()
            specs = {name: {"dtype": "<f4", "bytes": 4, "shape": [1]} for name in OUTPUTS}
            metadata = {"steps": 2, "outputs": specs, "tolerances": TOLERANCES}
            for name in OUTPUTS:
                for step in range(2):
                    data = np.array([step + 1], dtype="<f4").tobytes()
                    for sequence in ("zero", "nonzero"):
                        (fixture / f"{sequence}.{name}.{step}.bin").write_bytes(data)
                    for sequence in ("zero", "nonzero", "reset", "fresh", "final"):
                        (device / f"{sequence}.{name}.{step}.bin").write_bytes(data)
            execution = {
                "status": "executed",
                "phase": "complete",
                "completed_sequences": 5,
                "completed_steps": 2,
                "per_step_host_state_uploads": 0,
                "application_readbacks": 9 * len(OUTPUTS),
            }
            report = device / "execution.json"
            report.write_text(json.dumps(execution))
            evidence = {
                "returncode": 0,
                "process_group_exited": True,
                "device_recovery_required": False,
            }
            result = evaluate(fixture, device, metadata, evidence)
            self.assertTrue(result["state_reuse_numerical"])
            self.assertFalse(result["device_residency_verified"])
            for name in ("reset.recurrent.1", "final.convolution.1", "fresh.output.1"):
                with self.subTest(name=name):
                    path = device / f"{name}.bin"
                    original = path.read_bytes()
                    path.write_bytes(np.array([1], dtype="<f4").tobytes())
                    self.assertEqual(
                        evaluate(fixture, device, metadata, evidence)["status"], "failed"
                    )
                    path.write_bytes(original)
            execution["phase"] = "release"
            report.write_text(json.dumps(execution))
            self.assertFalse(evaluate(fixture, device, metadata, evidence)["lifecycle"])


if __name__ == "__main__":
    unittest.main()
