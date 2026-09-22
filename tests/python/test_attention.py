"""Keep Attention acceptance independent of device access and downloaded weights."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.qwen3_5.precision import TOLERANCES
from specferry.models.qwen3_5.validation_attention import (
    OUTPUTS,
    captured_inputs,
    compare_prefix,
    evaluate,
)


class AttentionAcceptanceTests(unittest.TestCase):
    def test_inputs_reject_bad_shape_precision_nonfinite_and_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            name = "token.0001.layer.3.self_attn.q_proj.input.0"
            valid = np.ones((1, 1, 1024), dtype=np.float16)
            np.savez(path, **{name: valid})
            values, names = captured_inputs(path, 512)
            self.assertEqual(len(values), 512)
            self.assertEqual(names, [name] * 512)
            for steps in (1, 513):
                with self.assertRaises(ValueError):
                    captured_inputs(path, steps)
            for value in (valid.astype(np.float32), valid.reshape(1, -1), valid * np.nan):
                np.savez(path, **{name: value})
                with self.assertRaises(ValueError):
                    captured_inputs(path, 2)

    def test_reset_repeat_ignores_only_the_stale_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left", Path(directory) / "right"
            initial = np.zeros((1, 2, 512, 256), dtype="<f2")
            spec = {"shape": list(initial.shape), "bytes": initial.nbytes, "dtype": "<f2"}
            left.write_bytes(initial.tobytes())
            initial[:, :, 1:] = 9
            right.write_bytes(initial.tobytes())
            self.assertTrue(compare_prefix(left, right, spec, 1)["passed"])
            initial[0, 1, 0, 255] = 1
            right.write_bytes(initial.tobytes())
            self.assertFalse(compare_prefix(left, right, spec, 1)["passed"])

    def test_missing_outputs_mask_leak_and_incomplete_release_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, device = Path(directory) / "fixture", Path(directory) / "device"
            fixture.mkdir()
            device.mkdir()
            outputs = {}
            for field in OUTPUTS:
                value = np.zeros(
                    (1, 2, 512, 256)
                    if field in ("keys", "values")
                    else (1, 8, 1, 512)
                    if field == "probabilities"
                    else (1,),
                    dtype="<f2",
                )
                outputs[field] = {"shape": list(value.shape), "bytes": value.nbytes, "dtype": "<f2"}
                for parent in (fixture, device):
                    (parent / f"zero.{field}.0.bin").write_bytes(value.tobytes())
            metadata = {
                "steps": 512,
                "tolerances": TOLERANCES,
                "observations": [{"sequence": "zero", "index": 0, "length": 1, "outputs": outputs}],
            }
            execution = {
                "status": "executed",
                "phase": "complete",
                "completed_sequences": 5,
                "completed_steps": 1042,
                "cache_writes": 1042,
                "invalid_truncate_rejected": True,
                "capacity_rejected": True,
                "cache_copies": 1,
                "cache_payload_bytes": 1048576,
                "slot_write_payload_bytes": 2048,
                "per_step_host_kv_transfers": False,
            }
            evidence = {
                "returncode": 0,
                "process_group_exited": True,
                "device_recovery_required": False,
            }
            report = device / "execution.json"
            report.write_text(json.dumps(execution))
            result = evaluate(fixture, device, metadata, evidence)
            self.assertTrue(result["cache_reuse_numerical"])
            self.assertFalse(result["device_residency_verified"])
            # Below the numerical tolerance, but masked probabilities must be exactly zero.
            path = device / "zero.probabilities.0.bin"
            original = path.read_bytes()
            value = np.frombuffer(original, dtype="<f2").copy()
            value[-1] = 0.001
            path.write_bytes(value.tobytes())
            self.assertEqual(evaluate(fixture, device, metadata, evidence)["status"], "failed")
            path.write_bytes(original)
            (device / "zero.key.0.bin").unlink()
            self.assertEqual(evaluate(fixture, device, metadata, evidence)["status"], "failed")
            execution["phase"] = "release"
            report.write_text(json.dumps(execution))
            self.assertFalse(evaluate(fixture, device, metadata, evidence)["lifecycle"])


if __name__ == "__main__":
    unittest.main()
