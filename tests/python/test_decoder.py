"""Decoder acceptance contracts, without device access or downloaded weights."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.qwen3_5.precision import TOLERANCES
from specferry.models.qwen3_5.validation_decoder import DecoderCache, captured_inputs, evaluate


class DecoderAcceptanceTests(unittest.TestCase):
    def test_inputs_use_pre_norm_layer_boundary_and_reject_invalid_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            valid = np.ones((1, 1, 1024), dtype=np.float16)
            pre_norm = "token.0001.layer.0.input.0"
            mixed = "token.0001.layer.0.linear_attn.in_proj_qkv.input.0"
            np.savez(path, **{pre_norm: valid, mixed: valid * 2})
            inputs, names = captured_inputs(path, 2, 0)
            self.assertEqual(names, [pre_norm, pre_norm])
            np.testing.assert_array_equal(inputs[0].numpy(), valid)
            for steps, first in ((1, 0), (513, 0), (2, 1)):
                with self.assertRaises(ValueError):
                    captured_inputs(path, steps, first)
            for value in (valid.astype(np.float32), valid.reshape(1, -1), valid * np.nan):
                np.savez(path, **{pre_norm: value})
                with self.assertRaises(ValueError):
                    captured_inputs(path, 2, 0)
            np.savez(path, **{mixed: valid})
            with self.assertRaises(ValueError):
                captured_inputs(path, 2, 0)

    def test_reference_reset_clears_each_recurrence_but_preserves_stale_kv(self):
        cache = DecoderCache()
        for index, layer in enumerate(cache.layers.values()):
            layer.recurrent_states[0].fill_(index + 1)
            layer.conv_states[0].fill_(index + 1)
        cache.attention[3].keys.fill_(7)
        cache.length = 12
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertTrue(bool((cache.attention[3].keys == 7).all()))
        for layer in cache.layers.values():
            self.assertEqual(layer.recurrent_states[0].count_nonzero().item(), 0)
            self.assertEqual(layer.conv_states[0].count_nonzero().item(), 0)
        cache.layers[0].recurrent_states[0].fill_(9)
        self.assertEqual(cache.layers[1].recurrent_states[0].count_nonzero().item(), 0)

    def test_numerical_success_requires_lifecycle_and_exact_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, actual = Path(directory) / "fixture", Path(directory) / "actual"
            fixture.mkdir()
            actual.mkdir()
            value = np.zeros((1,), dtype="<f2")
            spec = {"bytes": value.nbytes, "dtype": value.dtype.str, "shape": list(value.shape)}
            for parent in (fixture, actual):
                for sequence in ("zero", "final"):
                    (parent / f"{sequence}.layer.3.output.1.bin").write_bytes(value.tobytes())
            metadata = {
                "steps": 2,
                "layers": [3],
                "tolerances": TOLERANCES,
                "memory_payload": {},
                "observations": [
                    {"sequence": name, "index": 1, "outputs": {"layer.3.output": spec}}
                    for name in ("zero", "final")
                ],
            }
            evidence = {
                "returncode": 0,
                "process_group_exited": True,
                "device_recovery_required": False,
            }
            execution = {
                "status": "executed",
                "phase": "complete",
                "completed_steps": 8,
                "completed_sequences": 4,
                "cache_writes": 8,
                "step_uploads": 24,
                "step_upload_bytes": 16448,
                "step_reads": 0,
                "invalid_input_rejected": True,
                "empty_output_rejected": True,
                "released": True,
            }
            report = actual / "execution.json"
            report.write_text(json.dumps(execution))
            result = evaluate(fixture, actual, metadata, evidence)
            self.assertEqual(result["status"], "numerical_pass")
            self.assertFalse(result["hardware_execution_proven"])
            self.assertFalse(result["device_residency_verified"])
            execution["step_reads"] = 1
            report.write_text(json.dumps(execution))
            self.assertEqual(evaluate(fixture, actual, metadata, evidence)["status"], "failed")
            execution["step_reads"] = 0
            report.write_text(json.dumps(execution))
            # Within the CPU tolerance, but final-only and observed SDK runs must agree exactly.
            changed = actual / "final.layer.3.output.1.bin"
            changed.write_bytes(np.array([0.001], dtype="<f2").tobytes())
            self.assertEqual(evaluate(fixture, actual, metadata, evidence)["status"], "failed")
            changed.write_bytes(value.tobytes())
            execution["released"] = False
            report.write_text(json.dumps(execution))
            self.assertEqual(evaluate(fixture, actual, metadata, evidence)["status"], "failed")
            execution["released"] = True
            report.write_text(json.dumps(execution))
            changed.unlink()
            self.assertEqual(evaluate(fixture, actual, metadata, evidence)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
