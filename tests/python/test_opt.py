"""OPT import, storage and acceptance guards without a model download or device."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.export.weights import write_weight_pack
from specferry.models.opt.checkpoint import load_checkpoint
from specferry.models.opt.config import CHECKPOINT_SHA256, TOLERANCES, Components
from specferry.models.opt.validation import compare_prefix, evaluate, load_inputs


class OptContractTests(unittest.TestCase):
    def test_invalid_shapes_capacity_and_layer_selection_are_rejected(self):
        config = Components(64, 96, 4, 24, 8)
        for changes in (
            {"hidden": 63},
            {"heads": 0},
            {"capacity": 513},
            {"hidden": True},
            {"intermediate": 1000000},
            {"epsilon": float("nan")},
            {"epsilon": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
        for selected in ([], [0, 0], [24], [-1], [False]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                config.select(selected)

    def test_checkpoint_rejects_missing_bias_wrong_precision_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "download-manifest.json").write_text("{}")
            manifest = {
                "files": [
                    {"path": "config.json"},
                    {"path": "pytorch_model.bin", "sha256": CHECKPOINT_SHA256},
                ]
            }
            valid = {
                "projection.weight": torch.ones(2, 2).half(),
                "projection.bias": torch.zeros(2).half(),
            }
            with (
                patch("specferry.models.opt.checkpoint.verify_download", return_value=manifest),
                patch(
                    "specferry.models.opt.checkpoint.checkpoint_shapes",
                    return_value={
                        "projection.weight": [2, 2],
                        "projection.bias": [2],
                    },
                ),
            ):
                torch.save(valid, root / "pytorch_model.bin")
                _, metadata = load_checkpoint(root)
                self.assertEqual(metadata["weight_bytes"], 12)
                for weights in (
                    {"projection.weight": valid["projection.weight"]},
                    valid | {"projection.bias": torch.zeros(2)},
                    valid | {"projection.weight": torch.ones(4).half()},
                    valid | {"projection.bias": torch.tensor([float("nan"), 0]).half()},
                ):
                    torch.save(weights, root / "pytorch_model.bin")
                    with self.assertRaises(ValueError):
                        load_checkpoint(root)

    def test_opaque_checkpoint_chunks_preserve_fp16_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Signed zero and subnormal values must survive packing unchanged.
            raw = np.array([0, 0x8000, 1, 0x8001, 0x3C01, 0xBC01], dtype="<u2").tobytes()
            entry = {
                "source_name": "decoder.weight",
                "text_name": "decoder.weight",
                "file": "pytorch_model.bin",
                "dtype": "F16",
                "target_dtype": "F16",
                "shape": [2, 3],
            }

            def blocks(source, record, limit):
                for offset in range(0, len(raw), limit):
                    yield memoryview(raw)[offset : offset + limit]

            records = write_weight_pack(root, [entry], root, chunk_bytes=4, block_reader=blocks)
            self.assertEqual((root / "weights.bin").read_bytes(), raw)
            self.assertIsNone(records[0]["source_data_offsets"])
            self.assertFalse(records[0]["transformed"])

    def test_boundary_inputs_use_numeric_token_order_and_reject_wrong_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            config = Components(4, 8, 2, 24, 8)
            one = np.ones((1, 1, 4), dtype=np.float16)
            np.savez(
                path,
                **{
                    "prompt.0.token.10.layer.23.input": one * 10,
                    "prompt.0.token.2.layer.23.input": one * 2,
                    "prompt.0.token.2.layer.23.output": one * 99,
                },
            )
            inputs, names = load_inputs(path, config, 23, 3, 0)
            self.assertIn("token.2.", names[0])
            self.assertEqual([value[0, 0, 0] for value in inputs], [2, 10, 2])
            np.savez(path, **{"prompt.0.token.0.layer.23.input": one.astype(np.float32)})
            with self.assertRaises(ValueError):
                load_inputs(path, config, 23, 2, 0)


class OptAcceptanceTests(unittest.TestCase):
    def test_only_valid_cache_prefix_participates_in_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            actual, expected = Path(directory) / "actual", Path(directory) / "expected"
            value = np.zeros((1, 2, 4, 2), dtype="<f2")
            expected.write_bytes(value.tobytes())
            value[:, :, 2:] = 99
            actual.write_bytes(value.tobytes())
            spec = {"shape": list(value.shape), "dtype": "<f2", "bytes": value.nbytes}
            exact = {"atol": 0, "rtol": 0}
            self.assertTrue(compare_prefix(actual, expected, spec, 2, exact)["passed"])
            value[:, :, 1] = 99
            actual.write_bytes(value.tobytes())
            self.assertFalse(compare_prefix(actual, expected, spec, 2, exact)["passed"])

    def test_sdk_exit_mask_and_repeat_are_required_even_when_values_are_close(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, actual = Path(directory) / "fixture", Path(directory) / "actual"
            fixture.mkdir()
            actual.mkdir()
            value = np.array([[[[1, 0, 0, 0]]]], dtype="<f2")
            spec = {"shape": list(value.shape), "dtype": "<f2", "bytes": value.nbytes}
            for parent in (fixture, actual):
                for sequence in ("zero", "reset"):
                    (parent / f"{sequence}.layer.0.probabilities.0.bin").write_bytes(
                        value.tobytes()
                    )
            metadata = {
                "steps": 2,
                "layers": [0],
                "config": {"hidden": 4, "capacity": 4},
                "weight_bytes": 0,
                "state_bytes": 64,
                "tolerances": TOLERANCES,
                "observations": [
                    {
                        "sequence": name,
                        "index": 0,
                        "length": 1,
                        "outputs": {"layer.0.probabilities": spec},
                    }
                    for name in ("zero", "reset")
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
                "released": True,
                "steps": 8,
                "sequences": 4,
                "cache_writes": 8,
                "uploads": 16,
                "upload_bytes": 96,
                "reads": 0,
                "invalid_input_rejected": True,
                "empty_output_rejected": True,
            }
            (actual / "execution.json").write_text(json.dumps(execution))
            result = evaluate(fixture, actual, metadata, evidence)
            self.assertEqual(result["status"], "numerical_pass")
            self.assertFalse(result["hardware_execution_proven"])
            for change in (
                {"released": False},
                {"reads": 1},
                {"cache_writes": 7},
                {"upload_bytes": 100},
                {"phase": "initialize"},
            ):
                (actual / "execution.json").write_text(json.dumps(execution | change))
                self.assertEqual(evaluate(fixture, actual, metadata, evidence)["status"], "failed")
            (actual / "execution.json").write_text(json.dumps(execution))
            changed = actual / "reset.layer.0.probabilities.0.bin"
            value[..., 0] = 0.999
            changed.write_bytes(value.tobytes())
            self.assertFalse(
                evaluate(fixture, actual, metadata, evidence)["checks"][
                    "repeat.reset.layer.0.probabilities.0"
                ]["passed"]
            )
            value[..., 0] = 1
            value[..., 3] = 0.001
            changed.write_bytes(value.tobytes())
            self.assertFalse(
                evaluate(fixture, actual, metadata, evidence)["checks"][
                    "mask.reset.layer.0.probabilities.0"
                ]["passed"]
            )


if __name__ == "__main__":
    unittest.main()
