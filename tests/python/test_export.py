"""Export integrity, precision, tensor contracts, and memory accounting regressions."""

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.export.memory import memory_budget
from specferry.export.schema import expected_text_tensors, validate_text_entries
from specferry.export.weights import (
    convert_bytes,
    verify_weight_pack,
    write_native_index,
    write_weight_pack,
)


def source_tensor(root, name, values, dtype):
    raw = values.view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {name: {"dtype": dtype, "shape": list(values.shape), "data_offsets": [0, len(raw)]}}
    ).encode()
    filename = name.replace(".", "_") + ".safetensors"
    (root / filename).write_bytes(struct.pack("<Q", len(header)) + header + raw)
    return {
        "source_name": name,
        "text_name": name,
        "file": filename,
        "dtype": dtype,
        "shape": list(values.shape),
        "data_offsets": [0, len(raw)],
    }


def complete_manifest(root, records):
    manifest = {
        "format": "specferry-np101-weights-v1",
        "status": "complete",
        "tensors": records,
        "aliases": {"lm_head.weight": "model.embed_tokens.weight"},
        "files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
            for name in ("weights.bin", "weights.index")
        },
    }
    (root / "deployment-manifest.json").write_text(json.dumps(manifest))
    return manifest


class PrecisionTests(unittest.TestCase):
    def test_bfloat16_conversion_matches_torch_at_rounding_and_subnormal_boundaries(self):
        values = torch.tensor(
            [0, -0.0, 1.0078125, -3.140625, 2**-14, 2**-24, 0.333984375], dtype=torch.bfloat16
        )
        raw = values.view(torch.uint8).numpy().tobytes()
        actual, dtype = convert_bytes(raw, "BF16")
        self.assertEqual(dtype, "F16")
        self.assertEqual(actual, values.half().view(torch.uint8).numpy().tobytes())

    def test_native_fp32_is_bitwise_preserved(self):
        values = np.array([-0.0, 1.0000001192092896, 3.1415927], dtype="<f4")
        result, dtype = convert_bytes(values.tobytes(), "F32")
        self.assertEqual(dtype, "F32")
        self.assertEqual(result, values.tobytes())

    def test_nonfinite_and_overflow_weights_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1e10):
            values = torch.tensor([value], dtype=torch.bfloat16)
            with self.subTest(value=value), self.assertRaises((ValueError, FloatingPointError)):
                convert_bytes(values.view(torch.uint8).numpy().tobytes(), "BF16")


class WeightPackTests(unittest.TestCase):
    def test_fp32_row_blocks_have_correct_byte_offsets_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            entry = source_tensor(root, "model.embed_tokens.weight", values, "F32")
            records = write_weight_pack(root, [entry], root, head_block_rows=2)
            write_native_index(root, records)
            manifest = complete_manifest(root, records)
            self.assertEqual(records[0]["row_blocks"][1]["offset"], 32)
            self.assertEqual(verify_weight_pack(root)["tensors"], 1)
            manifest["tensors"][0]["row_blocks"][1]["offset"] = 16
            (root / "deployment-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "row block"):
                verify_weight_pack(root)

    def test_chunks_alignment_shared_head_and_tail_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = torch.arange(30, dtype=torch.float32).reshape(5, 6).to(torch.bfloat16)
            precise = torch.tensor([1.0000001192092896, 3.1415927], dtype=torch.float32)
            entries = [
                source_tensor(root, "model.embed_tokens.weight", values, "BF16"),
                source_tensor(root, "model.norm.weight", precise, "F32"),
            ]
            records = write_weight_pack(root, entries, root, chunk_bytes=8, head_block_rows=2)
            write_native_index(root, records)
            complete_manifest(root, records)
            self.assertEqual(verify_weight_pack(root)["tensors"], 2)
            packed = (root / "weights.bin").read_bytes()
            self.assertEqual(records[1]["offset"] % 64, 0)
            self.assertEqual(records[0]["row_blocks"][-1]["rows"], 1)
            self.assertEqual(packed[:60], values.half().view(torch.uint8).numpy().tobytes())
            self.assertEqual(packed[64:], precise.view(torch.uint8).numpy().tobytes())
            self.assertEqual(records[0]["sdk_shape"], [6, 5])

    def test_same_size_corruption_and_manifest_range_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
            entry = source_tensor(root, "model.embed_tokens.weight", values, "BF16")
            records = write_weight_pack(root, [entry], root)
            write_native_index(root, records)
            manifest = complete_manifest(root, records)
            original = (root / "weights.bin").read_bytes()
            (root / "weights.bin").write_bytes(b"\0" + original[1:])
            # The first FP16 byte is already zero; change the high byte as well.
            (root / "weights.bin").write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_weight_pack(root)
            (root / "weights.bin").write_bytes(original)
            manifest["tensors"][0]["offset"] = 64
            (root / "deployment-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "offset"):
                verify_weight_pack(root)

    def test_truncated_source_never_produces_a_complete_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = source_tensor(
                root, "model.embed_tokens.weight", torch.ones(8).bfloat16(), "BF16"
            )
            path = root / entry["file"]
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "truncated"):
                write_weight_pack(root, [entry], root, chunk_bytes=4)
            self.assertFalse((root / "deployment-manifest.json").exists())


class MemoryTests(unittest.TestCase):
    def records(self):
        return [
            {"name": "model.embed_tokens.weight", "bytes": 248320 * 1024 * 2},
            {"name": "model.other", "bytes": 996231872},
        ]

    def test_unknown_costs_never_become_zero_or_a_fit_claim(self):
        budget = memory_budget(self.records(), pool_bytes=2008023040)
        self.assertEqual(budget["status"], "unverified")
        self.assertEqual(len(budget["unknown_components"]), 2)
        self.assertFalse(budget["actual_device_allocation_verified"])
        self.assertEqual(budget["known_bytes"]["compact_kv_cache"], 6 * 1024 * 1024)
        self.assertEqual(budget["known_bytes"]["recurrent_state_double_buffer"], 36 * 1024 * 1024)

    def test_second_layout_and_workspace_can_exceed_the_reported_pool(self):
        single = memory_budget(self.records(), pool_bytes=2008023040)
        duplicate = memory_budget(self.records(), pool_bytes=2008023040, duplicate_head=True)
        self.assertEqual(duplicate["status"], "exceeds_pool")
        self.assertEqual(
            duplicate["known_lower_bound_bytes"] - single["known_lower_bound_bytes"],
            self.records()[0]["bytes"],
        )
        estimated = memory_budget(
            self.records(),
            pool_bytes=2008023040,
            sdk_overhead_bytes=1024**3,
            workspace_bytes=64 * 1024**2,
        )
        self.assertEqual(estimated["status"], "exceeds_pool")


class SchemaTests(unittest.TestCase):
    def test_missing_renamed_or_wrong_precision_text_tensor_is_rejected(self):
        # Read only the small, public architecture configuration; no model load/download.
        config = {
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "vocab_size": 248320,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "tie_word_embeddings": True,
            "layer_types": [
                "full_attention" if i % 4 == 3 else "linear_attention" for i in range(24)
            ],
        }
        expected = expected_text_tensors(config)
        self.assertEqual(len(expected), 320)
        entries = [
            {"text_name": name, "shape": shape, "dtype": dtype, "group": "text"}
            for name, (shape, dtype) in expected.items()
        ]
        metadata = {"text_config": config, "tensors": entries}
        self.assertEqual(len(validate_text_entries(metadata)), 320)
        metadata["tensors"] = entries[:-1]
        with self.assertRaises(ValueError):
            validate_text_entries(metadata)
        metadata["tensors"] = entries
        precise = next(entry for entry in entries if entry["dtype"] == "F32")
        precise["dtype"] = "BF16"
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_text_entries(metadata)


if __name__ == "__main__":
    unittest.main()
