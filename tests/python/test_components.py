"""Model-independent payloads and explicit component configuration boundaries."""

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.data.checkpoint import inventory
from specferry.export.memory import memory_budget
from specferry.export.weights import (
    convert_bytes,
    verify_weight_pack,
    write_native_index,
    write_weight_pack,
)
from specferry.models.qwen3_5.components import BASELINE, Components
from tests.python.weight_fixtures import complete_manifest, source_tensor


class GenericCheckpointTests(unittest.TestCase):
    def test_arbitrary_identity_names_and_payload_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = source_tensor(root, "encoder.projection", torch.arange(4).half(), "F16")
            name, filename = entry["source_name"], entry["file"]
            index = {"metadata": {"total_size": 8}, "weight_map": {name: filename}}
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            (root / "config.json").write_text('{"model_type": "test_fixture"}')
            manifest = {
                "status": "complete",
                "repo_id": "fixture/tiny",
                "resolved_revision": "fixed",
                "files": [{"path": filename, "size": (root / filename).stat().st_size}],
            }
            (root / "download-manifest.json").write_text(json.dumps(manifest))
            result = inventory(root, repo_id="fixture/tiny", revision="fixed")
            self.assertEqual(result["tensors"][0]["source_name"], name)
            self.assertNotIn("group", result["tensors"][0])
            with self.assertRaises(ValueError):
                inventory(root, repo_id="fixture/tiny", revision="different")
            original = (root / filename).read_bytes()
            header_size = struct.unpack("<Q", original[:8])[0]
            header = json.loads(original[8 : 8 + header_size])
            header[name]["data_offsets"] = [1, 9]
            changed = json.dumps(header).encode()
            (root / filename).write_bytes(
                struct.pack("<Q", len(changed)) + changed + original[8 + header_size :]
            )
            with self.assertRaisesRegex(ValueError, "offsets"):
                inventory(root, repo_id="fixture/tiny", revision="fixed")


class GenericWeightTests(unittest.TestCase):
    def test_fp16_bytes_preserve_sign_subnormals_and_rounding(self):
        raw = np.array([0, -0.0, 2**-24, -(2**-14), 1.0009765625, 65504], dtype="<f2").tobytes()
        self.assertEqual(convert_bytes(raw, "F16", "F16"), (raw, "F16"))
        for source, target in (("F32", "F16"), ("F16", "F32"), ("BF16", "F32")):
            with self.assertRaises(ValueError):
                convert_bytes(raw, source, target)

    def test_pack_without_embedding_and_direct_alias_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = source_tensor(
                root, "encoder.projection", torch.arange(6).half().reshape(2, 3), "F16"
            )
            records = write_weight_pack(root, [entry], root, chunk_bytes=4)
            write_native_index(root, records)
            manifest = complete_manifest(root, records)
            manifest["aliases"] = {"output": "encoder.projection"}
            path = root / "deployment-manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(verify_weight_pack(root)["tensors"], 1)
            self.assertFalse(records[0]["transformed"])
            self.assertEqual(
                hashlib.sha256((root / "weights.bin").read_bytes()).hexdigest(),
                records[0]["sha256"],
            )
            for aliases in (
                {"output": "missing"},
                {"encoder.projection": "encoder.projection"},
                {"first": "second", "second": "encoder.projection"},
                {"first": "second", "second": "first"},
                {"output": []},
            ):
                manifest["aliases"] = aliases
                path.write_text(json.dumps(manifest))
                with self.subTest(aliases=aliases), self.assertRaisesRegex(ValueError, "aliases"):
                    verify_weight_pack(root)
            manifest["aliases"] = {}
            index = root / "weights.index"
            index.write_text(index.read_text().replace("encoder.projection", "different.weight"))
            manifest["files"]["weights.index"] = {
                "bytes": index.stat().st_size,
                "sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            }
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "index disagrees"):
                verify_weight_pack(root)

    def test_budget_accepts_explicit_resources_without_model_names(self):
        budget = memory_budget([{"name": "matrix", "bytes": 64}], {"cache": 128}, capacity=8)
        self.assertEqual(budget["known_lower_bound_bytes"], 192)
        self.assertEqual(len(budget["unknown_components"]), 2)
        self.assertFalse(budget["actual_device_allocation_verified"])
        with self.assertRaises(ValueError):
            memory_budget([{"bytes": -1}], {}, capacity=8)


class ComponentConfigurationTests(unittest.TestCase):
    def test_dimensions_selection_and_accounting_change_together(self):
        small = replace(
            BASELINE,
            hidden=64,
            intermediate=96,
            query_heads=4,
            kv_heads=2,
            head_dim=16,
            capacity=8,
            rotary_dim=8,
            delta_heads=2,
            key_dim=8,
            value_dim=8,
            convolution_width=3,
            layer_types=("delta", "attention"),
        )
        self.assertEqual(small.recurrent_shape, (1, 2, 8, 8))
        self.assertEqual(small.convolution_shape, (1, 48, 3))
        self.assertEqual(small.state_bytes([0, 1]), 2624)
        self.assertEqual(small.transfers([0])["upload_bytes"], 128)
        self.assertEqual(small.transfers([1])["upload_bytes"], 136)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(Components(**small.write(root)), small)
            self.assertEqual(
                (root / "components.txt").read_text().splitlines()[-1], "delta attention"
            )

    def test_invalid_components_rejected_without_sdk(self):
        for change in (
            {"query_heads": 3},
            {"capacity": 513},
            {"rotary_dim": 63},
            {"rotary_dim": 512},
            {"delta_heads": 2**32 - 1},
            {"epsilon": float("nan")},
            {"convolution_width": 1},
            {"hidden": -1},
            {"layer_types": ("unknown",)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(BASELINE, **change)


if __name__ == "__main__":
    unittest.main()
