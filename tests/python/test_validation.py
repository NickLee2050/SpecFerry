"""Numerical regressions for the compact reporting and graph-cache checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.data.checkpoint import sha256
from specferry.models.opt.config import Components
from specferry.models.opt.graph_validation import evaluate, validate_prefix_fixture
from specferry.models.opt.metrics import summarize
from specferry.validation.allocation import MIB, capacity_result, check_segment_budget
from specferry.validation.arrays import compare_arrays


class PrefixFixtureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config = {
            "hidden_size": 4,
            "ffn_dim": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "word_embed_proj_dim": 2,
            "vocab_size": 5,
            "max_position_embeddings": 8,
            "bos_token_id": 2,
            "eos_token_id": 2,
            "pad_token_id": 1,
        }
        (self.root / "deployment-manifest.json").write_text(json.dumps({"config": config}))
        Components(4, 8, 2, 1, 4).write(self.root / "components.txt")
        (self.root / "model.txt").write_text("specferry-opt-model 1\n2 5 8 2 4096 2 2 1\n")
        (self.root / "tokens.txt").write_text("2 3\n")
        observations = {}
        for step in range(2):
            shapes = {
                "lookup": (1, 1, 2),
                "input_projection": (1, 1, 4),
                "position_embedding": (1, 1, 4),
                "embedding": (1, 1, 4),
                "projected": (1, 1, 2),
                "logits": (1, 1, 5),
                "layer.0": (1, 1, 4),
                "layer.0.keys": (2, step + 1, 2),
                "layer.0.values": (2, step + 1, 2),
            }
            for name, shape in shapes.items():
                key = f"teacher.{step}.{name}"
                values = np.ones(shape, dtype="<f2")
                values.tofile(self.root / f"{key}.bin")
                observations[key] = {"shape": list(shape), "dtype": "<f2", "bytes": values.nbytes}
        self.metadata = {
            "mode": "prefix",
            "layers": 1,
            "count": 2,
            "capacity": 4,
            "tokens": [0, 0],
            "observations": observations,
            "manifest_sha256": sha256(self.root / "deployment-manifest.json"),
            "reference_checks": {"cached_logits": {"passed": True}},
            "fixture_sha256": {path.name: sha256(path) for path in self.root.iterdir()},
        }
        self.save()

    def save(self):
        (self.root / "reference.json").write_text(json.dumps(self.metadata))

    def validate(self, layers=1):
        return validate_prefix_fixture(self.root, self.root, layers)

    def test_truncated_reference_accepts_only_matching_depth_and_valid_prefix_shapes(self):
        self.assertEqual(self.validate()["layers"], 1)
        with self.assertRaisesRegex(ValueError, "layer count"):
            self.validate(layers=2)
        key = "teacher.1.layer.0.keys"
        self.metadata["observations"][key]["shape"] = [2, 4, 2]
        self.save()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.validate()

    def test_stale_files_and_full_model_native_configuration_are_rejected(self):
        path = self.root / "teacher.0.lookup.bin"
        path.write_bytes(bytes(path.stat().st_size))
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.validate()
        self.metadata["fixture_sha256"][path.name] = sha256(path)
        Components(4, 8, 2, 2, 4).write(self.root / "components.txt")
        self.metadata["fixture_sha256"]["components.txt"] = sha256(self.root / "components.txt")
        self.save()
        with self.assertRaisesRegex(ValueError, "native configuration"):
            self.validate()

    def test_unchecked_cpu_reference_and_wrong_expected_token_are_rejected(self):
        self.metadata["reference_checks"]["cached_logits"]["passed"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "consistency"):
            self.validate()
        self.metadata["reference_checks"]["cached_logits"]["passed"] = True
        self.metadata["tokens"][1] = 4
        self.save()
        with self.assertRaisesRegex(ValueError, "logits"):
            self.validate()


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
