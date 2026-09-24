"""Guard complete-model acceptance against missing data and host fallback."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.opt.generation import resolve_selection, validate_tokens
from specferry.models.opt.validation_generation import compare_generation, evaluate


class GenerationContractTests(unittest.TestCase):
    def test_checkpoint_defaults_and_explicit_sampling(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root)
            path = checkpoint / "generation_config.json"
            # OPT's published file omits decoding flags; Transformers supplies them.
            path.write_text(json.dumps({"bos_token_id": 2, "eos_token_id": 2, "pad_token_id": 1}))
            selection = resolve_selection(checkpoint)
            self.assertEqual(selection["origin"], "model_default")
            self.assertEqual(selection["mode"], "greedy")
            self.assertFalse(selection["resolved_config"]["do_sample"])
            self.assertEqual(selection["resolved_config"]["num_beams"], 1)
            self.assertIsNone(selection["top_k"])  # Inactive while greedy.
            sampled = resolve_selection(checkpoint, sample=True, seed=42)
            self.assertEqual(sampled["origin"], "explicit_sample")
            self.assertEqual(sampled["mode"], "multinomial")
            self.assertEqual(sampled["top_k"], 0)
            self.assertEqual(sampled["seed"], 42)
            path.write_text('{"do_sample":true,"top_k":0}')
            changed = resolve_selection(checkpoint, seed=42)
            self.assertEqual(changed["origin"], "model_default")
            self.assertEqual(changed["mode"], "multinomial")
            self.assertNotEqual(changed["source_sha256"], selection["source_sha256"])

    def test_unsupported_defaults_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root)
            path = checkpoint / "generation_config.json"
            for config in (
                {"num_beams": 2},
                {"do_sample": True},  # Inherits top_k=50, currently unsupported.
                {"repetition_penalty": 1.2},
                {"forced_eos_token_id": 2},
                {"stop_strings": ["END"]},
                {"min_new_tokens": 8},
            ):
                path.write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    resolve_selection(checkpoint)
            path.write_text("{}")
            for seed in (-1, 2**32, True):
                with self.assertRaises(ValueError):
                    resolve_selection(checkpoint, sample=True, seed=seed)
            with self.assertRaises(ValueError):
                resolve_selection(checkpoint, seed=42)

    def test_cpu_comparison_uses_checkpoint_policy_without_forcing_decoding(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root)
            (checkpoint / "generation_config.json").write_text(
                '{"eos_token_id":2,"pad_token_id":1}'
            )
            model = SimpleNamespace(generate=Mock(return_value=torch.tensor([[2, 5]])))
            with patch(
                "specferry.models.opt.validation_generation.load_model", return_value=(model, {})
            ):
                result = compare_generation(
                    checkpoint, {"tokens": [2], "capacity": 8}, {"tokens": [5]}, 1
                )
            self.assertTrue(result["passed"])
            self.assertEqual(model.generation_config.eos_token_id, 2)
            arguments = model.generate.call_args.kwargs
            self.assertNotIn("do_sample", arguments)
            self.assertNotIn("num_beams", arguments)
            self.assertEqual(arguments["max_new_tokens"], 1)

    def test_token_bounds_and_capacity(self):
        self.assertEqual(validate_tokens([0, 50271], 50272, 2), [0, 50271])
        self.assertEqual(validate_tokens([], 50272, 512), [])
        for invalid in ([-1], [50272], [True], [1.0], [0, 1, 2]):
            with self.assertRaises(ValueError):
                validate_tokens(invalid, 50272, 2)

    def test_full_model_acceptance_requires_device_selection_and_exact_repeats(self):
        with tempfile.TemporaryDirectory() as root:
            fixture, actual = Path(root) / "fixture", Path(root) / "actual"
            fixture.mkdir()
            actual.mkdir()
            array = np.array([0.5, 1.0], dtype="<f2")
            spec = {"shape": [2], "dtype": "<f2", "bytes": array.nbytes}
            metadata = {
                "mode": "teacher",
                "count": 1,
                "layers": 24,
                "observations": {"teacher.0.logits": spec},
                "tokens": [1],
                "tolerances": {"atol": 0.08, "rtol": 0.02},
            }
            (fixture / "teacher.0.logits.bin").write_bytes(array.tobytes())
            for sequence in ("teacher", "reset", "final", "fresh"):
                (actual / f"{sequence}.0.logits.bin").write_bytes(array.tobytes())
                (actual / f"{sequence}.tokens.txt").write_text("1\n")
            evidence = {
                "returncode": 0,
                "process_group_exited": True,
                "device_recovery_required": False,
            }
            execution = {
                "status": "executed",
                "phase": "complete",
                "released": True,
                "bounds_rejected": True,
                "reset_rejected_stale": True,
                "steps": 4,
                "cache_writes": 96,
                "uploads": 104,
                "upload_bytes": 416,
                "reads": 4,
                "read_bytes": 16,
            }

            def result(change=None, device=None):
                (actual / "execution.json").write_text(json.dumps(execution | (change or {})))
                return evaluate(fixture, actual, metadata, evidence | (device or {}))

            self.assertEqual(result()["status"], "numerical_pass")
            self.assertFalse(result()["hardware_execution_proven"])
            for change in (
                {"released": False},
                {"phase": "initialize"},
                {"bounds_rejected": False},
                {"reset_rejected_stale": False},
            ):
                self.assertEqual(result(change)["status"], "failed")
            self.assertEqual(result(device={"returncode": -11})["status"], "failed")
            self.assertEqual(result(device={"device_recovery_required": True})["status"], "failed")
            (actual / "fresh.tokens.txt").write_text("0\n")
            self.assertFalse(result()["checks"]["fresh.tokens"]["passed"])
            (actual / "fresh.tokens.txt").write_text("1\n")
            # A numerically close reset still fails exact device-repeat acceptance.
            (actual / "reset.0.logits.bin").write_bytes((array + 0.001).astype("<f2").tobytes())
            self.assertFalse(result()["checks"]["resetteacher.0.logits"]["passed"])
            (actual / "teacher.0.logits.bin").write_bytes(b"")
            self.assertFalse(result()["checks"]["teacher.0.logits"]["passed"])


if __name__ == "__main__":
    unittest.main()
