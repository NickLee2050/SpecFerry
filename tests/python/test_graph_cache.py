"""Numerical regressions for the compact reporting and graph-cache checks."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.opt.graph_validation import (
    evaluate,
)
from tests.python.validation_fixtures import GraphEvidenceFixture, write_generation_cache


class GraphBlockPrefillTests(GraphEvidenceFixture):
    def setUp(self):
        super().setUp()
        self.options = {
            "variant": "merged-kv",
            "decoder_group_size": 1,
            "prefill_head_policy": "every-token",
        }
        self.metadata = {
            "layers": 2,
            "heads": 2,
            "width": 2,
            "capacity": 16,
            "prefill_block": 4,
            "vocabulary": 5,
            "logits_dtype": "<f2",
            "logits_scope": "last_consumed_token",
            "requests": [
                {"tokens": [1, 2, 3], "consumed": 10, "prompt_length": 8},
                {"tokens": [3, 4], "consumed": 6, "prompt_length": 5},
                {"tokens": [1, 2, 3], "consumed": 10, "prompt_length": 8, "repeat_of": 0},
            ],
        }
        self.initialized = self.options | {
            "prefill_block": 4,
            "launches": 0,
            "prefill_steps": 0,
            "decode_steps": 0,
            "head_runs": 0,
        }
        self.native_run = self.options | {
            "graph_layout": "partitioned",
            "kv_reset_uploads_per_request": 4,
            "prefill_block": 4,
            "launches": 11,
            "prefill_steps": 5,
            "decode_steps": 6,
            "head_runs": 11,
            "cache_readback": "kv",
            "cache_first_layer": 1,
            "cache_last_layer": 2,
            "diagnostic_reads": 12,
            "diagnostic_read_bytes": 1536,
        }
        self.results = []
        for index, request in enumerate(self.metadata["requests"]):
            prefill_steps = 1 if index == 1 else 2
            launches = 3 if index == 1 else 4
            self.results.append(
                request
                | {
                    "prefill_block": 4,
                    "launches": launches,
                    "prefill_steps": prefill_steps,
                    "decode_steps": 2,
                    "head_runs": launches,
                    "uploads": launches + (4 if index else 0),
                    "reads": len(request["tokens"]),
                }
            )
            logits = np.full(5, -0.5, dtype="<f2")
            logits[request["tokens"][-1]] = 1
            logits.tofile(self.fixture / f"expected.{index}.logits.bin")
            logits.tofile(self.device / f"measured.{index}.logits.bin")
        write_generation_cache(
            self.fixture,
            self.device,
            self.metadata,
            lambda layer, prefix: (
                np.arange(4 * prefix, dtype="<f2").reshape(2, prefix, 2) / 32 + layer
            ),
        )
        self.save_metadata()

    def evaluate(self, **options):
        expected = self.options | {"cache_readback": "kv", "cache_layer_range": [1, 2]} | options
        if expected["cache_readback"] != "kv":
            expected["cache_layer_range"] = None
        return evaluate(self.fixture, self.device, self.metadata, 4, **expected)[0]

    def use_logits(self):
        (self.fixture / "model.txt").write_text("specferry-opt-model 1\n2 5 8 2 3 2 2 1\n")
        self.native_run.update(
            cache_readback="logits", diagnostic_reads=6, diagnostic_read_bytes=30
        )
        self.save_metadata()

    def test_request_matrix_only_compares_explicit_repeats(self):
        self.use_logits()
        self.metadata["requests"][2]["repeat_of"] = None
        logits = np.fromfile(self.device / "measured.2.logits.bin", dtype="<f2")
        logits[0] += np.float16(0.01)
        logits.tofile(self.device / "measured.2.logits.bin")
        checks = self.evaluate(cache_readback="logits")
        self.assertTrue(all(checks.values()), checks)
        self.assertNotIn("measured.2.logits.reset", checks)
        self.metadata["requests"][2]["repeat_of"] = 0
        self.assertFalse(self.evaluate(cache_readback="logits")["measured.2.logits.reset"])
        self.metadata["requests"][2]["repeat_of"] = 2
        with self.assertRaises(ValueError):
            self.evaluate(cache_readback="logits")

    def test_fresh_context_rebuild_requires_release_and_exact_logits(self):
        from specferry.models.opt.graph_validation import evaluate_rebuild

        lifecycle = dict(cycles=2, reused_context=False, recreated_context=True, released=True)
        (self.device / "lifecycle.json").write_text(json.dumps(lifecycle))
        native = dict(
            released=True,
            status="executed",
            closed_checks=1,
            closed_rejections=1,
            const_live_bytes=0,
            nonconst_live_bytes=0,
            const_peak_bytes=10,
            nonconst_peak_bytes=100,
        )
        for cycle in range(2):
            directory = self.device / f"cycle.{cycle}"
            directory.mkdir()
            (directory / "run.json").write_text(json.dumps(native))
            for index in range(len(self.results)):
                name = f"measured.{index}.logits.bin"
                (directory / name).write_bytes((self.device / name).read_bytes())
        self.assertTrue(all(evaluate_rebuild(self.device, self.metadata).values()))
        native["nonconst_live_bytes"] = 4
        (self.device / "cycle.1/run.json").write_text(json.dumps(native))
        self.assertFalse(evaluate_rebuild(self.device, self.metadata)["lifecycle.1.nonconst"])
        (self.device / "cycle.1/measured.0.logits.bin").write_bytes(b"wrong")
        self.assertFalse(
            evaluate_rebuild(self.device, self.metadata)["lifecycle.1.request.0.exact_logits"]
        )
        lifecycle["reused_context"] = True
        (self.device / "lifecycle.json").write_text(json.dumps(lifecycle))
        self.assertFalse(evaluate_rebuild(self.device, self.metadata)["lifecycle.completed"])

    def test_full_blocks_tail_decode_and_reset_require_independent_values(self):
        checks = self.evaluate()
        self.assertTrue(all(checks.values()), checks)
        self.assertTrue(checks["measured.2.layer.1.values.reset"])
        self.assertTrue(checks["measured.1.layer.0.keys.suffix"])

    def test_block_or_counters_cannot_be_omitted_or_forged_with_correct_outputs(self):
        documents = (
            ("initialization", self.initialized),
            ("run", self.native_run),
            ("request.0", self.results[0]),
        )
        for scope, document in documents:
            for name in ("prefill_block", "launches", "prefill_steps", "decode_steps", "head_runs"):
                original = document[name]
                for invalid in (None, original + 1, float(original), True):
                    with self.subTest(scope=scope, name=name, invalid=invalid):
                        document[name] = invalid
                        self.save_metadata()
                        self.assertFalse(self.evaluate()[f"{scope}.{name}"])
                del document[name]
                self.save_metadata()
                self.assertFalse(self.evaluate()[f"{scope}.{name}"])
                document[name] = original
        self.save_metadata()

    def test_reference_block_and_token_consumption_must_match(self):
        for invalid in (None, 1, 4.0, True):
            with self.subTest(block=invalid):
                self.metadata["prefill_block"] = invalid
                self.assertFalse(self.evaluate()["reference.prefill_block"])
        self.metadata["prefill_block"] = 4
        self.metadata["requests"][0]["prompt_length"] = 7
        self.assertFalse(self.evaluate()["reference.0.consumed"])

    def test_final_logits_are_required_even_when_greedy_tokens_match(self):
        self.use_logits()
        path = self.device / "measured.0.logits.bin"
        original = path.read_bytes()
        for damage in ("missing", "short", "numeric", "nonfinite"):
            with self.subTest(damage=damage):
                path.write_bytes(original)
                if damage == "missing":
                    path.unlink()
                elif damage == "short":
                    path.write_bytes(original[:-2])
                else:
                    values = np.frombuffer(original, dtype="<f2").copy()
                    # Alter a losing logit: the selected token remains unchanged.
                    values[0] = 0.5 if damage == "numeric" else np.nan
                    values.tofile(path)
                checks = self.evaluate(cache_readback="logits")
                self.assertTrue(checks["request.0"])
                self.assertFalse(checks["measured.0.logits"])
        path.write_bytes(original)
        (self.fixture / "expected.0.logits.bin").unlink()
        self.assertFalse(self.evaluate(cache_readback="logits")["measured.0.logits"])

    def test_token_only_requires_zero_diagnostic_io_and_still_rejects_wrong_tokens(self):
        for directory in (self.fixture, self.device):
            for path in directory.glob("*.bin"):
                path.unlink()
        self.native_run.update(cache_readback="none", diagnostic_reads=0, diagnostic_read_bytes=0)
        for result in self.results:
            result["read_bytes"] = len(result["tokens"]) * 4
        self.save_metadata()
        checks = self.evaluate(cache_readback="none")
        self.assertTrue(all(checks.values()), checks)
        self.assertFalse(any(name.startswith("measured.") for name in checks))
        for field, value in (
            ("cache_readback", "prefix"),
            ("diagnostic_reads", 1),
            ("diagnostic_read_bytes", 2),
            ("diagnostic_reads", None),
        ):
            with self.subTest(field=field):
                original = self.native_run[field]
                self.native_run[field] = value
                self.save_metadata()
                self.assertFalse(self.evaluate(cache_readback="none")[field])
                self.native_run[field] = original
        self.results[0]["tokens"] = [4, 2, 3]
        self.save_metadata()
        self.assertFalse(self.evaluate(cache_readback="none")["request.0"])

    def test_token_only_records_cannot_be_accepted_as_diagnostic_readback(self):
        self.native_run.update(cache_readback="none", diagnostic_reads=0, diagnostic_read_bytes=0)
        self.save_metadata()
        # Stale diagnostic files cannot widen the recorded scope.
        self.assertFalse(self.evaluate()["cache_readback"])

    def test_selected_kv_checks_values_and_scope_without_other_diagnostics(self):
        self.native_run.update(
            cache_readback="kv",
            cache_first_layer=2,
            cache_last_layer=2,
            diagnostic_reads=6,
            diagnostic_read_bytes=768,
        )
        for path in self.device.glob("*.bin"):
            if ".layer.1." not in path.name:
                path.unlink()
        self.save_metadata()
        options = {"cache_readback": "kv", "cache_layer_range": [2, 2]}
        checks = self.evaluate(**options)
        self.assertTrue(all(checks.values()), checks)
        self.assertTrue(checks["measured.2.layer.1.keys.reset"])
        self.assertNotIn("measured.0.logits", checks)
        for field in ("diagnostic_reads", "diagnostic_read_bytes", "cache_first_layer"):
            original = self.native_run[field]
            self.native_run[field] += 1
            self.save_metadata()
            self.assertFalse(self.evaluate(**options)[field])
            self.native_run[field] = original
        self.save_metadata()
        path = self.device / "measured.1.layer.1.values.bin"
        values = np.fromfile(path, dtype="<f2")
        values[0] += 2
        values.tofile(path)
        self.assertFalse(self.evaluate(**options)["measured.1.layer.1.values"])
        for bounds in (None, [0, 1], [2, 1], [1, 3], [True, 1]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                self.evaluate(cache_readback="kv", cache_layer_range=bounds)

    def test_logits_only_requires_complete_values_and_bit_exact_repeat(self):
        (self.fixture / "model.txt").write_text("specferry-opt-model 1\n2 5 8 2 3 2 2 1\n")
        self.use_logits()
        for path in self.device.glob("*.layer.*.bin"):
            path.unlink()
        self.save_metadata()
        checks = self.evaluate(cache_readback="logits")
        self.assertTrue(all(checks.values()), checks)
        self.assertTrue(checks["measured.2.logits.reset"])
        self.native_run["diagnostic_reads"] = 3
        self.save_metadata()
        self.assertFalse(self.evaluate(cache_readback="logits")["diagnostic_reads"])
        self.native_run["diagnostic_reads"] = 6
        self.save_metadata()
        path = self.device / "measured.2.logits.bin"
        values = np.fromfile(path, dtype="<f2")
        values[0] += 0.001  # Within numeric tolerance but fails the bit-exact repeated A gate.
        values.tofile(path)
        checks = self.evaluate(cache_readback="logits")
        self.assertTrue(checks["measured.2.logits"])
        self.assertFalse(checks["measured.2.logits.reset"])
        values[0] += 1
        values.tofile(path)
        self.assertFalse(self.evaluate(cache_readback="logits")["measured.2.logits"])

    def test_cache_layout_and_unwritten_slots_are_independently_checked(self):
        path = self.device / "measured.0.layer.0.keys.bin"
        original = path.read_bytes()
        for damage in ("prefix", "suffix"):
            with self.subTest(damage=damage):
                values = np.frombuffer(original, dtype="<f2").reshape(2, 16, 2).copy()
                if damage == "prefix":
                    values[:, 0] = values[::-1, 0]
                else:
                    values[0, 10, 0] = 1
                values.tofile(path)
                key = "measured.0.layer.0.keys" + (".suffix" if damage == "suffix" else "")
                self.assertFalse(self.evaluate()[key])
        path.write_bytes(original)

    def test_unimplemented_partitioned_combinations_are_rejected(self):
        for options in (
            {"variant": "baseline"},
            {"variant": "compiled-ffn-finish"},
            {"variant": "merged-kv-local-const"},
            {"decoder_group_size": 2},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.evaluate(**options)

    def test_last_token_policy_requires_actual_head_skips_with_blocks_and_tails(self):
        self.initialized["prefill_head_policy"] = "last-token"
        self.native_run.update(prefill_head_policy="last-token", head_runs=8)
        for result in self.results:
            result["head_runs"] = len(result["tokens"])
        self.save_metadata()
        checks = self.evaluate(prefill_head_policy="last-token")
        self.assertTrue(all(checks.values()), checks)
        self.results[1]["head_runs"] += 1
        self.save_metadata()
        self.assertFalse(self.evaluate(prefill_head_policy="last-token")["request.1.head_runs"])
