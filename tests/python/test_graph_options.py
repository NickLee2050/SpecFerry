"""Numerical regressions for the compact reporting and graph-cache checks."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.opt.graph_validation import (
    evaluate,
)
from tests.python.validation_fixtures import GraphEvidenceFixture


class GraphExecutionOptionsTests(GraphEvidenceFixture):
    def setUp(self):
        super().setUp()
        self.metadata = {
            "layers": 2,
            "heads": 1,
            "width": 2,
            "capacity": 4,
            "requests": [
                {"tokens": [1, 2], "consumed": 4, "prompt_length": 3},
                {"tokens": [3, 4], "consumed": 2, "prompt_length": 1},
                {"tokens": [1, 2], "consumed": 4, "prompt_length": 3},
            ],
        }
        self.options = {
            "variant": "merged-kv",
            "decoder_group_size": 1,
            "prefill_head_policy": "last-token",
            "cache_readback": "none",
        }
        self.initialized = self.options.copy()
        self.native_run = self.options | {
            "graph_layout": "partitioned",
            "kv_reset_uploads_per_request": 4,
            "head_runs": 6,
            "diagnostic_reads": 0,
            "diagnostic_read_bytes": 0,
        }
        self.save_metadata()
        for index, request in enumerate(self.metadata["requests"]):
            consumed = request["consumed"]
            result = request | {
                "launches": consumed,
                "uploads": consumed + (4 if index else 0),
                "reads": len(request["tokens"]),
                "read_bytes": 4 * len(request["tokens"]),
            }
            (self.device / f"measured.{index}.json").write_text(json.dumps(result))

    def evaluate(self, **options):
        return evaluate(self.fixture, self.device, self.metadata, 1, **options)[0]

    def test_head_counts_distinguish_full_prompt_heads_from_last_prompt_head(self):
        for policy, expected_heads in (("every-token", 10), ("last-token", 6)):
            with self.subTest(policy=policy):
                self.initialized["prefill_head_policy"] = policy
                self.native_run.update(prefill_head_policy=policy, head_runs=expected_heads)
                self.save_metadata()
                checks = self.evaluate(**(self.options | {"prefill_head_policy": policy}))
                self.assertTrue(all(checks.values()), checks)
                self.assertTrue(checks["head_runs"])

    def test_ignored_options_fail_even_when_all_tokens_and_cache_match(self):
        for scope, document in (("run", self.native_run), ("initialization", self.initialized)):
            for field, ignored in (
                ("decoder_group_size", 2),
                ("decoder_group_size", 1.0),
                ("prefill_head_policy", "every-token"),
                ("prefill_head_policy", None),
                ("variant", "merged-kv-local-const"),
            ):
                with self.subTest(scope=scope, field=field, ignored=ignored):
                    original = document[field]
                    document[field] = ignored
                    self.save_metadata()
                    checks = self.evaluate(**self.options)
                    self.assertFalse(checks[f"{scope}.{field}"])
                    self.assertTrue(all(checks[f"request.{index}"] for index in range(3)))
                    self.assertTrue(all(v for k, v in checks.items() if k.startswith("measured.")))
                    document[field] = original
        self.save_metadata()

    def test_missing_initialization_or_wrong_head_count_cannot_accept_optimization(self):
        (self.device / "initialization.json").unlink()
        self.assertFalse(self.evaluate(**self.options)["initialization.prefill_head_policy"])
        for count in (None, 0, 5, 10, 6.0, "6", True):
            with self.subTest(count=count):
                self.native_run["head_runs"] = count
                self.save_metadata()
                self.assertFalse(self.evaluate(**self.options)["head_runs"])

    def test_invalid_expected_execution_options_are_rejected(self):
        for options in (
            {"decoder_group_size": True},
            {"decoder_group_size": 3},
            {"prefill_head_policy": "unknown"},
            {"decoder_group_size": 2, "variant": "local-const"},
            {"readback_transport": "unknown"},
            {"readback_transport": "mapped"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.evaluate(**options)

    def test_readback_transport_must_match_both_initialization_and_execution(self):
        options = self.options | {
            "cache_readback": "kv",
            "cache_layer_range": [1, 2],
            "readback_transport": "mapped",
        }
        self.initialized["readback_transport"] = "mapped"
        self.native_run.update(
            readback_transport="mapped",
            cache_readback="kv",
            cache_first_layer=1,
            cache_last_layer=2,
            diagnostic_reads=12,
            diagnostic_read_bytes=192,
        )
        for index, request in enumerate(self.metadata["requests"]):
            for layer in range(2):
                for name in ("keys", "values"):
                    np.zeros(request["consumed"] * 2, dtype="<f2").tofile(
                        self.fixture / f"expected.{index}.{layer}.{name}.bin"
                    )
                    np.zeros(8, dtype="<f2").tofile(
                        self.device / f"measured.{index}.layer.{layer}.{name}.bin"
                    )
        self.save_metadata()
        checks = self.evaluate(**options)
        self.assertTrue(all(checks.values()), checks)
        self.native_run["readback_transport"] = "sdk"
        self.save_metadata()
        checks = self.evaluate(**options)
        self.assertFalse(checks["run.readback_transport"])
        self.assertTrue(checks["initialization.readback_transport"])
