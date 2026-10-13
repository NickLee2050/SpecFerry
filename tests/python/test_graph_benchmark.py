"""Numerical regressions for the compact reporting and graph-cache checks."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


from tests.python.validation_fixtures import GraphEvidenceFixture


class GraphBenchmarkTests(GraphEvidenceFixture):
    def setUp(self):
        super().setUp()
        self.options = {
            "variant": "merged-kv",
            "decoder_group_size": 1,
            "prefill_head_policy": "every-token",
        }
        self.metadata = {
            "layers": 2,
            "capacity": 16,
            "prefill_block": 4,
            "requests": [{"prompt_length": 5, "tokens": [1, 2, 3], "consumed": 7}],
        }
        counts = dict(launches=4, prefill_steps=1, decode_steps=3, head_runs=4)
        self.initialized = self.options | {"prefill_block": 4} | dict.fromkeys(counts, 0)
        self.native_run = (
            self.options
            | {key: value * 3 for key, value in counts.items()}
            | {
                "status": "executed",
                "released": True,
                "graph_layout": "partitioned",
                "prefill_block": 4,
                "kv_reset_uploads_per_request": 4,
                "closed_checks": 1,
                "closed_rejections": 1,
                "capacity_checks": 0,
                "capacity_rejections": 0,
                "cache_readback": "none",
                "diagnostic_reads": 0,
                "diagnostic_read_bytes": 0,
                "const_live_bytes": 0,
                "const_peak_bytes": 12,
                "nonconst_live_bytes": 0,
                "nonconst_peak_bytes": 24,
            }
        )
        self.manifest = {
            "schema_version": 1,
            "completed": True,
            "request_count": 1,
            "warmups": 1,
            "repeats": 2,
            "runs": [],
        }
        for ordinal, (kind, iteration) in enumerate(
            (("warmup", 0), ("measured", 0), ("measured", 1))
        ):
            name = f"{kind}.0.{iteration}"
            self.manifest["runs"].append(
                {
                    "name": name,
                    "request_index": 0,
                    "iteration": iteration,
                    "prompt_length": 5,
                    "kind": kind,
                    "result_file": f"{name}.json",
                    "tokens_match": True,
                }
            )
            result = (
                counts
                | self.metadata["requests"][0]
                | {
                    "prefill_block": 4,
                    "uploads": 4 + (4 if ordinal else 0),
                    "reads": 3,
                    "read_bytes": 12,
                }
            )
            (self.device / f"{name}.json").write_text(json.dumps(result))
        self.save_metadata()
        (self.device / "benchmark.json").write_text(json.dumps(self.manifest))

    def evaluate(self):
        from specferry.models.opt.graph_validation import evaluate_benchmark

        return evaluate_benchmark(
            self.device, self.metadata, 4, warmups=1, repeats=2, **self.options
        )

    def test_all_repetitions_are_checked_and_only_measured_runs_are_returned(self):
        checks, runs = self.evaluate()
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual([run["iteration"] for run in runs], [0, 1])
        self.assertTrue(checks["warmup.0.0"])
        self.assertTrue(checks["run.decode_steps"])

    def test_wrong_warmup_tokens_or_missing_measured_output_fail(self):
        path = self.device / "warmup.0.0.json"
        result = json.loads(path.read_text())
        result["tokens"] = [2, 1, 3]
        path.write_text(json.dumps(result))
        self.assertFalse(self.evaluate()[0]["warmup.0.0"])
        (self.device / "measured.0.1.json").unlink()
        self.assertFalse(self.evaluate()[0]["measured.0.1"])

    def test_incomplete_manifest_diagnostic_reads_and_live_memory_fail(self):
        self.manifest["runs"].pop()
        (self.device / "benchmark.json").write_text(json.dumps(self.manifest))
        self.native_run.update(diagnostic_reads=1, nonconst_live_bytes=8)
        self.save_metadata()
        checks, _ = self.evaluate()
        for field in ("benchmark.records", "diagnostic_reads", "memory.nonconst"):
            self.assertFalse(checks[field])
