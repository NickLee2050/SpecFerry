"""Performance summaries must preserve phase boundaries and nonoverlapping spans."""

import csv
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.validation.performance import (
    SUMMARY_COLUMNS,
    read_summary,
    summarize_request,
    summarize_runs,
)


def row(phase, api, seconds, *, count=1, nested=0, role="decoder"):
    return {
        "request": "measured.0",
        "phase": phase,
        "role": role,
        "api": api,
        "count": count,
        "total_seconds": seconds,
        "min_seconds": seconds / count,
        "max_seconds": seconds / count,
        "completed": count,
        "exceptions": 0,
        "nested_count": nested,
    }


def generation(*, consumed=8, total=13, tokens=None):
    return {
        "tokens": [21, 22, 23, 24] if tokens is None else tokens,
        "consumed": consumed,
        "prefill_seconds": 4.5,
        "first_token_seconds": 4.6,
        "total_seconds": total,
    }


class PerformanceSummaryTests(unittest.TestCase):
    def test_zero_generation_has_no_graph_metrics_but_partial_prompt_is_rejected(self):
        script = Path(__file__).resolve().parents[2] / "scripts/generate_opt.py"
        summarize_graph_metrics = runpy.run_path(str(script))["summarize_graph_metrics"]
        with tempfile.TemporaryDirectory() as directory:
            device = Path(directory)
            rows = [
                row("initialize", "vsi_nn_CreateContext", 0.1) | {"request": ""},
                row("release", "vsi_nn_ReleaseContext", 0.1) | {"request": ""},
            ]
            with (device / "sdk-summary.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            empty = {
                "tokens": [],
                "consumed": 0,
                "prefill_seconds": 0,
                "first_token_seconds": 0,
                "total_seconds": 0,
            }
            request = {"tokens": [2] * 5}
            self.assertEqual(summarize_graph_metrics(device, [empty], request), {})
            with self.assertRaisesRegex(ValueError, "smaller"):
                summarize_graph_metrics(device, [empty | {"consumed": 3}], request)

    def test_tsv_contract_rejects_nonfinite_counts_and_duplicate_aggregates(self):
        base = row("prefill", "vsi_nn_RunGraph", 4)
        for records in (
            [base | {"total_seconds": "nan"}],
            [base | {"count": 0}],
            [base | {"nested_count": 2}],
            [base | {"exceptions": 1}],
            [base, base],
        ):
            with self.subTest(records=records), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sdk-summary.tsv"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(records)
                with self.assertRaises(ValueError):
                    read_summary(path)

    def test_phases_separate_reset_verify_io_and_use_consumed_decode_denominator(self):
        rows = [
            row("prefill", "vsi_nn_RunGraph", 4, count=2, role="embedding"),
            row("decode", "vxProcessGraph", 6, count=3),
            row("prefill", "vsi_nn_CopyDataToTensor", 0.2),
            row("decode", "vsi_nn_ConvertTensorToData", 0.3),
            row("decode", "vsi_nn_VerifyGraph", 1),
            row("decode", "vsi_nn_SetupGraph", 0.5),
            row("reset", "vsi_nn_CopyDataToTensor", 1000),
        ]
        result = summarize_request(rows, "measured.0", generation(), 5, [21, 22, 23, 24])
        self.assertEqual(result["decode_steps"], 3)
        self.assertEqual(result["graph_decode_ms_per_token"], 2000)
        self.assertEqual(result["graph_decode_tokens_per_second"], 0.5)
        self.assertEqual(result["graph_prefill_ms_per_input"], 800)
        self.assertEqual(result["reset_seconds"], 1000)
        self.assertEqual(result["accounted_generation_sdk_seconds"], 12)
        self.assertEqual(result["unattributed_generation_seconds"], 1)
        self.assertEqual(result["runtime_verify_seconds_by_phase"], {"decode": 1})
        self.assertEqual(result["runtime_setup_seconds_by_phase"], {"decode": 0.5})
        self.assertEqual(result["graph_counts_by_phase"], {"prefill": 2, "decode": 3})
        self.assertEqual(result["role_hotspots"][0]["api"], "vxProcessGraph")
        self.assertTrue(result["tokens_match_expected"])
        self.assertEqual(len(result["attribution_risks"]), 2)
        self.assertTrue(result["runtime_compilation_observed"])

    def test_openvx_runtime_verify_and_setup_are_separate_from_graph_execution(self):
        rows = [
            row("prefill", "vsi_nn_RunGraph", 6),
            row("prefill", "vxVerifyGraph", 2),
            row("prefill", "vsi_nn_VerifyGraph", 0.5),
            row("prefill", "vsi_nn_SetupGraph", 1),
        ]
        result = summarize_request(rows, "measured.0", generation(consumed=5, total=11), 5)
        self.assertEqual(result["graph_total_seconds"], 6)
        self.assertEqual(result["runtime_verify_seconds_by_phase"], {"prefill": 2.5})
        self.assertEqual(result["runtime_setup_seconds_by_phase"], {"prefill": 1})
        self.assertEqual(result["io_seconds_by_phase"], {})
        self.assertEqual(result["other_sdk_seconds_by_phase"], {})
        self.assertEqual(result["accounted_generation_sdk_seconds"], 9.5)
        self.assertEqual(result["unattributed_generation_seconds"], 1.5)
        self.assertTrue(result["graph_attribution_complete"])
        nested = [rows[0], row("prefill", "vxVerifyGraph", 2, nested=1)]
        result = summarize_request(nested, "measured.0", generation(consumed=5, total=11), 5)
        self.assertEqual(result["graph_total_seconds"], 6)
        self.assertEqual(result["accounted_generation_sdk_seconds"], 6)
        self.assertFalse(result["graph_attribution_complete"])
        self.assertIsNone(result["graph_prefill_ms_per_input"])

    def test_nested_and_mixed_rows_do_not_double_count_graph_spans(self):
        rows = [
            row("prefill", "vsi_nn_RunGraph", 10),
            row("prefill", "vxProcessGraph", 6, nested=1),
            row("prefill", "vsi_nn_ConvertTensorToData", 4, nested=1),
        ]
        result = summarize_request(rows, "measured.0", generation(consumed=5, total=11), 5)
        self.assertEqual(result["graph_total_seconds"], 10)
        self.assertEqual(result["accounted_generation_sdk_seconds"], 10)
        self.assertEqual(result["io_seconds_by_phase"], {})
        self.assertTrue(result["graph_attribution_complete"])
        self.assertEqual(len(result["attribution_risks"]), 2)
        mixed = [row("prefill", "vsi_nn_RunGraph", 10, count=2, nested=1)]
        result = summarize_request(mixed, "measured.0", generation(consumed=5, total=11), 5)
        self.assertEqual(result["graph_total_seconds"], 0)
        self.assertFalse(result["graph_attribution_complete"])
        self.assertIsNone(result["graph_prefill_ms_per_input"])

    def test_single_output_has_no_decode_rate_and_run_summary_preserves_samples(self):
        rows = [row("prefill", "vsi_nn_RunGraph", 4)]
        sample = summarize_request(
            rows, "measured.0", generation(consumed=5, total=5, tokens=[21]), 5, [22]
        )
        self.assertEqual(sample["decode_steps"], 0)
        self.assertIsNone(sample["graph_decode_ms_per_token"])
        self.assertIsNone(sample["graph_decode_tokens_per_second"])
        self.assertFalse(sample["tokens_match_expected"])
        samples = [sample, sample | {"graph_total_seconds": 6}]
        summary = summarize_runs(samples)
        self.assertEqual(summary["samples"], samples)
        self.assertEqual(summary["metrics"]["graph_total_seconds"]["median"], 5)
        self.assertIsNone(summary["metrics"]["graph_decode_tokens_per_second"])
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_request(rows, "measured.0", generation(), 0)
        with self.assertRaisesRegex(ValueError, "smaller"):
            summarize_request(rows, "measured.0", generation(consumed=4), 5)
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_runs([])

    def test_generation_cli_helper_groups_graph_matrix_and_component_without_warmups(self):
        script = Path(__file__).resolve().parents[2] / "scripts/generate_opt.py"
        summarize_graph_metrics = runpy.run_path(str(script))["summarize_graph_metrics"]
        for backend in ("graph", "component"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                device = Path(directory)
                rows = [row("prefill", "vsi_nn_RunGraph", 999) | {"request": "warmup.0"}]
                runs = []
                references = [
                    {"prompt_length": 5, "tokens": [21, 22, 23, 24]},
                    {"prompt_length": 15, "tokens": [31, 32, 33, 34]},
                ]
                for index, reference in enumerate(
                    references if backend == "graph" else references[:1]
                ):
                    for iteration in range(2):
                        name = (
                            f"measured.{index}.{iteration}"
                            if backend == "graph"
                            else f"measured.{iteration}"
                        )
                        run = generation(
                            consumed=reference["prompt_length"] + 3,
                            total=9 + index + iteration,
                            tokens=reference["tokens"],
                        )
                        if backend == "graph":
                            run.update(request_index=index, iteration=iteration)
                        runs.append(run)
                        rows.extend(
                            [
                                row("prefill", "vsi_nn_RunGraph", 2 + index) | {"request": name},
                                row("decode", "vxProcessGraph", 4 + iteration, count=3)
                                | {"request": name},
                            ]
                        )
                with (device / "sdk-summary.tsv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)
                grouped = summarize_graph_metrics(
                    device,
                    runs,
                    {"tokens": [2] * 5},
                    {"requests": references} if backend == "graph" else None,
                    benchmark=backend == "graph",
                    expected=references[0]["tokens"],
                )
                self.assertEqual(len(grouped), 2 if backend == "graph" else 1)
                for summary in grouped.values():
                    self.assertEqual(summary["sample_count"], 2)
                    self.assertEqual(
                        summary["metrics"]["graph_decode_ms_per_token"]["median"], 1500
                    )
                    self.assertTrue(
                        all(sample["tokens_match_expected"] for sample in summary["samples"])
                    )
                    self.assertTrue(
                        all(
                            sample["request"].startswith("measured.")
                            for sample in summary["samples"]
                        )
                    )
                    self.assertLess(summary["metrics"]["graph_total_seconds"]["max"], 999)
                if backend == "component":
                    unchecked = summarize_graph_metrics(device, runs, {"tokens": [2] * 5})
                    self.assertTrue(
                        all(
                            sample["tokens_match_expected"] is None and sample["prompt_length"] == 5
                            for sample in unchecked["request.0"]["samples"]
                        )
                    )


if __name__ == "__main__":
    unittest.main()
