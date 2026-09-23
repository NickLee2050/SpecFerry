"""Keep timing observations separate from correctness and hardware acceptance."""

import contextlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.opt.acceptance import evaluate_benchmark, summarize, valid_generation

ROOT = Path(__file__).resolve().parents[2]


def request():
    return {
        "tokens": [2, 7],
        "capacity": 8,
        "weight_bytes": 100,
        "kv_bytes": 32,
        "selection": {"resolved_config": {"eos_token_id": 2}},
    }


def generation():
    return {
        "tokens": [3, 4, 5],
        "consumed": 4,
        "stop_reason": "max_new_tokens",
        "prefill_seconds": 2.0,
        "first_token_seconds": 3.0,
        "token_seconds": [0.5, 0.5],
        "total_seconds": 4.0,
    }


def evidence():
    return {
        "returncode": 0,
        "timeout": False,
        "process_group_exited": True,
        "remaining_processes": [],
        "device_recovery_required": False,
        "driver_traced": False,
        "boot_id": "fixture",
        "sdk_sha256": {"library": "fixture"},
        "runtime_options": {"VIV_VX_ENABLE_PRINT_TARGET": "0", "VIV_VX_PROFILE": None},
    }


class AcceptanceTests(unittest.TestCase):
    def fixture(self, directory):
        for name in ("warmup.0", "measured.0", "measured.1"):
            result = generation()
            if name.startswith("warmup"):
                result.update(prefill_seconds=50, first_token_seconds=100, total_seconds=101)
            (directory / f"{name}.json").write_text(json.dumps(result))
        execution = {
            "status": "executed",
            "phase": "complete",
            "released": True,
            "layers": 24,
            "steps": 12,
            "cache_writes": 288,
            "uploads": 312,
            "upload_bytes": 1248,
            "reads": 9,
            "read_bytes": 36,
            "initialize_seconds": 5,
            "host_peak_rss_kib": 1024,
        }
        (directory / "execution.json").write_text(json.dumps(execution))
        phases = [
            "before_initialize",
            "after_initialize",
            "warmup.0",
            "measured.0",
            "measured.1",
            "after_release",
        ]
        (directory / "host-resources.jsonl").write_text(
            "".join(json.dumps({"phase": phase, "rss_bytes": 1024}) + "\n" for phase in phases)
        )
        return execution

    def evaluate(self, directory, execution_evidence=None):
        return evaluate_benchmark(
            directory,
            request(),
            {"tokens": [3, 4, 5]},
            execution_evidence or evidence(),
            1,
            2,
            3,
        )

    def test_warmup_is_checked_but_excluded_from_timing_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            self.fixture(directory)
            result = self.evaluate(directory)
            self.assertEqual(result["status"], "numerical_pass")
            self.assertEqual(result["metrics"]["first_token_seconds"]["median"], 3)
            self.assertEqual(result["metrics"]["first_token_seconds"]["count"], 2)
            self.assertEqual(result["metrics"]["decode_tokens_per_second"], 2)
            self.assertIsNone(result["physical_device_peak_bytes"])
            self.assertEqual(result["measured_host_rss_growth_bytes"], 0)
            self.assertFalse(result["hardware_execution_proven"])
            (directory / "warmup.0.json").write_text(
                json.dumps(generation() | {"tokens": [3, 4, 6]})
            )
            failed = self.evaluate(directory)
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["metrics"])

    def test_missing_output_invalid_timing_and_extra_transfers_fail(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            execution = self.fixture(directory)
            for change in (
                {"released": False},
                {"read_bytes": 40},
                {"cache_writes": 287},
                {"layers": 23},
            ):
                with self.subTest(change=change):
                    (directory / "execution.json").write_text(json.dumps(execution | change))
                    self.assertEqual(self.evaluate(directory)["status"], "failed")
            (directory / "execution.json").write_text(json.dumps(execution))
            for change in (
                {"prefill_seconds": -1},
                {"first_token_seconds": float("nan")},
                {"total_seconds": 3.1},
                {"token_seconds": [0.5]},
                {"consumed": 5},
                {"stop_reason": "capacity"},
            ):
                with self.subTest(change=change):
                    self.assertFalse(
                        valid_generation(generation() | change, request(), [3, 4, 5], 3)
                    )
            (directory / "measured.1.json").unlink()
            self.assertEqual(self.evaluate(directory)["status"], "failed")

    def test_correct_bytes_do_not_hide_process_failure_or_traced_timing(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            self.fixture(directory)
            for change in (
                {"returncode": -8},
                {"timeout": True},
                {"remaining_processes": [{"pid": 1}]},
                {"device_recovery_required": True},
                {"driver_traced": True},
                {"runtime_options": {"VIV_VX_ENABLE_PRINT_TARGET": "1"}},
                {"runtime_options": {"VIV_VX_ENABLE_PRINT_TARGET": "0", "VIV_VX_PROFILE": "1"}},
                {"runtime_options": {"VIV_VX_ENABLE_PRINT_TARGET": "0", "VIV_MEMORY_PROFILE": "1"}},
            ):
                with self.subTest(change=change):
                    result = self.evaluate(directory, evidence() | change)
                    self.assertEqual(result["status"], "failed")
                    self.assertIsNone(result["metrics"])

    def test_eos_and_one_prediction_have_no_decode_latency(self):
        result = generation() | {
            "tokens": [2],
            "consumed": 2,
            "stop_reason": "eos",
            "token_seconds": [],
        }
        self.assertTrue(valid_generation(result, request(), [2], 3))
        self.assertFalse(valid_generation(result | {"consumed": 3}, request(), [2], 3))
        capacity = request() | {"capacity": 2}
        result.update(tokens=[3], stop_reason="capacity")
        self.assertTrue(valid_generation(result, capacity, [3], 3))

    def test_all_numerical_passes_still_leave_hardware_acceptance_open(self):
        stages = [{"name": "selection", "passed": True}, {"name": "teacher", "passed": True}]
        passed = [{"status": "numerical_pass"}] * 2
        report = summarize(stages, passed, 2)
        self.assertEqual(report["status"], "hardware_pending")
        self.assertFalse(report["formal_performance_accepted"])
        self.assertFalse(summarize(stages, passed[:1], 2)["numerical_pass"])
        self.assertFalse(summarize([], passed, 2)["numerical_pass"])
        self.assertFalse(summarize(stages[:1], passed, 2)["numerical_pass"])
        self.assertFalse(summarize(stages, [], 0)["numerical_pass"])
        growth = [{"status": "numerical_pass", "measured_host_rss_growth_bytes": 1024}] * 2
        self.assertIn(
            "explain host allocation growth across repeated requests",
            summarize(stages, growth, 2)["remaining_gates"],
        )

    def test_failed_preflight_stops_before_full_model_execution(self):
        spec = importlib.util.spec_from_file_location(
            "acceptance_cli", ROOT / "scripts/check_np101_acceptance.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        args = SimpleNamespace(
            cycles=2,
            binary=Path("generator"),
            io_binary=Path("selector"),
            model=Path("weights"),
            sdk_lib=Path("sdk"),
            shader_header=Path("header"),
            timeout=30,
        )
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            (output / "selection").mkdir()
            with (
                patch.object(cli, "device_lock", return_value=contextlib.nullcontext()),
                patch.object(cli.os, "access", return_value=True),
                patch.object(
                    cli, "snapshot_binary", side_effect=lambda source, destination: destination
                ),
                patch.object(cli, "record_sources"),
                patch.object(
                    cli, "run_device", return_value=evidence() | {"returncode": -8}
                ) as run,
                patch.object(cli, "evaluate", return_value={"status": "numerical_pass"}),
            ):
                report = cli.run_suite(args, output, {"selection": {}, "requests": [{}]})
            self.assertEqual(run.call_count, 1)
            self.assertFalse(report["numerical_pass"])
            self.assertEqual(report["completed_benchmarks"], 0)


if __name__ == "__main__":
    unittest.main()
