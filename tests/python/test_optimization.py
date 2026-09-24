"""Protect short graph gates against stale evidence and misleading performance passes."""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.opt.optimization import (
    CASES,
    capacity_comparison,
    evaluate,
    launch_count,
    verify_prepared,
    write_results,
)
from specferry.validation.device import fingerprint, write_json

ROOT = Path(__file__).resolve().parents[2]


def execution():
    return {
        "returncode": 0,
        "timeout": False,
        "process_group_exited": True,
        "remaining_processes": [],
        "device_recovery_required": False,
        "driver_traced": False,
        "boot_id": "test-boot",
        "binary_sha256": "current",
        "sdk_sha256": {"libGAL.so": "current"},
        "runtime_options": {"SPECFERRY_SDK_TIMING": "summary", "VIV_VX_ENABLE_PRINT_TARGET": "0"},
    }


class OptimizationTests(unittest.TestCase):
    def test_runner_passes_cpu_oracle_and_uses_current_budget_for_old_fixtures(self):
        spec = importlib.util.spec_from_file_location(
            "graph_cli", ROOT / "scripts/check_np101_optimization.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            case = self.fixture(directory) | {"timeout": 1}
            write_json(directory / "prepared.json", {})
            arguments = [
                "check",
                "--output",
                str(directory / "result"),
                "--prepared",
                str(directory),
                "--case",
                "decode",
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(cli, "verify_prepared", return_value={"cases": {"decode": case}}),
                patch.object(cli, "prerequisites", return_value=(execution(), None)),
                patch.object(cli, "device_lock", return_value=contextlib.nullcontext()),
                patch.object(cli, "snapshot_binary", return_value=Path("binary")),
                patch.object(cli, "record_sources"),
                patch.object(cli, "run_device", return_value=execution()) as run,
                patch.object(cli, "evaluate", return_value={"status": "failed", "metrics": None}),
                patch.object(cli, "write_results"),
                patch.dict(cli.os.environ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(cli.main(), 1)
            run.assert_called_once()
            native_arguments = run.call_args.args[1]
            self.assertEqual(len(native_arguments), 7)
            self.assertEqual(Path(native_arguments[-1]).read_text(), "3 4 5\n")
            self.assertEqual(run.call_args.args[4], CASES["decode"]["timeout"])

    def fixture(self, directory, repeats=1):
        request = {
            "tokens": [2, 7],
            "capacity": 8,
            "weight_bytes": 100,
            "manifest_sha256": "weights",
            "selection": {"resolved_config": {"eos_token_id": 2}},
        }
        case = {
            "requests": [{"request": request, "expected_tokens": [3, 4, 5]}] * repeats,
            "block": 1,
            "maximum": 3,
            "layers": 1,
            "heads": 2,
            "head_dim": 2,
            "capacity": 8,
        }
        sample = {
            "tokens": [3, 4, 5],
            "consumed": 4,
            "stop_reason": "max_new_tokens",
            "prefill_seconds": 2.0,
            "first_token_seconds": 3.0,
            "total_seconds": 4.0,
            "token_seconds": [0.5, 0.5],
            "launches": 4,
            "uploads": 4,
            "upload_bytes": 48,
            "reads": 3,
            "read_bytes": 12,
        }
        prefix = np.arange(16, dtype=np.float32).reshape(2, 4, 2) / 16
        for index in range(repeats):
            write_json(directory / f"measured.{index}.json", sample)
            np.savez(
                directory / f"cache.{index}.npz",
                **{"layer.0.keys": prefix, "layer.0.values": prefix},
            )
            actual = np.full((2, 8, 2), 9, dtype="<f2")
            actual[:, :4] = prefix
            for kind in ("keys", "values"):
                actual.tofile(directory / f"measured.{index}.layer.0.{kind}.bin")
        write_json(
            directory / "run.json",
            {
                "status": "executed",
                "released": True,
                "layers": 1,
                "prefill_block": 1,
                "capacity": 8,
                "launches": 4 * repeats,
                "weight_payload_bytes": 100,
                "initialize_seconds": 1,
                "host_peak_rss_kib": 128,
            },
        )
        phases = ["before_initialize", "after_initialize"]
        phases += [f"measured.{i}" for i in range(repeats)] + ["after_release"]
        (directory / "host-resources.jsonl").write_text(
            "".join(
                json.dumps({"phase": phase, "rss_bytes": 1000 + i}) + "\n"
                for i, phase in enumerate(phases)
            )
        )
        counts = {
            "vsi_nn_RunGraph": 4 * repeats,
            "vsi_nn_CopyDataToTensor": 4 * repeats,
            "vsi_nn_ConvertTensorToData": 3 * repeats,
        }
        (directory / "sdk-timing.tsv").write_text(
            "request\tphase\tcomponent\tapi\tcalls\ttotal_seconds\tmax_seconds\texceptions\n"
            + "".join(
                f"measured.0\tdecode\tmodel\t{api}\t{count}\t0.1\t0.01\t0\n"
                for api, count in counts.items()
            )
            + "diagnostic.0\tdecode\tmodel\tvsi_nn_ConvertTensorToData\t2\t10\t5\t0\n"
        )
        return case

    def test_tail_and_eos_use_actual_predictions(self):
        self.assertEqual(launch_count(32, 4, 4), 11)
        self.assertEqual(launch_count(7, 1, 4), 4)
        self.assertEqual(launch_count(3, 4, 4), 6)
        for dimensions in ((0, 1, 1), (1, 0, 1), (4, 4, 0), (4, 4, 9)):
            with self.assertRaises(ValueError):
                launch_count(*dimensions)

    def test_correct_gate_reports_small_sample_scope_and_scalar_transfers(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            case = self.fixture(directory)
            report = evaluate(case, directory, directory, execution(), True)
            self.assertEqual(report["status"], "numerical_pass")
            self.assertEqual(report["metrics"]["decode_tokens_per_second"], 2)
            self.assertEqual(report["metrics"]["warmups"], 0)
            self.assertFalse(report["hardware_execution_proven"])
            self.assertFalse(report["formal_performance_accepted"])
            write_results(directory / "results.md", report)
            self.assertIn("2.0000 tokens/s", (directory / "results.md").read_text())

    def test_corruption_transfer_and_lifecycle_failures_withhold_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            case = self.fixture(directory)
            sample_path = directory / "measured.0.json"
            original = json.loads(sample_path.read_text())
            for change in (
                {"tokens": [3, 4, 6]},
                {"uploads": 5},
                {"read_bytes": 16},
                {"launches": 5},
                {"total_seconds": float("nan")},
            ):
                with self.subTest(change=change):
                    sample_path.write_text(json.dumps(original | change))
                    report = evaluate(case, directory, directory, execution(), True)
                    self.assertIsNone(report["metrics"])
            write_json(sample_path, original)
            for change in (
                {"returncode": -11},
                {"device_recovery_required": True},
                {"driver_traced": True},
            ):
                self.assertIsNone(
                    evaluate(case, directory, directory, execution() | change, True)["metrics"]
                )
            path = directory / "measured.0.layer.0.keys.bin"
            original_bytes = path.read_bytes()
            for invalid_bytes in (b"short", original_bytes + b"\0"):
                path.write_bytes(invalid_bytes)
                report = evaluate(case, directory, directory, execution(), True)
                self.assertFalse(report["checks"]["cache.0.layer.0.keys"])
                self.assertIsNone(report["metrics"])
            self.fixture(directory)
            path = directory / "run.json"
            write_json(path, json.loads(path.read_text()) | {"released": False})
            self.assertIsNone(evaluate(case, directory, directory, execution(), True)["metrics"])

    def test_reset_detects_changes_smaller_than_cpu_tolerance(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            case = self.fixture(directory, repeats=3)
            path = directory / "measured.2.layer.0.keys.bin"
            actual = np.fromfile(path, dtype="<f2")
            actual[0] += 0.01
            actual.tofile(path)
            report = evaluate(case, directory, directory, execution(), True)
            self.assertTrue(report["checks"]["cache.2.layer.0.keys"])
            self.assertFalse(report["checks"]["reset.2.layer.0.keys"])
            self.assertIsNone(report["metrics"])

    def test_reverify_or_bad_timing_accounting_cannot_pass(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            case = self.fixture(directory)
            path = directory / "sdk-timing.tsv"
            original = path.read_text()
            for observation in (
                "measured.0\tdecode\tmodel\tvxVerifyGraph\t1\t0.1\t0.1\t0\n",
                "measured.0\tdecode\tmodel\tunexpected\t1\t10\t10\t0\n",
            ):
                path.write_text(original + observation)
                self.assertIsNone(
                    evaluate(case, directory, directory, execution(), True)["metrics"]
                )
            path.unlink()
            self.assertIsNone(evaluate(case, directory, directory, execution(), True)["metrics"])

    def test_prepared_data_rejects_changed_model_and_fixtures(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            manifest = directory / "deployment-manifest.json"
            manifest.write_text("model")
            fixture = directory / "tokens.txt"
            fixture.write_text("2 7")
            metadata = {
                "status": "prepared",
                "deployment_manifest_sha256": fingerprint(manifest),
                "files": {"tokens.txt": fingerprint(fixture)},
            }
            write_json(directory / "prepared.json", metadata)
            verify_prepared(directory, directory)
            fixture.write_text("2 8")
            with self.assertRaises(ValueError):
                verify_prepared(directory, directory)
            manifest.write_text("different model")
            with self.assertRaises(ValueError):
                verify_prepared(directory, directory)

    def test_capacity_comparison_requires_same_inputs_and_environment(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            case = self.fixture(directory)
            base = evaluate(case, directory, directory, execution(), True)
            previous, current = copy.deepcopy(base), copy.deepcopy(base)
            previous["capacity"], current["capacity"] = 64, 128
            self.assertIn("capacity64", capacity_comparison(previous, current))
            for field, value in (("boot_id", "old"), ("binary_sha256", "old"), ("sdk_sha256", {})):
                changed = copy.deepcopy(current)
                changed["evidence"][field] = value
                with self.assertRaises(ValueError):
                    capacity_comparison(previous, changed)
            current["requests"][0]["request"]["tokens"] = [8, 9]
            with self.assertRaises(ValueError):
                capacity_comparison(previous, current)

    def test_missing_or_stale_small_gate_stops_before_hardware(self):
        spec = importlib.util.spec_from_file_location(
            "graph_cli", ROOT / "scripts/check_np101_optimization.py"
        )
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            args = SimpleNamespace(case="decode", cache_gate=directory, previous=None)
            write_json(directory / "summary.json", {"status": "numerical_pass", "mode": "stack"})
            write_json(directory / "execution-evidence.json", execution())
            with (
                patch.object(cli, "host_boot_id", return_value="test-boot"),
                patch.object(cli, "fingerprint", return_value="current"),
            ):
                cli.prerequisites(args)
                for change in (
                    {"boot_id": "old"},
                    {"binary_sha256": "old"},
                    {"returncode": -11},
                    {"sdk_sha256": {"libGAL.so": "old"}},
                ):
                    write_json(directory / "execution-evidence.json", execution() | change)
                    with self.assertRaises(ValueError):
                        cli.prerequisites(args)
            args.cache_gate = None
            with self.assertRaises(ValueError):
                cli.prerequisites(args)


if __name__ == "__main__":
    unittest.main()
