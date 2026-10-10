"""Numerical regressions for the compact reporting and graph-cache checks."""

import contextlib
import io
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.data.checkpoint import sha256
from specferry.models.opt.config import Components
from specferry.models.opt.graph_validation import (
    evaluate,
    evaluate_prefix_readback,
    prefix_readback_scope,
    validate_prefix_fixture,
)
from specferry.models.opt.metrics import summarize
from specferry.validation import device as device_runner
from specferry.validation import recovery_controls
from specferry.validation.allocation import MIB, capacity_result, check_segment_budget
from specferry.validation.arrays import compare_arrays


class RecoveryControlTests(unittest.TestCase):
    def test_running_child_is_stopped_when_health_monitor_reports_fault(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = Mock(pid=12345)
            process.wait.side_effect = [subprocess.TimeoutExpired("control", 5), -15]
            with (
                patch.object(device_runner, "RECOVERY_ROOT", root),
                patch.object(device_runner.subprocess, "Popen", return_value=process),
                patch.object(device_runner, "terminate_group") as terminate,
                patch.object(device_runner, "process_group_members", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                evidence = device_runner.run_device(
                    Path(sys.executable),
                    [],
                    root / "device",
                    root,
                    60,
                    abort_check=lambda: True,
                )
            terminate.assert_called_once()
            self.assertFalse(evidence["timeout"])
            self.assertTrue(evidence["device_recovery_required"])
            self.assertIn("kernel fault", evidence["health_monitor_failure"])
            self.assertTrue((root / "np101-recovery-required.json").exists())

    def test_kernel_page_fault_and_watchdog_are_fatal_but_hdma_wait_is_not(self):
        for line in (
            "kernel: BUG: unable to handle page fault for address: 0000000100000008",
            "kernel: Oops: 0000 [#1] SMP PTI",
            "kernel: [galcore]: NP[0] hang, automatic recovery.",
            "--------gckEDMA_StartTransfer---timeout---src:0x55c24e40",
        ):
            self.assertIsNotNone(recovery_controls.FAULT.search(line), line)
        self.assertIsNone(recovery_controls.FAULT.search("kernel: Waiting for HDMA Transfer..."))

    def test_delayed_fault_stops_before_llm_even_when_numbers_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = SimpleNamespace(
                case="recovery",
                prepare_only=False,
                sudo_read_only=False,
                cold_restarted=False,
                timeout=None,
            )
            evidence = dict(
                returncode=0,
                timeout=False,
                process_group_exited=True,
                remaining_processes=[],
                device_recovery_required=False,
            )
            healthy = dict(faults=[], owners="")
            faulty = dict(faults=["NP[0] hang"], owners="")

            def device_run(binary, arguments, output, sdk, deadline, **kwargs):
                output.mkdir(parents=True)
                return evidence

            with (
                patch.object(recovery_controls, "RECOVERY_ROOT", root),
                patch.object(
                    recovery_controls,
                    "controls",
                    return_value=[
                        ("vendor-conv", Path("demo"), [], 60),
                        ("old-component", Path("llm"), [], 180),
                    ],
                ),
                patch.object(recovery_controls, "snapshot", side_effect=[healthy, healthy, faulty]),
                patch.object(
                    recovery_controls, "device_lock", return_value=contextlib.nullcontext()
                ),
                patch.object(recovery_controls, "run_device", side_effect=device_run) as runner,
                patch.object(recovery_controls, "numerical_check", return_value=True),
                patch.object(recovery_controls.time, "sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(recovery_controls.run_controls(root / "output", options, True), 1)
            self.assertEqual(runner.call_count, 1)
            self.assertTrue((root / "np101-recovery-required.json").exists())
            incident = json.loads((root / "output/vendor-conv/incident.json").read_text())
            self.assertEqual(incident["kernel_gate"]["faults"], faulty["faults"])

    def test_conv_rejects_nan_error_and_llm_rejects_wrong_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = Path(temporary)
            (device / "sdk.log").write_text("total=64 mismatches=0 max_abs_err=nan")
            self.assertFalse(recovery_controls.numerical_check("vendor-conv", device))
            measured = dict(tokens=[78, 632], consumed=6, stop_reason="max_new_tokens")
            execution = dict(
                phase="complete",
                released=True,
                const_live_bytes=0,
                nonconst_live_bytes=0,
                const_peak_bytes=1,
                nonconst_peak_bytes=1,
            )
            (device / "measured.0.json").write_text(json.dumps(measured))
            (device / "execution.json").write_text(json.dumps(execution))
            self.assertFalse(recovery_controls.numerical_check("old-component", device))

    def test_repository_conv_requires_both_finite_results_and_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            device = Path(temporary)
            log = "PASS iteration=0 max_abs_error=0.003\nPASS iteration=1 max_abs_error=0.004\n"
            complete = "Completed; graph/context released\n"
            for text, expected in (
                (log + complete, True),
                (log, False),
                (log.replace("0.004", "nan") + complete, False),
                (log.replace("0.004", "0.2") + complete, False),
                (log.splitlines(True)[0] + complete, False),
            ):
                with self.subTest(text=text):
                    (device / "sdk.log").write_text(text)
                    self.assertEqual(
                        recovery_controls.numerical_check("repository-conv", device), expected
                    )

    def test_prepare_only_never_checks_or_opens_device(self):
        with (
            patch.object(
                recovery_controls, "controls", return_value=[("vendor-conv", Path("demo"), [], 60)]
            ),
            patch.object(recovery_controls, "snapshot") as snapshot,
            patch.object(recovery_controls, "run_device") as runner,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                recovery_controls.run_controls(
                    Path("unused"),
                    SimpleNamespace(case="conv", prepare_only=True, timeout=None),
                    False,
                ),
                0,
            )
        snapshot.assert_not_called()
        runner.assert_not_called()


def write_generation_cache(fixture, device, metadata, expected_values):
    """Write independent CPU prefixes and padded device backings for validator tests."""
    for index, request in enumerate(metadata["requests"]):
        for layer in range(metadata["layers"]):
            expected = expected_values(layer, request["consumed"])
            actual = np.zeros(
                (metadata["heads"], metadata["capacity"], metadata["width"]), dtype="<f2"
            )
            actual[:, : request["consumed"]] = expected
            for name in ("keys", "values"):
                expected.tofile(fixture / f"expected.{index}.{layer}.{name}.bin")
                actual.tofile(device / f"measured.{index}.layer.{layer}.{name}.bin")


class GraphEvidenceFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture, self.device = self.root / "fixture", self.root / "device"
        self.fixture.mkdir()
        self.device.mkdir()
        self.results = []

    def save_metadata(self):
        (self.device / "initialization.json").write_text(json.dumps(self.initialized))
        (self.device / "run.json").write_text(json.dumps(self.native_run))
        for index, result in enumerate(self.results):
            (self.device / f"measured.{index}.json").write_text(json.dumps(result))


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
        (self.root / "predictions.txt").write_text("0 0\n")
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
                "layer.0.probabilities": (1, 2, 1, step + 1),
                "layer.0.keys": (2, step + 1, 2),
                "layer.0.values": (2, step + 1, 2),
            }
            for name in (
                "query",
                "key",
                "value",
                "attended",
                "mixer",
                "attention_residual",
                "attention_norm",
                "ffn_residual",
            ):
                shapes[f"layer.0.{name}"] = (1, 1, 4)
            shapes["layer.0.mlp"] = (1, 4)
            for name in ("fc1", "activation"):
                shapes[f"layer.0.{name}"] = (1, 8)
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
        path = self.root / "teacher.0.layer.0.query.bin"
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
        (self.root / "predictions.txt").write_text("0 4\n")
        self.metadata["fixture_sha256"]["predictions.txt"] = sha256(self.root / "predictions.txt")
        self.save()
        with self.assertRaisesRegex(ValueError, "logits"):
            self.validate()

    def test_attention_probability_shape_is_checked(self):
        key = "teacher.1.layer.0.probabilities"
        self.metadata["observations"][key]["shape"] = [1, 2, 1, 4]
        self.save()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.validate()

    def test_grouped_scope_distinguishes_intermediate_and_graph_outputs(self):
        metadata = {"layers": 5}
        scope = prefix_readback_scope(metadata, "boundaries", [2, 3, 5], decoder_group_size=2)
        self.assertEqual(scope["decoder_graph_output_layers"], [2, 4, 5])
        self.assertEqual(scope["intermediate_decoder_readback_layers"], [3])
        scope = prefix_readback_scope(metadata, "outputs-kv", [], decoder_group_size=2)
        self.assertEqual(scope["execution_graphs"], 5)
        self.assertEqual(scope["planned_checks"], 54)
        self.assertTrue(scope["kv_checks_requested"])

    def saved_coarse_result(self, repeat=False):
        device = self.root / "device"
        device.mkdir(exist_ok=True)
        self.metadata["weight_payload_bytes"] = 64
        steps = [
            {
                "index": step,
                "executed": True,
                "checks": 2,
                "failures": 0,
                "reads": 2,
                "read_bytes": 14,
                "uploads": 1,
                "upload_bytes": 12,
            }
            for step in range(2)
        ]
        validation = {
            "mode": "outputs",
            "boundary_layers": [],
            "layer_count": 1,
            "decoder_group_size": 1,
            "variant": "merged-kv",
            "execution_graphs": 3,
            "planned_steps": 2,
            "planned_checks": 4,
            "completed_steps": 2,
            "checks": 4,
            "failures": 0,
            "kv_checks_requested": False,
            "weight_readback_requested": False,
            "loaded_weight_payload_bytes": 64,
            "verified_weight_bytes": 0,
            "transfer_scope": "explicit_tensor_helpers_excluding_sdk_internal_and_addtensor_initialization",
            "transfers": {"reads": 4, "read_bytes": 28, "uploads": 2, "upload_bytes": 24},
            "steps": steps,
        }
        execution = {"phase": "complete", "released": True, "checks": 4, "failures": 0}
        for cycle in range(2 if repeat else 1):
            directory = device / f"cycle.{cycle}" if repeat else device
            directory.mkdir(exist_ok=True)
            (directory / "validation.json").write_text(json.dumps(validation))
            (directory / "execution.json").write_text(json.dumps(execution))
            for step in range(2):
                name = f"teacher.{step}.logits.bin"
                (directory / name).write_bytes((self.root / name).read_bytes())
                np.array([0], dtype="<i4").tofile(directory / f"teacher.{step}.selected.bin")
        if repeat:
            execution["checks"] = 8
            (device / "execution.json").write_text(json.dumps(execution))
        return device

    def evaluate_coarse_result(
        self, device, repeat=False, decoder_group_size=1, variant="merged-kv"
    ):
        return evaluate_prefix_readback(
            self.root,
            device,
            self.metadata,
            {
                "returncode": 0,
                "timeout": False,
                "process_group_exited": True,
                "remaining_processes": [],
                "device_recovery_required": False,
            },
            "outputs",
            [],
            repeat=repeat,
            decoder_group_size=decoder_group_size,
            variant=variant,
        )

    def test_variant_evaluation_rejects_wrong_variant_and_incomplete_weight_payload(self):
        device = self.saved_coarse_result()
        path = device / "validation.json"
        for variant in ("merged-kv-local-const", "merged-kv-local-mutable"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    self.evaluate_coarse_result(device, variant=variant)["status"], "failed"
                )
                validation = json.loads(path.read_text())
                validation["variant"] = variant
                path.write_text(json.dumps(validation))
                self.assertEqual(
                    self.evaluate_coarse_result(device, variant=variant)["status"], "numerical_pass"
                )
                validation["loaded_weight_payload_bytes"] = 63
                path.write_text(json.dumps(validation))
                self.assertEqual(
                    self.evaluate_coarse_result(device, variant=variant)["status"], "failed"
                )
                self.saved_coarse_result()

    def saved_outputs_kv_result(self):
        device = self.saved_coarse_result()
        path = device / "validation.json"
        validation = json.loads(path.read_text())
        validation.update(
            mode="outputs-kv",
            variant="merged-kv",
            planned_checks=14,
            checks=14,
            kv_checks_requested=True,
            transfers={"reads": 8, "read_bytes": 156, "uploads": 2, "upload_bytes": 24},
        )
        source = (np.arange(16).reshape(2, 4, 2) + 1).astype("<f2") / 4
        for step in range(2):
            validation["steps"][step].update(checks=6 + 2 * step, reads=4, read_bytes=78)
            for kind in ("keys", "values"):
                name = f"teacher.{step}.layer.0.{kind}"
                source[:, : step + 1].tofile(self.root / f"{name}.bin")
                actual = np.zeros((2, 4, 2), dtype="<f2")
                actual[:, : step + 1] = source[:, : step + 1]
                actual.tofile(device / f"{name}.backing.bin")
        path.write_text(json.dumps(validation))
        execution = {"phase": "complete", "released": True, "checks": 14, "failures": 0}
        (device / "execution.json").write_text(json.dumps(execution))
        return device

    def test_outputs_kv_checks_current_history_suffix_and_actual_read_counts(self):
        def result(device):
            return evaluate_prefix_readback(
                self.root,
                device,
                self.metadata,
                {
                    "returncode": 0,
                    "timeout": False,
                    "process_group_exited": True,
                    "remaining_processes": [],
                    "device_recovery_required": False,
                },
                "outputs-kv",
                [],
                variant="merged-kv",
            )

        device = self.saved_outputs_kv_result()
        self.assertEqual(result(device)["status"], "numerical_pass")
        for damage in ("current", "history", "negative_zero_suffix", "size", "read_count"):
            with self.subTest(damage=damage):
                device = self.saved_outputs_kv_result()
                path = device / "teacher.1.layer.0.keys.backing.bin"
                data = np.fromfile(path, dtype="<u2").reshape(2, 4, 2)
                if damage == "current":
                    data[:, 1] = 0
                elif damage == "history":
                    data[0, 0, 0] ^= 1
                elif damage == "negative_zero_suffix":
                    data[0, 3, 0] = 0x8000
                data.tofile(path)
                if damage == "size":
                    path.write_bytes(bytes(2))
                elif damage == "read_count":
                    metadata_path = device / "validation.json"
                    validation = json.loads(metadata_path.read_text())
                    validation["steps"][0]["reads"] += 1
                    metadata_path.write_text(json.dumps(validation))
                self.assertEqual(result(device)["status"], "failed")

    def test_grouped_evaluation_rejects_a_result_from_the_other_partition(self):
        device = self.saved_coarse_result()
        self.assertEqual(
            self.evaluate_coarse_result(device, decoder_group_size=2)["status"], "failed"
        )
        path = device / "validation.json"
        validation = json.loads(path.read_text())
        validation["decoder_group_size"] = 2
        path.write_text(json.dumps(validation))
        self.assertEqual(
            self.evaluate_coarse_result(device, decoder_group_size=2)["status"], "numerical_pass"
        )
        validation["execution_graphs"] = 4
        path.write_text(json.dumps(validation))
        self.assertEqual(
            self.evaluate_coarse_result(device, decoder_group_size=2)["status"], "failed"
        )

    def test_coarse_evaluation_requires_both_complete_numerical_outputs(self):
        device = self.saved_coarse_result()
        result = self.evaluate_coarse_result(device)
        self.assertEqual(result["status"], "numerical_pass")
        self.assertEqual(result["scope"]["planned_checks"], 4)
        self.assertFalse(result["scope"]["kv_checks_requested"])
        self.assertFalse(result["full_model_lifecycle_accepted"])
        for damage in ("missing", "short", "nonfinite", "wrong_logits", "wrong_token"):
            with self.subTest(damage=damage):
                device = self.saved_coarse_result()
                path = device / "teacher.1.logits.bin"
                if damage == "missing":
                    path.unlink()
                elif damage == "short":
                    path.write_bytes(bytes(2))
                elif damage == "nonfinite":
                    np.full(5, np.nan, dtype="<f2").tofile(path)
                elif damage == "wrong_logits":
                    np.zeros(5, dtype="<f2").tofile(path)
                else:
                    np.array([1], dtype="<i4").tofile(device / "teacher.1.selected.bin")
                self.assertEqual(self.evaluate_coarse_result(device)["status"], "failed")

    def test_clean_exit_does_not_accept_incomplete_or_unexpected_readback(self):
        for damage in ("missing", "bad_json", "mode", "steps", "transfers", "release"):
            with self.subTest(damage=damage):
                device = self.saved_coarse_result()
                path = device / "validation.json"
                validation = json.loads(path.read_text())
                if damage == "missing":
                    path.unlink()
                elif damage == "bad_json":
                    path.write_text("{")
                elif damage == "release":
                    path = device / "execution.json"
                    execution = json.loads(path.read_text())
                    execution["released"] = False
                    path.write_text(json.dumps(execution))
                else:
                    if damage == "mode":
                        validation["mode"] = "full"
                    elif damage == "steps":
                        validation["completed_steps"] = 1
                    else:
                        validation["transfers"]["reads"] += 1
                    path.write_text(json.dumps(validation))
                self.assertEqual(self.evaluate_coarse_result(device)["status"], "failed")

    def test_repeat_requires_two_complete_instances(self):
        device = self.saved_coarse_result(repeat=True)
        result = self.evaluate_coarse_result(device, repeat=True)
        self.assertEqual(result["status"], "numerical_pass")
        self.assertEqual(result["scope"]["planned_checks"], 8)
        (device / "cycle.1" / "validation.json").unlink()
        self.assertEqual(self.evaluate_coarse_result(device, repeat=True)["status"], "failed")


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
            "decoder_group_size": 2,
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
                ("decoder_group_size", 1),
                ("decoder_group_size", 2.0),
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


class GenerationCommandTests(GraphEvidenceFixture):
    def test_prepare_and_benchmark_share_options_and_three_path_contract(self):
        path = Path(__file__).resolve().parents[2] / "scripts/generate_opt.py"
        main = runpy.run_path(str(path))["main"]
        request = dict(capacity=16, selection={"mode": "greedy"}, tokens=[2] * 5, prompt="test")
        metadata = {
            "layers": 2,
            "requests": [{"prompt_length": 5, "tokens": [1, 2, 3], "consumed": 7}],
        }
        generation = metadata["requests"][0] | dict(
            request_index=0,
            iteration=0,
            prefill_seconds=1,
            first_token_seconds=1,
            total_seconds=3,
            token_seconds=[1, 1],
        )
        tokenizer = Mock()
        tokenizer.decode.return_value = "test"

        def prepare_request(*arguments, **kwargs):
            arguments[2].mkdir()
            return request, tokenizer

        for prepare_only in (True, False):
            output = self.root / f"benchmark.{prepare_only}"
            arguments = [
                str(path),
                "--backend",
                "graph",
                "--block",
                "4",
                "--benchmark",
                "--warmups",
                "1",
                "--repeats",
                "3",
                "--max-new-tokens",
                "3",
                "--output",
                str(output),
            ]
            if prepare_only:
                arguments.append("--prepare-only")
            runner = Mock(return_value={})
            with (
                self.subTest(prepare_only=prepare_only),
                patch.object(sys, "argv", arguments),
                patch.dict(
                    main.__globals__,
                    prepare_request=prepare_request,
                    device_lock=Mock(return_value=contextlib.nullcontext()),
                    run_device=runner,
                    lifecycle=Mock(return_value=True),
                ),
                patch("specferry.models.opt.graph_validation.prepare", return_value=metadata),
                patch(
                    "specferry.models.opt.graph_validation.evaluate_benchmark",
                    return_value=({"checked": True}, [generation]),
                ) as evaluator,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(), 0)
            options = dict(
                line.split()
                for line in (output / "request/run-options.txt").read_text().splitlines()
            )
            expected = dict(block="4", benchmark="1", max_new_tokens="3", warmups="1", repeats="3")
            self.assertEqual({name: options[name] for name in expected}, expected)
            self.assertEqual(options["readback"], "none")
            if prepare_only:
                runner.assert_not_called()
                evaluator.assert_not_called()
            else:
                self.assertEqual(
                    runner.call_args.args[1],
                    [
                        Path(main.__globals__["ROOT"]) / ".cache/np101/opt-350m",
                        output / "request",
                        output / "device",
                    ],
                )
                self.assertEqual(evaluator.call_args.kwargs["repeats"], 3)
                self.assertTrue((output / "result.json").is_file())


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
        # A complete scan can find corruption and still cover every retained block,
        # including blocks after the first error and after a final allocation refusal.
        scanned = broken | {
            "uploaded_bytes": 24 * MIB,
            "verified_bytes": 16 * MIB,
            "retained_tensors": 3,
            "scanned_bytes": 24 * MIB,
            "scanned_blocks": 3,
            "size_mismatch_blocks": 0,
            "mismatched_bytes": 24,
            "mismatched_blocks": 1,
            "mismatched_ranges": 3,
            "rejected_block_bytes": 8 * MIB,
            "rejection_phase": "allocate",
        }
        completed = result(scanned, evidence | {"returncode": 1}, "all")
        self.assertTrue(completed["full_scan_completed"])
        self.assertTrue(completed["next_block_rejected"])
        self.assertFalse(completed["target_reached"])
        self.assertFalse(completed["readback_and_release_passed"])
        self.assertEqual(completed["matching_bytes"], 24 * MIB - 24)
        for changed in (
            {"scanned_bytes": 16 * MIB},
            {"scanned_blocks": 2},
            {"size_mismatch_blocks": 1},
        ):
            with self.subTest(changed=changed):
                self.assertFalse(
                    result(scanned | changed, evidence | {"returncode": 1}, "all")[
                        "full_scan_completed"
                    ]
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

    def test_nonfinite_and_wrong_shape_cannot_pass(self):
        reference = np.ones(4)
        for actual in (np.zeros(4), np.ones(3), np.full(4, np.nan)):
            self.assertFalse(compare_arrays(actual, reference, 0.08, 0.02)["passed"])


if __name__ == "__main__":
    unittest.main()
