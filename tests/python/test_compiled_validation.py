"""Compiled acceptance must reject changed artifacts, bad state and inflated throughput."""

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.data.checkpoint import sha256
from specferry.export.compiled import compiled_ports, public_port_order
from specferry.models.opt.compiled import DRIVER, PARTS, SDK_FILES, load_ffn_package
from specferry.models.opt.compiled_model import _validate_ports, expected_ports
from specferry.validation import compiled
from specferry.validation.generation_metrics import summarize


class CompiledValidationTests(unittest.TestCase):
    def test_qwen_matmul_fallback_rejects_invalid_layers_before_loading_files(self):
        from specferry.models.qwen3_5.generation import prepare_request

        with patch("specferry.models.qwen3_5.generation.verify_export") as verify:
            for layer in (-1, 24, True, "9"):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "0..23"):
                    prepare_request(
                        None, None, None, "Hi", 64, 1, 1, matmul_feed_forward_layer=layer
                    )
            verify.assert_not_called()

    def test_cold_restart_requires_new_clean_boot_and_retains_gate_on_delayed_fault(self):
        original_read_text = Path.read_text
        driver = Path("/sys/module/galcore/srcversion")

        def read_text(path, *args, **kwargs):
            return "pinned" if path == driver else original_read_text(path, *args, **kwargs)

        for boot, preflight_faults, delayed_faults in (
            ("old", [], []),
            ("new", ["NP hang"], []),
            ("new", [], ["NP hang"]),
            ("new", [], []),
        ):
            with self.subTest(boot=boot, preflight=preflight_faults, delayed=delayed_faults):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    (root / "docs").mkdir()
                    (root / "docs/np101-recovery-incidents.md").write_text("test ledger")
                    marker = root / "np101-recovery-required.json"
                    previous = {"boot_id": "old", "reason": "previous device fault"}
                    marker.write_text(json.dumps(previous))
                    clean = dict(boot_id=boot, faults=[], owners="")
                    output = root / "diagnostic"
                    with (
                        patch.object(Path, "read_text", read_text),
                        patch.object(compiled, "ROOT", root),
                        patch.object(compiled, "RECOVERY_ROOT", root),
                        patch.object(compiled, "device_lock", return_value=nullcontext()),
                        patch.object(
                            compiled,
                            "snapshot",
                            side_effect=[
                                clean | {"faults": preflight_faults},
                                clean,
                                clean | {"faults": delayed_faults},
                            ],
                        ),
                        patch.object(compiled, "run_device", return_value={}) as run,
                        patch.object(compiled, "clean_execution", return_value=True),
                        patch.object(compiled.time, "sleep") as sleep,
                    ):
                        arguments = (
                            dict(binary=sys.executable, arguments=[]),
                            dict(directory=str(root), sha256={}, module_srcversion="pinned"),
                            output,
                        )
                        options = dict(
                            cold_restarted=True,
                            sdk_timing=True,
                            evaluate_result=lambda case, device: {"passed": True},
                        )
                        if boot == "old" or preflight_faults:
                            with self.assertRaises((ValueError, RuntimeError)):
                                compiled.run_case(*arguments, **options)
                            run.assert_not_called()
                            self.assertEqual(json.loads(marker.read_text()), previous)
                        else:
                            report = compiled.run_case(*arguments, **options)
                            self.assertEqual(report["passed"], not delayed_faults)
                            self.assertEqual(marker.exists(), bool(delayed_faults))
                            self.assertEqual(
                                json.loads((output / "previous-recovery-marker.json").read_text()),
                                previous,
                            )
                            self.assertTrue(run.call_args.kwargs["sdk_timing"])
                            sleep.assert_called_once_with(45)

    def test_resaved_graph_ports_use_public_order_instead_of_original_kernel_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            wrapper = directory / "export_model_nbg_unify"
            wrapper.mkdir()
            definitions = []
            for index, (name, dtype, shape) in enumerate(
                [
                    ("hidden", "FLOAT16", [1024, 1]),
                    ("selected", "INT32", [1]),
                    ("logits", "FLOAT16", [50272, 1]),
                ]
            ):
                definitions.append(f"/* @{name}:out0 */ attr.dim_num = {len(shape)};")
                definitions.extend(f"attr.size[{i}] = {size};" for i, size in enumerate(shape))
                definitions.append(
                    f"NEW_NORM_TENSOR(norm_tensor[{index}], attr, VSI_NN_TYPE_{dtype})"
                )
            bindings = [
                "node[0]->input.tensors[0] = norm_tensor[0];",
                "graph->input.tensors[0] = norm_tensor[0];",
                "node[0]->output.tensors[0] = norm_tensor[1];",
                "node[0]->output.tensors[1] = norm_tensor[2];",
                "graph->output.tensors[0] = norm_tensor[2];",
                "graph->output.tensors[1] = norm_tensor[1];",
            ]
            (wrapper / "vnn_network.c").write_text("\n".join(definitions + bindings))
            source = SimpleNamespace(
                graph=SimpleNamespace(
                    input=[SimpleNamespace(name="hidden")],
                    output=[SimpleNamespace(name="logits"), SimpleNamespace(name="selected")],
                )
            )
            original = compiled_ports(directory, source)
            saved = compiled_ports(directory, source, saved_public_graph=True)
            self.assertEqual(original["output_order"], [1, 0])
            self.assertEqual(saved["output_order"], [0, 1])
            self.assertEqual([port["name"] for port in saved["outputs"]], ["logits", "selected"])
            self.assertEqual([port["dtype"] for port in saved["outputs"]], ["F16", "I32"])

    def test_sdk_and_public_output_orders_are_distinct_and_complete(self):
        # Regression for V/K/Q SDK outputs with Q/K/V public graph outputs.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            wrapper = directory / "export_model_nbg_unify"
            wrapper.mkdir()
            text = "\n".join(
                [
                    "node[0]->output.tensors[0] = norm_tensor[3];",
                    "node[0]->output.tensors[1] = norm_tensor[2];",
                    "node[0]->output.tensors[2] = norm_tensor[1];",
                    "graph->output.tensors[0] = norm_tensor[1];",
                    "graph->output.tensors[1] = norm_tensor[2];",
                    "graph->output.tensors[2] = norm_tensor[3];",
                ]
            )
            path = wrapper / "vnn_network.c"
            path.write_text(text)
            self.assertEqual(public_port_order(directory, "output"), [2, 1, 0])
            path.write_text(
                text.replace(
                    "graph->output.tensors[2] = norm_tensor[3]",
                    "graph->output.tensors[2] = norm_tensor[1]",
                )
            )
            with self.assertRaisesRegex(ValueError, "permutation"):
                public_port_order(directory, "output")

    def test_compiled_decoder_ports_cannot_swap_state_roles_or_precision(self):
        inputs, outputs = expected_ports("decoders", 128)
        entry = dict(
            inputs=[dict(name=n, dtype=d, shape=s) for n, d, s in inputs],
            outputs=[dict(name=n, dtype=d, shape=s) for n, d, s in reversed(outputs)],
            input_order=[0, 1, 2, 3, 4],
            output_order=[2, 1, 0],
        )
        _validate_ports(entry, "decoders", 128)
        entry["output_order"] = [2, 0, 1]
        with self.assertRaisesRegex(ValueError, "present_key"):
            _validate_ports(entry, "decoders", 128)
        entry["output_order"] = [2, 1, 0]
        entry["outputs"][0]["dtype"] = "F16"
        with self.assertRaisesRegex(ValueError, "present_value"):
            _validate_ports(entry, "decoders", 128)

    def test_portable_ffn_package_rejects_corruption_and_unvalidated_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {"name": f"decoder.layers.{index}.{part}", "sha256": "0" * 64, "bytes": 1}
                for index in range(24)
                for part in PARTS
            ]
            (root / "deployment-manifest.json").write_text(
                json.dumps(dict(repo_id="facebook/opt-350m", revision="fixed", tensors=records))
            )
            (root / "weights.index").write_text("pinned index")
            (root / "compiled-ffn.txt").write_text("pinned native contract")
            layers = []
            for index in range(24):
                path = root / f"layer{index:02d}.nb"
                path.write_bytes(bytes([index]))
                layers.append(
                    dict(
                        layer=index,
                        path=path.name,
                        nbg_sha256=sha256(path),
                        constant_bytes=100,
                        weight_sha256={
                            f"decoder.layers.{index}.{part}": "0" * 64 for part in PARTS
                        },
                    )
                )
            manifest = dict(
                schema_version=1,
                kind="opt-ffn-finish",
                repo_id="facebook/opt-350m",
                revision="fixed",
                shape=dict(capacity=64, tokens=1, hidden=1024, intermediate=4096, layers=24),
                graph_version=[1, 1, 34],
                deployment_sha256={
                    name: sha256(root / name)
                    for name in ("weights.index", "deployment-manifest.json")
                },
                native_manifest_sha256=sha256(root / "compiled-ffn.txt"),
                sdk=dict(module_srcversion=DRIVER, sha256={name: "0" * 64 for name in SDK_FILES}),
                layers=layers,
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(len(load_ffn_package(root, root)["layers"]), 24)
            with self.assertRaisesRegex(ValueError, "C64/M1"):
                load_ffn_package(root, root, capacity=128)
            manifest["supported_capacities"] = [64, 128]
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(len(load_ffn_package(root, root, capacity=128)["layers"]), 24)
            with self.assertRaisesRegex(ValueError, "geometry"):
                load_ffn_package(root, root, capacity=256)
            (root / "layer00.nb").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_ffn_package(root, root)
            (root / "layer00.nb").write_bytes(b"\0")
            layers[0]["constant_bytes"] = 0
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "constant bound"):
                load_ffn_package(root, root)

    def test_plan_pins_independent_reference_and_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reference.json").write_text(json.dumps({"requests": [{"tokens": [1]}]}))
            np.array([1], dtype="<f2").tofile(root / "expected.0.logits.bin")
            artifact = root / "network.nb"
            artifact.write_bytes(b"independent artifact")
            files = {str(path): sha256(path) for path in root.iterdir()}
            plan = dict(
                schema_version=1,
                sdk={},
                file_sets={"artifacts": files},
                cases={
                    "model": dict(
                        kind="fullgraph",
                        binary=sys.executable,
                        request=str(root),
                        repeats=1,
                        stop_reason="max_new_tokens",
                        payload_bound_bytes=dict(const=100, nonconst=100),
                        file_sets=["artifacts"],
                    )
                },
            )
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with (
                patch.object(compiled, "snapshot") as kernel,
                patch.object(compiled, "run_device") as device,
            ):
                compiled.load_case(path, "model")
                kernel.assert_not_called()
                device.assert_not_called()
            plan["cases"]["model"]["cache"] = dict(
                dtype="<f2",
                shape=[1, 1, 1],
                keys=str(root / "expected.0.logits.bin"),
                values=str(root / "expected.0.logits.bin"),
            )
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "dtype must be F16 or F32"):
                compiled.load_case(path, "model")
            del plan["cases"]["model"]["cache"]
            path.write_text(json.dumps(plan))
            artifact.write_bytes(b"changed artifact")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                compiled.load_case(path, "model")
            artifact.write_bytes(b"independent artifact")
            del files[str(root / "reference.json")]
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "references require pinned"):
                compiled.load_case(path, "model")

    def test_generation_rate_excludes_warmup_and_keeps_actual_output_counts(self):
        result = compiled.summarize(
            [
                dict(
                    tokens=[1] * 100, total_seconds=100, first_token_seconds=99, token_seconds=[1]
                ),
                dict(
                    tokens=[1, 2, 3],
                    total_seconds=1.2,
                    first_token_seconds=1,
                    token_seconds=[0.1, 0.1],
                ),
                dict(tokens=[1, 2], total_seconds=1.1, first_token_seconds=1, token_seconds=[0.1]),
            ]
        )
        self.assertEqual(result["generated_tokens"], 5)
        self.assertAlmostEqual(result["output_tokens_per_second"], 5 / 2.3)
        self.assertAlmostEqual(result["decode_tokens_per_second"], 10)
        self.assertAlmostEqual(result["decode_ms"], 100)
        metrics = summarize(
            [
                dict(
                    tokens=[1, 2, 3],
                    total_seconds=1.2,
                    first_token_seconds=1,
                    prefill_seconds=0.9,
                    token_seconds=[0.1, 0.1],
                )
            ]
        )
        self.assertEqual(metrics["generated_tokens"], 3)
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 2.5)
        self.assertAlmostEqual(metrics["decode_tokens_per_second"], 10)

    def test_tokens_alone_cannot_accept_bad_warmup_cache_or_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device = root / "device"
            device.mkdir()
            generation = dict(
                tokens=[1, 2],
                consumed=2,
                stop_reason="max_new_tokens",
                total_seconds=2,
                first_token_seconds=1,
                token_seconds=[1],
            )
            case = dict(
                kind="fullgraph",
                repeats=1,
                request=root,
                reference=generation,
                stop_reason="max_new_tokens",
                cache=dict(
                    dtype="F32",
                    shape=[1, 4, 2],
                    keys=str(root / "keys.bin"),
                    values=str(root / "values.bin"),
                ),
            )
            for name in ("warmup", "measured"):
                (device / (name + ".json")).write_text(json.dumps(generation))
            logits = np.array([1, 2], dtype="<f2")
            logits.tofile(root / "expected.0.logits.bin")
            logits.tofile(device / "logits.bin")
            cache = np.zeros((1, 4, 2), dtype="<f4")
            cache[:, :2] = 1
            for name in ("key", "value"):
                cache.tofile(root / ("keys.bin" if name == "key" else "values.bin"))
                cache.tofile(device / f"layer23.present_{name}.bin")
            native = dict(
                released=True,
                const_live_bytes=0,
                nonconst_live_bytes=0,
                const_peak_bytes=100,
                nonconst_peak_bytes=100,
            )
            (device / "execution.json").write_text(json.dumps(native))
            self.assertTrue(compiled.evaluate(case, device)["passed"])
            cache[:, 3] = -0.0
            cache.tofile(device / "layer23.present_key.bin")
            self.assertFalse(compiled.evaluate(case, device)["passed"])
            cache[:, 3] = 0.0
            cache.tofile(device / "layer23.present_key.bin")
            (device / "warmup.json").write_text(json.dumps(generation | {"tokens": [9]}))
            self.assertIsNone(compiled.evaluate(case, device)["performance"])
            (device / "warmup.json").write_text(json.dumps(generation))
            (device / "execution.json").write_text(json.dumps(native | {"nonconst_live_bytes": 1}))
            self.assertFalse(compiled.evaluate(case, device)["passed"])
