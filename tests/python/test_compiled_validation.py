"""Compiled acceptance must reject changed artifacts, bad state and inflated throughput."""

import json
import runpy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.data.checkpoint import sha256
from specferry.export.compiled import compiled_ports, public_port_order
from specferry.models.opt.compiled import DRIVER, PARTS, SDK_FILES, load_ffn_package
from specferry.models.opt.compiled_model import _validate_ports, expected_ports
from specferry.validation import compiled
from specferry.validation.generation_metrics import summarize


class CompiledValidationTests(unittest.TestCase):
    def test_compiled_prefill_accounts_for_fp32_block_commits_and_reset(self):
        from specferry.models.opt.graph_validation import _mapped_prefill_transfers

        metadata = dict(
            layers=24, heads=16, width=64, capacity=64, prefill_backend="compiled-attention"
        )
        for blocks, reset, expected in (
            (1, False, (931, 9447688, 880, 9437248)),
            (3, True, (1129, 47204888, 976, 22020160)),
            (1, True, (1027, 34613512, 880, 9437248)),
        ):
            actual = _mapped_prefill_transfers(
                metadata, dict(decode_steps=16, prefill_steps=blocks), 16, reset
            )
            self.assertEqual(
                tuple(actual[k] for k in ("uploads", "upload_bytes", "reads", "read_bytes")),
                expected,
            )

    def test_mapped_prefill_requires_tail_handoff_and_complete_reset_accounting(self):
        from specferry.models.opt.graph_validation import evaluate, evaluate_benchmark

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Exact-block input deliberately leaves all sixteen tokens to M1:
            # the prefill consumer has no head and must leave a final M1 token.
            expected = [
                (16, 31, 0, 31, 1612, 6107000, 1504, 6094912),
                (17, 17, 1, 16, 977, 25172228, 832, 4718656),
                (49, 19, 3, 16, 979, 31463948, 832, 7864384),
            ]
            metadata = dict(layers=24, heads=16, width=64, capacity=64, prefill_block=16)
            metadata["requests"] = [
                dict(prompt_length=row[0], consumed=row[0] + 15, tokens=[7] * 16)
                for row in expected
            ]
            options = dict(
                variant="compiled-fullgraph",
                decoder_group_size=1,
                prefill_head_policy="last-token",
                kv_transfer="mapped-slot",
            )
            native = options | dict(
                graph_layout="partitioned",
                history_mode="reset",
                prefill_block=16,
                cache_readback="none",
                diagnostic_reads=0,
                diagnostic_read_bytes=0,
                kv_reset_uploads_per_request=96,
                launches=67,
                prefill_steps=4,
                decode_steps=63,
                head_runs=63,
            )
            (root / "run.json").write_text(json.dumps(native))
            initialized = native | dict(launches=0, prefill_steps=0, decode_steps=0, head_runs=0)
            (root / "initialization.json").write_text(json.dumps(initialized))
            results = []
            for index, row in enumerate(expected):
                length, launches, prefill, decode, uploads, upload_bytes, reads, read_bytes = row
                result = dict(
                    tokens=[7] * 16,
                    consumed=length + 15,
                    launches=launches,
                    prefill_steps=prefill,
                    decode_steps=decode,
                    head_runs=decode,
                    prefill_block=16,
                    uploads=uploads,
                    upload_bytes=upload_bytes,
                    reads=reads,
                    read_bytes=read_bytes,
                )
                results.append(result)
                (root / f"measured.{index}.json").write_text(json.dumps(result))

            checks, _ = evaluate(root, root, metadata, 16, **options)
            self.assertTrue(
                all(checks.values()), [name for name, passed in checks.items() if not passed]
            )
            for field in (
                "uploads",
                "upload_bytes",
                "reads",
                "read_bytes",
                "launches",
                "head_runs",
            ):
                with self.subTest(field=field):
                    changed = results[1] | {field: results[1][field] - 1}
                    (root / "measured.1.json").write_text(json.dumps(changed))
                    checks, _ = evaluate(root, root, metadata, 16, **options)
                    self.assertFalse(all(checks.values()))
            (root / "measured.1.json").write_text(json.dumps(results[1]))
            for field, value in (("kv_reset_uploads_per_request", 48), ("history_mode", "retain")):
                with self.subTest(field=field):
                    (root / "run.json").write_text(json.dumps(native | {field: value}))
                    checks, _ = evaluate(root, root, metadata, 16, **options)
                    self.assertFalse(all(checks.values()))
            with self.assertRaisesRegex(ValueError, "last-token"):
                evaluate(
                    root, root, metadata, 16, **(options | {"prefill_head_policy": "every-token"})
                )
            with self.assertRaisesRegex(ValueError, "benchmark is not yet supported"):
                evaluate_benchmark(root, metadata, 16, warmups=0, repeats=1, **options)

    def test_compiled_delta_norm_rejects_narrowed_reduction(self):
        from specferry.models.qwen3_5.compiled import audit_mixer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "export_model").mkdir()
            (root / "export_model/vnn_network.h").write_text(
                "#define VNN_VERSION_MAJOR 1\n"
                "#define VNN_VERSION_MINOR 1\n"
                "#define VNN_VERSION_PATCH 35\n"
            )
            for dtype in ("FLOAT32", "FLOAT16"):
                with self.subTest(dtype=dtype):
                    (root / "export_model/vnn_network.c").write_text(
                        "/* @present_recurrent:out0 */\n"
                        "NEW_VIRTUAL_TENSOR(0, attr, VSI_NN_TYPE_FLOAT32);\n"
                        "/* @ReduceMean_input_norm_variance_1:out0 */\n"
                        f"NEW_VIRTUAL_TENSOR(1, attr, VSI_NN_TYPE_{dtype});\n"
                        "/* @Cast_input_normalized_2:out0 */\n"
                        "NEW_VIRTUAL_TENSOR(2, attr, VSI_NN_TYPE_FLOAT16);\n"
                    )
                    if dtype == "FLOAT32":
                        self.assertEqual(audit_mixer(root), [1, 1, 35])
                    else:
                        with self.assertRaisesRegex(ValueError, "narrowed required FP32"):
                            audit_mixer(root)

    def test_completed_python_numerical_failure_still_rejects_health_and_budget_faults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "device").mkdir()
            native = dict(
                status="executed",
                phase="complete",
                released=True,
                const_live_bytes=0,
                nonconst_live_bytes=0,
                const_peak_bytes=64,
                nonconst_peak_bytes=64,
            )
            result = dict(
                passed=False,
                compilation=dict(passed=True),
                checks={"request.0.tokens": False, "release_budget": True},
            )
            previous = dict(
                evidence=str(root),
                kernel_gate=dict(faults=[], owners=""),
                execution=dict(
                    returncode=0,
                    timeout=False,
                    device_recovery_required=False,
                    process_group_exited=True,
                    remaining_processes=[],
                ),
            )
            for problem in (None, "kernel", "live_budget", "compilation", "non_numerical"):
                with self.subTest(problem=problem):
                    execution_record = native | {"const_live_bytes": int(problem == "live_budget")}
                    comparison = dict(result)
                    marker = dict(previous)
                    if problem == "kernel":
                        marker["kernel_gate"] = dict(faults=["EDMA timeout"], owners="")
                    if problem == "compilation":
                        comparison["compilation"] = dict(passed=False)
                    if problem == "non_numerical":
                        comparison["checks"] = {"request.0.tokens": False, "release_budget": False}
                    (root / "device/execution.json").write_text(json.dumps(execution_record))
                    (root / "result.json").write_text(json.dumps(comparison))
                    if problem:
                        with self.assertRaisesRegex(ValueError, "healthy numerical failure"):
                            compiled.validate_numerical_failure(marker)
                    else:
                        compiled.validate_numerical_failure(marker)

    def test_qwen_default_entry_writes_the_accepted_native_policies(self):
        import torch

        from specferry.models.qwen3_5 import generation

        script = Path(__file__).resolve().parents[2] / "scripts/generate_qwen.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "deployment-manifest.json").write_text(
                json.dumps({"text_config": {"bos_token_id": 1, "eos_token_id": 2}})
            )
            with patch.object(
                sys, "argv", [str(script), "--compiled-mixers", str(root), "--output", str(root)]
            ):
                args = runpy.run_path(str(script))["parse_arguments"]()
            self.assertEqual(args.head_backend, "matmul")
            cli_options = {
                "embedding_lookup": args.embedding_lookup,
                "matmul_feed_forward_layer": args.matmul_feed_forward_layer,
                "token_readback": args.token_readback,
            }
            for name, options in (("api", {}), ("cli", cli_options)):
                with (
                    self.subTest(entry=name),
                    patch.object(generation, "verify_export"),
                    patch.object(generation.Components, "from_text_config", return_value=Mock()),
                    patch.object(generation, "load_tokenizer", return_value=Mock()),
                    patch.object(generation, "encode", return_value=("Hi", torch.tensor([[1, 2]]))),
                ):
                    request, _ = generation.prepare_request(
                        root, root, root / name, "Hi", 64, 1, 1, **options
                    )
                for key, value, filename in (
                    ("embedding_lookup", "gather-nd", "embedding-lookup.txt"),
                    ("matmul_feed_forward_layer", 9, "matmul-feed-forward-layer.txt"),
                    ("token_readback", "mapped", "token-readback.txt"),
                ):
                    self.assertEqual(request[key], value)
                    self.assertEqual((root / name / filename).read_text().strip(), str(value))

    def test_qwen_matmul_fallback_rejects_invalid_layers_before_loading_files(self):
        from specferry.models.qwen3_5.generation import prepare_request

        with patch("specferry.models.qwen3_5.generation.verify_export") as verify:
            for layer in (None, -1, 24, True, "9"):
                with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, "0..23"):
                    prepare_request(
                        None, None, None, "Hi", 64, 1, 1, matmul_feed_forward_layer=layer
                    )
            verify.assert_not_called()

    def test_qwen_token_transport_rejects_unknown_policy_before_loading_files(self):
        from specferry.models.qwen3_5.generation import prepare_request

        with patch("specferry.models.qwen3_5.generation.verify_export") as verify:
            for transport in ("", "automatic", True):
                with (
                    self.subTest(transport=transport),
                    self.assertRaisesRegex(ValueError, "readback transport"),
                ):
                    prepare_request(None, None, None, "Hi", 64, 1, 1, token_readback=transport)
            verify.assert_not_called()

    def test_board_power_cycle_allows_same_boot_and_retains_gate_on_delayed_fault(self):
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
                            board_power_cycled=True,
                            sdk_timing=True,
                            evaluate_result=lambda case, device: {"passed": True},
                        )
                        if preflight_faults:
                            with self.assertRaises((ValueError, RuntimeError)):
                                compiled.run_case(*arguments, **options)
                            run.assert_not_called()
                            self.assertEqual(json.loads(marker.read_text()), previous)
                        else:
                            report = compiled.run_case(*arguments, **options)
                            self.assertEqual(report["passed"], not delayed_faults)
                            self.assertEqual(marker.exists(), bool(delayed_faults))
                            self.assertEqual(
                                (root / "np101-recovery-epoch.json").exists(), not delayed_faults
                            )
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
                "if (!inference_with_nbg) {",
                "node[0]->output.tensors[0] = norm_tensor[1];",
                "node[0]->output.tensors[1] = norm_tensor[2];",
                "} else {",
                "node[0]->output.tensors[0] = norm_tensor[2];",
                "node[0]->output.tensors[1] = norm_tensor[1];",
                "}",
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
            with self.assertRaisesRegex(ValueError, "geometry"):
                load_ffn_package(root, root, capacity=128)
            manifest["supported_capacities"] = [64, 128]
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(len(load_ffn_package(root, root, capacity=128)["layers"]), 24)
            with self.assertRaisesRegex(ValueError, "geometry"):
                load_ffn_package(root, root, capacity=256)
            manifest["supported_capacities"] = [64, 128, 256]
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(len(load_ffn_package(root, root, capacity=256)["layers"]), 24)
            (root / "layer00.nb").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_ffn_package(root, root)
            (root / "layer00.nb").write_bytes(b"\0")
            layers[0]["constant_bytes"] = 0
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "constant bound"):
                load_ffn_package(root, root)

    def test_prefill_contract_checks_m1_binding_source_and_total_const_budget(self):
        from specferry.models.opt.compiled import validate_prefill_package

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, layers = {}, []
            for index in range(4, 24):
                name = f"prefill{index:02d}.nb"
                (root / name).write_bytes(bytes([index]))
                identities = {}
                for part in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
                    key = f"decoder.layers.{index}.{part}"
                    records[key] = dict(bytes=1, sha256="a" * 64)
                    identities[key] = "a" * 64
                layers.append(
                    dict(
                        layer=index,
                        path=name,
                        nbg_sha256=sha256(root / name),
                        constant_bytes=100,
                        weight_sha256=identities,
                    )
                )
            native = root / "compiled-prefill.txt"
            native.write_text("specferry-opt-prefill 1 m1 index deployment 16 20\n")
            prefill = dict(
                tokens=16,
                ordinary_layers=[0, 1, 2, 3],
                supported_capacities=[64],
                layers=layers,
                native_manifest_sha256=sha256(native),
            )
            manifest = dict(
                head={},
                native_manifest_sha256="m1",
                prefill=prefill,
                deployment_sha256={
                    "weights.index": "index",
                    "deployment-manifest.json": "deployment",
                },
            )
            manifest["head"] = dict(path="shared-head.nb")
            expected = 20 * 100 + 24 * 4 * 1024 * 1024 * 2
            self.assertEqual(validate_prefill_package(root, manifest, records), expected)
            manifest["prefill"]["supported_capacities"] = [64, 256]
            self.assertEqual(validate_prefill_package(root, manifest, records), expected)
            # Outer package validation already charges the complete NBG projection set.
            self.assertEqual(
                validate_prefill_package(root, manifest | {"projections": [{}]}, records),
                20 * 100,
            )
            # Every M16 artifact alone fits; the existing M1 FCL weights still count.
            for item in layers:
                item["constant_bytes"] = 42 * 1024 * 1024
            self.assertGreater(validate_prefill_package(root, manifest, records), 2**30)
            layers[0]["weight_sha256"][next(iter(layers[0]["weight_sha256"]))] = "b" * 64
            with self.assertRaisesRegex(ValueError, "identity/source"):
                validate_prefill_package(root, manifest, records)
            with self.assertRaisesRegex(ValueError, "one head"):
                validate_prefill_package(root, manifest | {"head": None}, records)
            changed = manifest | {"native_manifest_sha256": "another-m1"}
            with self.assertRaisesRegex(ValueError, "bind the M1"):
                validate_prefill_package(root, changed, records)

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
