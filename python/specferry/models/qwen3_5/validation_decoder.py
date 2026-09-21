"""Official four-layer decoder reference and independent trajectory acceptance."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as reference

from specferry.models.qwen3_5.components import BASELINE, Components
from specferry.models.qwen3_5.export import verify_export
from specferry.models.qwen3_5.precision import TOLERANCES, precision
from specferry.validation.arrays import compare_file, save_tensor
from specferry.validation.capabilities import compare_arrays
from specferry.validation.device import fingerprint, write_json

from .model import source_record
from .validation_attention import ReferenceCache, checkpoint, compare_prefix


class DecoderCache:
    """Official per-layer cache protocol with one sequence-wide committed position."""

    def __init__(self, config=BASELINE, selected=(0, 1, 2, 3)):
        self.config = config
        self.length = 0
        self.layers = {
            index: SimpleNamespace(
                recurrent_states=[torch.zeros(config.recurrent_shape, dtype=torch.float32)],
                conv_states=[torch.zeros(config.convolution_shape, dtype=torch.float16)],
                record_past=False,
            )
            for index in selected
            if config.layer_types[index] == "delta"
        }
        self.attention = {
            index: ReferenceCache(config, index)
            for index in selected
            if config.layer_types[index] == "attention"
        }

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_recurrent_state(self, state, layer_idx):
        self.layers[layer_idx].recurrent_states[0] = state

    def update(self, key, value, layer_idx):
        cache = self.attention[layer_idx]
        cache.length = self.length
        return cache.update(key, value, layer_idx)

    @torch.inference_mode()
    def reset(self):
        self.length = 0
        for layer in self.layers.values():
            layer.recurrent_states[0].zero_()
            layer.conv_states[0].zero_()
        for cache in self.attention.values():
            cache.length = 0


def captured_inputs(trace: Path, steps: int, first_layer: int, config=BASELINE):
    if not 2 <= steps <= config.capacity or not 0 <= first_layer < len(config.layer_types):
        raise ValueError("decoder steps or first layer exceed configured bounds")
    suffix = f".layer.{first_layer}.input.0"
    with np.load(trace, allow_pickle=False) as arrays:
        names = sorted(name for name in arrays.files if name.endswith(suffix))
        values = [arrays[name].copy() for name in names]
    if not values or any(
        value.shape != (1, 1, config.hidden)
        or value.dtype != np.float16
        or not np.isfinite(value).all()
        for value in values
    ):
        raise ValueError("trace must contain finite pre-normalization FP16 decoder inputs")
    indices = [index % len(values) for index in range(steps)]
    return [torch.from_numpy(values[i].copy()) for i in indices], [names[i] for i in indices]


def load_layers(model: Path, manifest: dict, selected):
    config = Qwen3_5TextConfig(**manifest["text_config"])
    config._attn_implementation = "eager"
    layers = {}
    with (model / "weights.bin").open("rb") as stream:
        for index in selected:
            prefix = f"model.layers.{index}."
            with torch.device("meta"):
                layer = reference.Qwen3_5DecoderLayer(config, index)
            state = {}
            for record in manifest["tensors"]:
                if not record["name"].startswith(prefix):
                    continue
                stream.seek(record["offset"])
                raw = stream.read(record["bytes"])
                if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                    raise ValueError(f"invalid decoder weight: {record['name']}")
                dtype = {"F16": "<f2", "F32": "<f4"}[record["dtype"]]
                value = np.frombuffer(raw, dtype=dtype).copy().reshape(record["shape"])
                state[record["name"].removeprefix(prefix)] = torch.from_numpy(value)
            layer.load_state_dict(state, strict=True, assign=True)
            layers[index] = layer.eval().requires_grad_(False)
    return layers, reference.Qwen3_5TextRotaryEmbedding(config)


def reference_sequence(
    layers, rotary, cache, inputs, destination, sequence, trace, final_only=False
):
    captured, handles, observations, alignment = {}, [], [], {}

    def output_hook(layer, name):
        def save(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[f"layer.{layer}.{name}"] = value.detach().clone()

        return save

    def residual_hook(layer):
        def save(module, args):
            captured[f"layer.{layer}.residual"] = args[0].detach().clone()

        return save

    for index, layer in layers.items():
        mixer = (
            layer.self_attn if cache.config.layer_types[index] == "attention" else layer.linear_attn
        )
        for name, module in (
            ("normalized", layer.input_layernorm),
            ("mixer", mixer),
            ("post_norm", layer.post_attention_layernorm),
            ("mlp", layer.mlp),
            ("output", layer),
        ):
            handles.append(module.register_forward_hook(output_hook(index, name)))
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(residual_hook(index))
        )
    try:
        with torch.inference_mode(), precision("deployment-fp16"), np.load(trace) as saved:
            for index, hidden in enumerate(inputs):
                captured.clear()
                position = torch.tensor([[cache.length]])
                embedding = rotary(hidden, position)
                for layer in layers.values():
                    hidden = layer(
                        hidden,
                        position_embeddings=embedding,
                        position_ids=position,
                        past_key_values=cache,
                        attention_mask=None,
                    )
                cache.length += 1
                # Independently anchor the CPU slice to the earlier full-model trace.
                if sequence == "zero" and index < 8:
                    for layer in (0, 3):
                        key = f"token.{index + 1:04d}.layer.{layer}.output"
                        if layer in layers and key in saved:
                            alignment[key] = compare_arrays(
                                captured[f"layer.{layer}.output"].numpy(),
                                saved[key],
                                **TOLERANCES["fp16_pipeline"],
                            )
                if not (index + 1 == len(inputs) if final_only else checkpoint(index, len(inputs))):
                    continue
                for layer in layers:
                    if layer in cache.layers:
                        state = cache.layers[layer]
                        captured[f"layer.{layer}.recurrent"] = state.recurrent_states[0]
                        captured[f"layer.{layer}.convolution"] = state.conv_states[0]
                for layer, attention in cache.attention.items():
                    captured[f"layer.{layer}.keys"] = attention.keys
                    captured[f"layer.{layer}.values"] = attention.values
                specs = {
                    name: save_tensor(destination / f"{sequence}.{name}.{index}.bin", value)
                    for name, value in captured.items()
                }
                observations.append({"sequence": sequence, "index": index, "outputs": specs})
    finally:
        for handle in handles:
            handle.remove()
    return observations, alignment


def prepare(
    model: Path, trace: Path, destination: Path, steps: int, first_layer: int, selected=None
):
    verify_export(model)
    manifest = json.loads((model / "deployment-manifest.json").read_text())
    config = Components.from_text_config(manifest["text_config"])
    selected = list(range(first_layer, 4)) if selected is None else list(selected)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or selected[0] != first_layer
        or any(
            type(layer) is not int or not 0 <= layer < len(config.layer_types) for layer in selected
        )
    ):
        raise ValueError("invalid decoder layer selection")
    inputs, input_names = captured_inputs(trace, steps, first_layer, config)
    layers, rotary = load_layers(model, manifest, selected)
    destination.mkdir(parents=True, exist_ok=False)
    component_config = config.write(destination)
    (destination / "layers.txt").write_text(" ".join(map(str, selected)) + "\n")
    for index, hidden in enumerate(inputs):
        save_tensor(destination / f"input.{index}.bin", hidden)
    cache = DecoderCache(config, selected)
    observations, alignment = reference_sequence(
        layers, rotary, cache, inputs, destination, "zero", trace
    )
    if not alignment or not all(result["passed"] for result in alignment.values()):
        raise ValueError("decoder CPU slice disagrees with the saved full-model trace")
    for name, count, final_only in (
        ("reset", min(8, steps), False),
        ("final", min(32, steps), True),
    ):
        cache.reset()
        values, _ = reference_sequence(
            layers, rotary, cache, inputs[:count], destination, name, trace, final_only
        )
        observations += values
    values, _ = reference_sequence(
        layers, rotary, DecoderCache(config, selected), inputs[:8], destination, "fresh", trace
    )
    observations += values
    selected_weights = [
        record
        for record in manifest["tensors"]
        if any(record["name"].startswith(f"model.layers.{i}.") for i in layers)
    ]
    weight_bytes = sum(record["bytes"] for record in selected_weights)
    duplicate_bytes = sum(
        record["bytes"] for record in selected_weights if ".linear_attn." in record["name"]
    )
    metadata = {
        "steps": steps,
        "first_layer": first_layer,
        "layers": selected,
        "component_config": component_config,
        "transfers": config.transfers(selected),
        "observations": observations,
        "reference": source_record(),
        "trace_alignment": alignment,
        "trace_sha256": fingerprint(trace),
        "input_names": input_names,
        "input_policy": "cycle saved pre-normalization single-token inputs; not text generation",
        "manifest_sha256": fingerprint(model / "deployment-manifest.json"),
        "tolerances": TOLERANCES,
        "memory_payload": {
            "unique_weight_bytes": weight_bytes,
            "duplicate_delta_weight_bytes": duplicate_bytes,
            "loaded_weight_bytes": weight_bytes + duplicate_bytes,
            "state_bytes": config.state_bytes(selected),
            "note": "excludes activations, constants, layouts and SDK workspace; not board measurements",
        },
        "fixture_sha256": {
            path.name: fingerprint(path) for path in sorted(destination.glob("*.bin"))
        },
    }
    write_json(destination / "reference.json", metadata)
    return metadata


def evaluate(fixture: Path, actual: Path, metadata: dict, evidence: dict):
    config = (
        Components(**metadata["component_config"]) if "component_config" in metadata else BASELINE
    )
    transfers = metadata.get("transfers", config.transfers(metadata["layers"]))
    checks = {}
    for observation in metadata["observations"]:
        sequence, index = observation["sequence"], observation["index"]
        for field, spec in observation["outputs"].items():
            key = f"{sequence}.{field}.{index}"
            tolerance = metadata["tolerances"][
                "fp32_state_from_fp16_pipeline" if field.endswith(".recurrent") else "fp16_pipeline"
            ]
            checks[key] = compare_file(
                actual / f"{key}.bin", fixture / f"{key}.bin", spec, tolerance
            )
            if sequence != "zero":
                current, initial = actual / f"{key}.bin", actual / f"zero.{field}.{index}.bin"
                checks[f"repeat.{key}"] = (
                    compare_prefix(current, initial, spec, index + 1)
                    if field.endswith((".keys", ".values"))
                    else compare_file(current, initial, spec, {"atol": 0, "rtol": 0})
                )
    path = actual / "execution.json"
    execution = json.loads(path.read_text()) if path.is_file() else {}
    steps = metadata["steps"]
    total = steps + min(32, steps) + 2 * min(8, steps)
    lifecycle = (
        evidence.get("returncode") == 0
        and evidence.get("process_group_exited") is True
        and evidence.get("device_recovery_required") is False
        and execution.get("status") == "executed"
        and execution.get("phase") == "complete"
        and execution.get("completed_steps") == total
        and execution.get("completed_sequences") == 4
        and execution.get("cache_writes") == total * transfers["attention_layers"]
        and execution.get("step_uploads") == total * transfers["uploads"]
        and execution.get("step_upload_bytes") == total * transfers["upload_bytes"]
        and execution.get("step_reads") == 0
        and execution.get("invalid_input_rejected") is True
        and execution.get("empty_output_rejected") is True
        and (steps != config.capacity or execution.get("capacity_rejected") is True)
        and execution.get("released") is True
    )
    checks["lifecycle"] = {"passed": lifecycle}
    log_path = actual / "sdk.log"
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    return {
        "status": "numerical_pass"
        if metadata["observations"] and all(check["passed"] for check in checks.values())
        else "failed",
        "checks": checks,
        "execution": execution,
        "evidence": evidence,
        "hardware_execution_proven": False,
        "device_residency_verified": False,
        "sdk_matrix_node_creation_warnings": log.count("Call vxBatchGemmNode fail"),
        "timing_scope": "host wall time; initialization/reset/readback separate; "
        + (
            "driver tracing enabled" if evidence.get("driver_traced") else "driver tracing disabled"
        ),
        "memory_payload": metadata["memory_payload"],
        "transfer_contract": (
            f"one {transfers['hidden_bytes']}-byte boundary upload; "
            f"{8 * transfers['attention_layers']}-byte Attention control uploads per step; "
            "diagnostic reads only; no application intermediate/state roundtrip"
        ),
    }
