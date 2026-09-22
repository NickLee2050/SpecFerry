"""Independent official OPT decoder trajectories and NP101 acceptance checks."""

import json
from dataclasses import asdict, replace

import numpy as np
import torch
from transformers import DynamicCache, OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from specferry.validation.arrays import compare_file, save_tensor
from specferry.validation.capabilities import compare_arrays
from specferry.validation.device import fingerprint, write_json

from .config import TOLERANCES, validate_config
from .export import read_layer_weights, verify_export
from .reference import capture_layers, source_record

OUTPUTS = (
    "query",
    "key",
    "value",
    "probabilities",
    "mixer",
    "attention_residual",
    "attention_norm",
    "fc1",
    "activation",
    "mlp",
    "ffn_residual",
    "output",
    "keys",
    "values",
)


def checkpoint(step, count):
    return step in (0, 1, 3, 7, 31, 255, 511, count - 1)


def load_inputs(trace, config, first, steps, prompt):
    with np.load(trace, allow_pickle=False) as archive:
        prefix = f"prompt.{prompt}.token."
        suffix = f".layer.{first}.input"
        names = sorted(
            (key for key in archive.files if key.startswith(prefix) and key.endswith(suffix)),
            key=lambda key: int(key.split(".")[3]),
        )
        values = [archive[key].copy() for key in names]
    if not values or any(
        value.shape != (1, 1, config.hidden)
        or value.dtype != np.float16
        or not np.isfinite(value).all()
        for value in values
    ):
        raise ValueError("trace lacks finite FP16 OPT layer-boundary inputs")
    return [torch.from_numpy(values[step % len(values)].copy()) for step in range(steps)], names


@torch.inference_mode()
def reference_sequence(
    layers, official_config, config, inputs, directory, sequence, final_only=False
):
    cache = DynamicCache(config=official_config)
    observations = []
    for index, hidden in enumerate(inputs):
        captured = {}
        with capture_layers(layers, captured):
            for layer in layers.values():
                hidden = layer(hidden, past_key_values=cache, use_cache=True)
        if (
            final_only
            and index + 1 != len(inputs)
            or not final_only
            and not checkpoint(index, len(inputs))
        ):
            continue
        tensors = {}
        for layer in layers:
            prefix = f"layer.{layer}."
            for short, name in (("q", "query"), ("k", "key"), ("v", "value")):
                value = captured[prefix + short]
                if short == "q":
                    value = value * config.head_dim**-0.5
                tensors[prefix + name] = value.reshape(
                    1, 1, config.heads, config.head_dim
                ).transpose(1, 2)
            for name in OUTPUTS:
                if name in ("query", "key", "value"):
                    continue
                if name in ("keys", "values"):
                    value = getattr(cache.layers[layer], name)
                    value = torch.nn.functional.pad(value, (0, 0, 0, config.capacity - index - 1))
                else:
                    value = captured[prefix + name]
                    if name == "probabilities":
                        value = torch.nn.functional.pad(value, (0, config.capacity - index - 1))
                tensors[prefix + name] = value
        specs = {
            name: save_tensor(directory / f"{sequence}.{name}.{index}.bin", value)
            for name, value in tensors.items()
        }
        observations.append(
            {"sequence": sequence, "index": index, "length": index + 1, "outputs": specs}
        )
    return observations


def prepare(model, trace, directory, selected=(0,), steps=8, capacity=512, prompt=0):
    verify_export(model)
    manifest = json.loads((model / "deployment-manifest.json").read_text())
    config = replace(validate_config(manifest["config"]), capacity=capacity)
    selected = config.select(selected)
    if len(selected) > 4 or selected != tuple(range(selected[0], selected[0] + len(selected))):
        raise ValueError("this validation covers contiguous slices of one to four layers")
    if not 2 <= steps <= capacity:
        raise ValueError("steps must be between two and the configured capacity")
    reference_report = json.loads((trace.parent / "reference.json").read_text())
    if reference_report.get("status") != "passed" or reference_report[
        "trace_sha256"
    ] != fingerprint(trace):
        raise ValueError("a passed, matching OPT CPU reference trace is required")
    inputs, input_names = load_inputs(trace, config, selected[0], steps, prompt)
    official = OPTConfig(**manifest["config"])
    official._attn_implementation = "eager"
    layers = {}
    for index in selected:
        with torch.device("meta"):
            layer = OPTDecoderLayer(official, layer_idx=index)
        layer.load_state_dict(
            read_layer_weights(model, manifest["tensors"], index), strict=True, assign=True
        )
        layers[index] = layer.eval().requires_grad_(False)
    directory.mkdir(parents=True, exist_ok=False)
    config.write(directory / "components.txt")
    (directory / "layers.txt").write_text(" ".join(map(str, selected)) + "\n")
    for index, hidden in enumerate(inputs):
        save_tensor(directory / f"input.{index}.bin", hidden)
    observations = []
    for name, count, final_only in (
        ("zero", steps, False),
        ("reset", min(8, steps), False),
        ("final", min(32, steps), True),
        ("fresh", min(8, steps), False),
    ):
        observations += reference_sequence(
            layers, official, config, inputs[:count], directory, name, final_only
        )
    alignment = {}
    with np.load(trace, allow_pickle=False) as saved:
        for item in observations:
            if item["sequence"] != "zero" or item["index"] >= len(input_names):
                continue
            for layer in selected:
                key = f"prompt.{prompt}.token.{item['index']}.layer.{layer}.output"
                actual = np.fromfile(
                    directory / f"zero.layer.{layer}.output.{item['index']}.bin", dtype="<f2"
                ).reshape(1, 1, config.hidden)
                alignment[key] = compare_arrays(actual, saved[key], **TOLERANCES["cpu"])
    if not alignment or not all(check["passed"] for check in alignment.values()):
        raise ValueError("OPT CPU slice disagrees with the full official model trace")
    weight_bytes = sum(
        entry["bytes"]
        for entry in manifest["tensors"]
        if any(entry["name"].startswith(f"decoder.layers.{layer}.") for layer in selected)
    )
    metadata = {
        "steps": steps,
        "layers": list(selected),
        "config": asdict(config),
        "prompt": prompt,
        "observations": observations,
        "trace_alignment": alignment,
        "reference": source_record(),
        "tolerances": TOLERANCES,
        "input_names": input_names,
        "input_policy": "cycle captured layer-boundary vectors; not text generation",
        "trace_sha256": fingerprint(trace),
        "manifest_sha256": fingerprint(model / "deployment-manifest.json"),
        "weight_bytes": weight_bytes,
        "state_bytes": config.state_bytes(selected),
        "fixture_sha256": {path.name: fingerprint(path) for path in sorted(directory.iterdir())},
    }
    write_json(directory / "reference.json", metadata)
    return metadata


def compare_prefix(actual, expected, spec, length, tolerance):
    if any(
        not path.is_file() or path.stat().st_size != spec["bytes"] for path in (actual, expected)
    ):
        return {"passed": False, "reason": "missing or wrong-size KV file"}
    arrays = [
        np.fromfile(path, dtype=spec["dtype"]).reshape(spec["shape"])[:, :, :length]
        for path in (actual, expected)
    ]
    return compare_arrays(*arrays, **tolerance)


def evaluate(fixture, actual, metadata, evidence):
    checks = {}
    for item in metadata["observations"]:
        sequence, index = item["sequence"], item["index"]
        for field, spec in item["outputs"].items():
            key = f"{sequence}.{field}.{index}"
            current, expected = actual / f"{key}.bin", fixture / f"{key}.bin"
            is_cache = field.endswith((".keys", ".values"))

            def comparator(a, b, policy):
                if is_cache:
                    return compare_prefix(a, b, spec, item["length"], policy)
                return compare_file(a, b, spec, policy)

            checks[key] = comparator(current, expected, metadata["tolerances"]["device"])
            if sequence != "zero":
                checks[f"repeat.{key}"] = comparator(
                    current, actual / f"zero.{field}.{index}.bin", {"atol": 0, "rtol": 0}
                )
            if field.endswith(".probabilities"):
                valid = current.is_file() and current.stat().st_size == spec["bytes"]
                checks[f"mask.{key}"] = {
                    "passed": bool(
                        valid
                        and np.all(
                            np.fromfile(current, dtype=spec["dtype"]).reshape(spec["shape"])[
                                ..., item["length"] :
                            ]
                            == 0
                        )
                    )
                }
    path = actual / "execution.json"
    execution = json.loads(path.read_text()) if path.is_file() else {}
    total = metadata["steps"] + min(32, metadata["steps"]) + 2 * min(8, metadata["steps"])
    layers = len(metadata["layers"])
    lifecycle = (
        evidence.get("returncode") == 0
        and evidence.get("process_group_exited") is True
        and evidence.get("device_recovery_required") is False
        and execution.get("status") == "executed"
        and execution.get("phase") == "complete"
        and execution.get("released") is True
        and execution.get("steps") == total
        and execution.get("sequences") == 4
        and execution.get("cache_writes") == total * layers
        and execution.get("uploads") == total * (1 + layers)
        and execution.get("upload_bytes") == total * (metadata["config"]["hidden"] * 2 + layers * 4)
        and execution.get("reads") == 0
        and execution.get("invalid_input_rejected") is True
        and execution.get("empty_output_rejected") is True
        and (
            metadata["steps"] != metadata["config"]["capacity"]
            or execution.get("capacity_rejected") is True
        )
    )
    checks["lifecycle"] = {"passed": lifecycle}
    return {
        "status": "numerical_pass"
        if checks and all(item["passed"] for item in checks.values())
        else "failed",
        "checks": checks,
        "execution": execution,
        "evidence": evidence,
        "hardware_execution_proven": False,
        "device_residency_verified": False,
        "weight_bytes": metadata["weight_bytes"],
        "state_bytes": metadata["state_bytes"],
    }
