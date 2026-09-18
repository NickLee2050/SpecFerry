"""Official single-layer Attention reference and single-buffer cache acceptance."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as reference

from specferry.export.weights import verify_export
from specferry.reference.model import source_record
from specferry.reference.runner import TOLERANCES, precision

from .capabilities import compare_arrays
from .delta_net import compare_file, save_tensor
from .device import fingerprint, write_json

CAPACITY = 512
PREFIX = "model.layers.3.self_attn."
OUTPUTS = ("output", "query", "key", "value", "keys", "values", "probabilities")


class ReferenceCache:
    """Keep stale slots so reset/truncation checks also exercise the valid-prefix mask."""

    def __init__(self):
        self.keys = torch.zeros((1, 2, CAPACITY, 256), dtype=torch.float16)
        self.values = torch.zeros_like(self.keys)
        self.length = 0

    def update(self, key, value, layer_idx):
        if layer_idx != 3 or not 0 <= self.length < CAPACITY:
            raise ValueError("invalid layer or full reference cache")
        self.keys[:, :, self.length : self.length + 1] = key
        self.values[:, :, self.length : self.length + 1] = value
        self.length += 1
        return self.keys[:, :, : self.length], self.values[:, :, : self.length]


def load_mixer(model: Path, manifest: dict):
    config = Qwen3_5TextConfig(**manifest["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        mixer = reference.Qwen3_5Attention(config, 3)
    state = {}
    with (model / "weights.bin").open("rb") as file:
        for record in manifest["tensors"]:
            if not record["name"].startswith(PREFIX):
                continue
            file.seek(record["offset"])
            raw = file.read(record["bytes"])
            if record["dtype"] != "F16" or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"invalid Attention weight: {record['name']}")
            value = np.frombuffer(raw, dtype="<f2").copy().reshape(record["shape"])
            state[record["name"].removeprefix(PREFIX)] = torch.from_numpy(value)
    mixer.load_state_dict(state, strict=True, assign=True)
    return mixer.eval().requires_grad_(False), reference.Qwen3_5TextRotaryEmbedding(config)


def captured_inputs(trace: Path, steps: int):
    if not 2 <= steps <= CAPACITY:
        raise ValueError("Attention validation requires 2-512 steps")
    with np.load(trace, allow_pickle=False) as arrays:
        names = sorted(
            name for name in arrays.files if name.endswith(".layer.3.self_attn.q_proj.input.0")
        )
        values = [arrays[name].copy() for name in names]
    if not values or any(
        x.shape != (1, 1, 1024) or x.dtype != np.float16 or not np.isfinite(x).all() for x in values
    ):
        raise ValueError("trace must contain finite single-token layer-3 FP16 hidden vectors")
    indices = [index % len(values) for index in range(steps)]
    return [torch.from_numpy(values[i].copy()) for i in indices], [names[i] for i in indices]


def checkpoint(index: int, count: int):
    return index in (0, 1, 3, 7, 31, 255, 511, count - 1)


def reference_sequence(mixer, rotary, cache, inputs, destination, name, final_only=False):
    captured = {}
    original = reference.eager_attention_forward
    observations = []

    def attention(module, query, key, value, *args, **kwargs):
        result, probabilities = original(module, query, key, value, *args, **kwargs)
        captured.update(
            query=query,
            key=key[:, :, -1:],
            value=value[:, :, -1:],
            probabilities=torch.nn.functional.pad(probabilities, (0, CAPACITY - cache.length)),
        )
        return result, probabilities

    with (
        torch.inference_mode(),
        precision("deployment-fp16"),
        patch.object(reference, "eager_attention_forward", attention),
    ):
        for index, hidden in enumerate(inputs):
            position = cache.length
            captured.clear()
            output, _ = mixer(
                hidden,
                position_embeddings=rotary(hidden, torch.tensor([[position]])),
                attention_mask=None,
                past_key_values=cache,
            )
            should_capture = (
                index + 1 == len(inputs) if final_only else checkpoint(index, len(inputs))
            )
            if not should_capture:
                continue
            captured.update(output=output, keys=cache.keys, values=cache.values)
            specs = {
                field: save_tensor(destination / f"{name}.{field}.{index}.bin", captured[field])
                for field in OUTPUTS
            }
            observations.append(
                {"sequence": name, "index": index, "length": cache.length, "outputs": specs}
            )
    return observations


def prepare(model: Path, trace: Path, destination: Path, steps: int):
    verify_export(model)
    manifest = json.loads((model / "deployment-manifest.json").read_text())
    inputs, input_names = captured_inputs(trace, steps)
    mixer, rotary = load_mixer(model, manifest)
    destination.mkdir(parents=True, exist_ok=False)
    for index, value in enumerate(inputs):
        save_tensor(destination / f"input.{index}.bin", value)
    cache = ReferenceCache()
    observations = reference_sequence(mixer, rotary, cache, inputs, destination, "zero")
    cache.length = min(3, steps - 1)
    observations += reference_sequence(
        mixer, rotary, cache, [inputs[0], inputs[-1]], destination, "branch"
    )
    cache.length = 0
    observations += reference_sequence(mixer, rotary, cache, inputs[:8], destination, "reset")
    cache.length = 0
    observations += reference_sequence(mixer, rotary, cache, inputs, destination, "final", True)
    observations += reference_sequence(
        mixer, rotary, ReferenceCache(), inputs[:8], destination, "fresh"
    )
    metadata = {
        "steps": steps,
        "observations": observations,
        "reference": source_record(),
        "trace_sha256": fingerprint(trace),
        "input_names": input_names,
        "input_policy": "cycle captured layer-3 single-token FP16 inputs; not text generation",
        "manifest_sha256": fingerprint(model / "deployment-manifest.json"),
        "tolerances": TOLERANCES,
        "fixture_sha256": {
            path.name: fingerprint(path) for path in sorted(destination.glob("*.bin"))
        },
    }
    write_json(destination / "reference.json", metadata)
    return metadata


def compare_prefix(left: Path, right: Path, spec: dict, length: int):
    """Reset preserves stale suffixes, so exact repeat checks use only the valid prefix."""
    if any(not path.is_file() or path.stat().st_size != spec["bytes"] for path in (left, right)):
        return {"passed": False, "reason": "missing or wrong-size cache snapshot"}
    arrays = [
        np.fromfile(path, dtype=spec["dtype"]).reshape(spec["shape"]) for path in (left, right)
    ]
    return compare_arrays(*(array[:, :, :length] for array in arrays), atol=0, rtol=0)


def evaluate_storage(actual: Path, evidence: dict):
    path = actual / "cache.json"
    execution = json.loads(path.read_text()) if path.is_file() else {}
    passed = (
        evidence.get("returncode") == 0
        and evidence.get("process_group_exited") is True
        and evidence.get("device_recovery_required") is False
        and execution.get("status") == "numerical_pass"
        and execution.get("released") is True
        and execution.get("writes") == 16
        and execution.get("cache_payload_bytes") == 1048576
        and execution.get("write_payload_bytes") == 2048
        and execution.get("fixed_reader_verifications") == 1
    )
    return {
        "status": "numerical_pass" if passed else "failed",
        "checks": {"slot_write_and_fixed_reader": {"passed": passed}},
        "execution": execution,
        "evidence": evidence,
        "hardware_execution_proven": False,
        "device_residency_verified": False,
    }


def evaluate(fixture: Path, actual: Path, metadata: dict, evidence: dict):
    results = {}
    for observation in metadata["observations"]:
        sequence, index = observation["sequence"], observation["index"]
        for field, spec in observation["outputs"].items():
            key = f"{sequence}.{field}.{index}"
            results[key] = compare_file(
                actual / f"{key}.bin",
                fixture / f"{key}.bin",
                spec,
                metadata["tolerances"]["fp16_pipeline"],
            )
            if sequence in ("reset", "fresh", "final"):
                current, initial = actual / f"{key}.bin", actual / f"zero.{field}.{index}.bin"
                results[f"repeat.{key}"] = (
                    compare_prefix(current, initial, spec, observation["length"])
                    if field in ("keys", "values")
                    else compare_file(current, initial, spec, {"atol": 0, "rtol": 0})
                )
            if field == "probabilities":
                path = actual / f"{key}.bin"
                passed = False
                if path.is_file() and path.stat().st_size == spec["bytes"]:
                    values = np.fromfile(path, dtype=spec["dtype"]).reshape(spec["shape"])
                    passed = bool(np.all(values[..., observation["length"] :] == 0))
                results[f"masked_suffix.{key}"] = {"passed": passed}
    report = actual / "execution.json"
    execution = json.loads(report.read_text()) if report.is_file() else {}
    steps = metadata["steps"]
    total = 2 * steps + 2 + 2 * min(8, steps)
    lifecycle = (
        evidence.get("returncode") == 0
        and evidence.get("process_group_exited") is True
        and evidence.get("device_recovery_required") is False
        and execution.get("status") == "executed"
        and execution.get("phase") == "complete"
        and execution.get("completed_sequences") == 5
        and execution.get("completed_steps") == total
        and execution.get("cache_writes") == total
        and execution.get("invalid_truncate_rejected") is True
        and (steps != CAPACITY or execution.get("capacity_rejected") is True)
        and execution.get("cache_copies") == 1
        and execution.get("cache_payload_bytes") == 1048576
        and execution.get("slot_write_payload_bytes") == 2048
        and execution.get("per_step_host_kv_transfers") is False
    )
    numerical = bool(results) and all(check["passed"] for check in results.values())
    log_path = actual / "sdk.log"
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    return {
        "status": "numerical_pass" if numerical and lifecycle else "failed",
        "numerical": numerical,
        "lifecycle": lifecycle,
        "cache_reuse_numerical": numerical and lifecycle,
        "hardware_execution_proven": False,
        "device_residency_verified": False,
        "sdk_matrix_node_creation_warnings": log.count("Call vxBatchGemmNode fail"),
        "sdk_warning_note": "Numerical success does not identify the selected execution backend.",
        "copy_graph_revalidations": execution.get("copy_graph_revalidations"),
        "checks": results,
        "execution": execution,
        "evidence": evidence,
    }
