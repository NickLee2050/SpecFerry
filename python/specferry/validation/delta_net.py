"""Independent single-layer reference fixtures and DeltaNet acceptance checks."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as reference

from specferry.export.weights import verify_export
from specferry.reference.model import source_record
from specferry.reference.runner import TOLERANCES, precision

from .capabilities import compare_arrays
from .device import fingerprint, write_json

PREFIX = "model.layers.0.linear_attn."
OUTPUTS = (
    "output",
    "recurrent",
    "convolution",
    "qkv",
    "convolved",
    "decay",
    "beta",
    "core",
    "gate",
    "gated",
)


class ReferenceState:
    """Only the cache protocol used by the official single-token mixer."""

    def __init__(self, recurrent, convolution):
        self.layers = [
            SimpleNamespace(
                recurrent_states=[recurrent.clone()],
                conv_states=[convolution.clone()],
                record_past=False,
            )
        ]

    def has_previous_state(self, layer_idx, state_idx=0):
        return True

    def update_recurrent_state(self, state, layer_idx):
        self.layers[layer_idx].recurrent_states[0] = state


def load_mixer(model: Path, manifest: dict):
    config = Qwen3_5TextConfig(**manifest["text_config"])
    with torch.device("meta"):
        mixer = reference.Qwen3_5GatedDeltaNet(config, 0)
    state = {}
    with (model / "weights.bin").open("rb") as file:
        for record in manifest["tensors"]:
            if not record["name"].startswith(PREFIX):
                continue
            file.seek(record["offset"])
            raw = file.read(record["bytes"])
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"corrupted mixer weight: {record['name']}")
            dtype = "<f2" if record["dtype"] == "F16" else "<f4"
            value = np.frombuffer(raw, dtype=dtype).copy().reshape(record["shape"])
            state[record["name"].removeprefix(PREFIX)] = torch.from_numpy(value)
    mixer.load_state_dict(state, strict=True, assign=True)
    return mixer.eval().requires_grad_(False)


def captured_inputs(trace: Path, steps: int):
    if not 1 <= steps <= 32:
        raise ValueError("DeltaNet validation requires 1-32 steps")
    suffix = ".layer.0.linear_attn.in_proj_qkv.input.0"
    with np.load(trace, allow_pickle=False) as arrays:
        names = sorted(name for name in arrays.files if name.endswith(suffix))
        if not names:
            raise ValueError("trace has no layer-0 mixer inputs")
        values = [arrays[name].copy() for name in names]
    if any(x.shape != (1, 1, 1024) or x.dtype != np.float16 for x in values):
        raise ValueError("trace must contain single-token FP16 hidden vectors")
    if any(not np.isfinite(x).all() for x in values):
        raise ValueError("trace contains nonfinite inputs")
    # Cycling saved activations is a layer trajectory, not meaningful text decoding.
    indices = [index % len(values) for index in range(steps)]
    return [torch.from_numpy(values[index].copy()) for index in indices], [
        names[i] for i in indices
    ]


def save_tensor(path: Path, value: torch.Tensor):
    array = value.detach().contiguous().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite reference: {path.name}")
    path.write_bytes(array.tobytes())
    return {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.nbytes}


def reference_sequence(mixer, inputs, recurrent, convolution, destination, name):
    cache = ReferenceState(recurrent, convolution)
    captured = {}
    original = reference.torch_recurrent_gated_delta_rule

    def recurrence(query, key, value, **kwargs):
        result = original(query, key, value, **kwargs)
        captured.update(
            convolved=torch.cat([x.reshape(1, 1, -1) for x in (query, key, value)], dim=-1),
            decay=kwargs["g"].exp(),
            beta=kwargs["beta"].float(),
            # Keep the official rounded values; only serialize them in the
            # device diagnostic's FP32 format. Do not alter the CPU arithmetic.
            core=result[0].float(),
        )
        return result

    def projection_hook(module, args, output):
        captured["qkv"] = output

    handle = mixer.in_proj_qkv.register_forward_hook(projection_hook)
    gate_handle = mixer.in_proj_z.register_forward_hook(
        lambda module, args, output: captured.update(gate=output)
    )
    norm_handle = mixer.norm.register_forward_hook(
        lambda module, args, output: captured.update(gated=output)
    )
    specs = {}
    try:
        with (
            torch.inference_mode(),
            precision("deployment-fp16"),
            patch.object(reference, "torch_recurrent_gated_delta_rule", recurrence),
        ):
            for index, hidden in enumerate(inputs):
                captured.clear()
                captured["output"] = mixer(hidden, cache_params=cache)
                captured["recurrent"] = cache.layers[0].recurrent_states[0]
                captured["convolution"] = cache.layers[0].conv_states[0]
                for output in OUTPUTS:
                    specs[output] = save_tensor(
                        destination / f"{name}.{output}.{index}.bin", captured[output]
                    )
    finally:
        handle.remove()
        gate_handle.remove()
        norm_handle.remove()
    return specs


def prepare(model: Path, trace: Path, destination: Path, steps: int):
    verify_export(model)
    manifest = json.loads((model / "deployment-manifest.json").read_text())
    inputs, input_names = captured_inputs(trace, steps)
    mixer = load_mixer(model, manifest)
    destination.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(109)
    recurrent = torch.randn((1, 16, 128, 128), generator=generator) * 0.01
    convolution = (torch.randn((1, 6144, 4), generator=generator) * 0.02).half()
    save_tensor(destination / "initial.recurrent.bin", recurrent)
    save_tensor(destination / "initial.convolution.bin", convolution)
    for index, value in enumerate(inputs):
        save_tensor(destination / f"input.{index}.bin", value)
    specs = reference_sequence(
        mixer,
        inputs,
        torch.zeros_like(recurrent),
        torch.zeros_like(convolution),
        destination,
        "zero",
    )
    reference_sequence(mixer, inputs, recurrent, convolution, destination, "nonzero")
    metadata = {
        "steps": steps,
        "outputs": specs,
        "reference": source_record(),
        "trace_sha256": fingerprint(trace),
        "input_names": input_names,
        "input_policy": "cycle captured layer-0 single-token FP16 inputs",
        "manifest_sha256": fingerprint(model / "deployment-manifest.json"),
        "seed": 109,
        "tolerances": TOLERANCES,
        "fixture_sha256": {
            path.name: fingerprint(path) for path in sorted(destination.glob("*.bin"))
        },
    }
    write_json(destination / "reference.json", metadata)
    return metadata


def compare_file(actual: Path, expected: Path, spec: dict, tolerance: dict):
    for path in (actual, expected):
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            return {"passed": False, "reason": f"missing or wrong-size file: {path.name}"}
    return compare_arrays(
        np.fromfile(actual, dtype=spec["dtype"]),
        np.fromfile(expected, dtype=spec["dtype"]),
        **tolerance,
    )


def evaluate(fixture: Path, actual: Path, metadata: dict, evidence: dict):
    results = {}
    policy = metadata["tolerances"]
    for sequence in ("zero", "nonzero", "reset", "fresh", "final"):
        expected_sequence = "nonzero" if sequence in ("nonzero", "final") else "zero"
        steps = [metadata["steps"] - 1] if sequence == "final" else range(metadata["steps"])
        for step in steps:
            for name, spec in metadata["outputs"].items():
                key = f"{sequence}.{name}.{step}"
                tolerance = policy[
                    "fixed_input_fp16_output"
                    if name == "qkv"
                    else "fp32_state_from_fp16_pipeline"
                    if np.dtype(spec["dtype"]) == np.float32
                    else "fp16_pipeline"
                ]
                results[key] = compare_file(
                    actual / f"{key}.bin",
                    fixture / f"{expected_sequence}.{name}.{step}.bin",
                    spec,
                    tolerance,
                )
                # A reset and a fresh graph must agree with the same device's
                # original sequence. Final-only execution must match observed execution.
                if sequence in ("reset", "fresh", "final"):
                    results[f"repeat.{key}"] = compare_file(
                        actual / f"{key}.bin",
                        actual / f"{expected_sequence}.{name}.{step}.bin",
                        spec,
                        {"atol": 0, "rtol": 0},
                    )
    path = actual / "execution.json"
    execution = json.loads(path.read_text()) if path.is_file() else {}
    lifecycle = (
        evidence.get("returncode") == 0
        and evidence.get("process_group_exited") is True
        and evidence.get("device_recovery_required") is False
        and execution.get("status") == "executed"
        and execution.get("phase") == "complete"
        and execution.get("completed_sequences") == 5
        and execution.get("completed_steps") == metadata["steps"]
        and execution.get("per_step_host_state_uploads") == 0
        and execution.get("application_readbacks") == (metadata["steps"] * 4 + 1) * len(OUTPUTS)
    )
    numerical = bool(results) and all(value["passed"] for value in results.values())
    log_path = actual / "sdk.log"
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    matrix_warnings = log.count("Call vxBatchGemmNode fail")
    return {
        "status": "numerical_pass" if numerical and lifecycle else "failed",
        "numerical": numerical,
        "lifecycle": lifecycle,
        "state_reuse_numerical": numerical and lifecycle,
        "hardware_execution_proven": False,
        "device_residency_verified": False,
        "sdk_matrix_node_creation_warnings": matrix_warnings,
        "sdk_warning_note": "A failed VX node creation followed by numerical success does not "
        "identify which execution backend was selected.",
        "blockers": ["per_node_hardware_evidence_missing", "sdk_state_transfer_evidence_missing"],
        "checks": results,
        "execution": execution,
        "evidence": evidence,
    }
