"""Projection probes using exported weights and captured CPU reference activations."""

import json
from functools import partial
from pathlib import Path

import numpy as np

from .device import fingerprint
from .fixtures import Fixture
from .operator_cases import Case


def projection_case(name, model: Path, trace: Path, module: str):
    manifest = json.loads((model / "deployment-manifest.json").read_text())
    record = next(item for item in manifest["tensors"] if item["name"] == f"model.{module}.weight")
    if record["dtype"] != "F16" or len(record["shape"]) != 2:
        raise ValueError("reference projection requires a rank-two FP16 weight")
    with (model / "weights.bin").open("rb") as file:
        file.seek(record["offset"])
        weight = np.fromfile(file, dtype="<f2", count=record["bytes"] // 2).reshape(record["shape"])
    label = module.replace("layers.", "layer.", 1)
    with np.load(trace, allow_pickle=False) as arrays:
        inputs = [arrays[f"token.{step:04d}.{label}.input.0"].reshape(1, -1) for step in (1, 2)]
        outputs = [arrays[f"token.{step:04d}.{label}.output"].reshape(1, -1) for step in (1, 2)]
    fixture = Fixture(name, "reference_projection")
    fixture.provenance = {
        "kind": "captured_cpu_reference",
        "trace_sha256": fingerprint(trace),
        "deployment_manifest_sha256": fingerprint(model / "deployment-manifest.json"),
        "weight": record["name"],
        "weight_sha256": record["sha256"],
        "module": module,
        "tokens": [1, 2],
        "expected": "captured module output",
    }
    fixture.input("x", inputs)
    fixture.constant("w", weight)
    fixture.node("MATRIXMUL", ["x", "w"], "y", outputs[0].shape, parameters=(0, 1))
    fixture.output("y", outputs)
    return fixture


def reference_catalog(model: Path, trace: Path) -> dict[str, Case]:
    modules = {
        "reference_delta_qkv": "layers.0.linear_attn.in_proj_qkv",
        "reference_delta_gate": "layers.0.linear_attn.in_proj_a",
        "reference_attention_q_gate": "layers.3.self_attn.q_proj",
        "reference_mlp_up": "layers.0.mlp.up_proj",
    }
    return {
        name: Case(
            name,
            "reference_projection",
            "model",
            partial(projection_case, name, model, trace, module),
        )
        for name, module in modules.items()
    }
