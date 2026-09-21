"""Qwen module-to-trace mappings for real-weight projection probes."""

import json
from functools import partial
from pathlib import Path

from specferry.validation.operator_cases import Case
from specferry.validation.reference_cases import projection_case


def reference_catalog(model: Path, trace: Path) -> dict[str, Case]:
    modules = {
        "reference_delta_qkv": "layers.0.linear_attn.in_proj_qkv",
        "reference_delta_gate": "layers.0.linear_attn.in_proj_a",
        "reference_attention_q_gate": "layers.3.self_attn.q_proj",
        "reference_mlp_up": "layers.0.mlp.up_proj",
    }
    policy = json.loads(Path(__file__).with_name("tolerances.json").read_text())
    return {
        name: Case(
            name,
            "reference_projection",
            "model",
            partial(
                projection_case,
                name,
                model,
                trace,
                f"model.{module}.weight",
                module.replace("layers.", "layer.", 1),
            ),
            tolerances=policy,
        )
        for name, module in modules.items()
    }
