"""Explicit Qwen dimensions/compositions extending the generic SDK catalog."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from specferry.validation.operator_cases import (
    Case,
    argmax,
    gather,
    masked_softmax,
    matrix,
    normalization,
    scatter,
    short_convolution,
)
from specferry.validation.operator_cases import (
    catalog as generic_catalog,
)

from .extended_cases import extend_catalog


def catalog() -> dict[str, Case]:
    cases = {}

    def add(name, family, scale, function, *args, **kwargs):
        cases[name] = Case(name, family, scale, partial(function, name, *args, **kwargs))

    for mean, kind, width in ((True, "rms", 1024), (False, "l2", 128)):
        add(f"{kind}_fp32_model", "normalization", "model", normalization, width, mean=mean)
    add("short_conv_model", "short_convolution", "model", short_convolution, 6144)
    add("mask_softmax_model", "attention_mask", "model", masked_softmax, 512)
    add("scatter_model", "cache_update", "model", scatter, 512, 512)
    add("argmax_model", "token_selection", "model", argmax, 4096)
    add("gather_vocabulary", "lookup", "model", gather, 248320, 1024)
    for name, k, n in (
        ("qkv", 1024, 6144),
        ("z", 1024, 2048),
        ("gate", 1024, 16),
        ("delta_out", 2048, 1024),
        ("q_gate", 1024, 4096),
        ("kv", 1024, 512),
        ("mlp_up", 1024, 3584),
        ("mlp_down", 3584, 1024),
        ("head_block", 1024, 4096),
    ):
        add(f"projection_{name}", "linear", "model", matrix, k, n)
    add(
        "state_matrix_model",
        "state_matrix",
        "model",
        matrix,
        128,
        128,
        dtype="F32",
        dynamic=True,
        batch=16,
    )
    add("projection_head_tail", "linear", "model", matrix, 1024, 2560)
    extend_catalog(cases, Case)
    policy = json.loads(Path(__file__).with_name("tolerances.json").read_text())
    return generic_catalog() | {
        name: replace(case, tolerances=policy) for name, case in cases.items()
    }
