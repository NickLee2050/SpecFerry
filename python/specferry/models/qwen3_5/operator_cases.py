"""Retained Qwen-scale capability catalog and independent small SDK probes."""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from specferry.validation.operator_cases import (
    Case,
    activation,
    argmax,
    conversion,
    gather,
    layout,
    masked_softmax,
    matrix,
    normalization,
    scatter,
    short_convolution,
)

from .extended_cases import extend_catalog


def catalog() -> dict[str, Case]:
    cases = {}

    def add(name, family, scale, function, *args, **kwargs):
        cases[name] = Case(name, family, scale, partial(function, name, *args, **kwargs))

    add("matmul_fp16_small", "linear", "small", matrix, 8, 5)
    add("fcl_fp16_small", "linear", "small", matrix, 8, 5, fcl=True)
    add("matmul_dynamic_fp32", "state_matrix", "small", matrix, 8, 5, dtype="F32", dynamic=True)
    add(
        "matmul_batched_fp32",
        "state_matrix",
        "small",
        matrix,
        8,
        5,
        dtype="F32",
        dynamic=True,
        batch=2,
        transpose=False,
    )
    for scale, width in (("small", 8), ("model", 1024)):
        for mean, kind in ((True, "rms"), (False, "l2")):
            add(
                f"{kind}_fp32_{scale}",
                "normalization",
                scale,
                normalization,
                128 if not mean and scale == "model" else width,
                mean=mean,
            )
        add(
            f"short_conv_{scale}",
            "short_convolution",
            scale,
            short_convolution,
            8 if scale == "small" else 6144,
        )
        add(
            f"mask_softmax_{scale}",
            "attention_mask",
            scale,
            masked_softmax,
            8 if scale == "small" else 512,
        )
        add(
            f"scatter_{scale}",
            "cache_update",
            scale,
            scatter,
            8 if scale == "small" else 512,
            4 if scale == "small" else 512,
        )
        add(f"argmax_{scale}", "token_selection", scale, argmax, 8 if scale == "small" else 4096)
    for operation, dtype in (
        ("SWISH", "F16"),
        ("SIGMOID", "F16"),
        ("SIGMOID", "F32"),
        ("SWISH", "F32"),
        ("EXP", "F32"),
        ("SOFTRELU", "F32"),
    ):
        add(
            f"{operation.lower()}_{dtype.lower()}",
            "activation",
            "small",
            activation,
            operation,
            16,
            dtype,
        )
    add("gather_small", "lookup", "small", gather, 17, 8)
    add("gather_vocabulary", "lookup", "model", gather, 248320, 1024)
    add("layout", "layout", "small", layout)
    for source, target in (("F16", "F32"), ("F32", "F16")):
        add(
            f"convert_{source.lower()}_{target.lower()}",
            "dtype_conversion",
            "small",
            conversion,
            source,
            target,
        )
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
    return {name: replace(case, tolerances=policy) for name, case in cases.items()}
