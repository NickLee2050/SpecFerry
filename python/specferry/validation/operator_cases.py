"""Capability cases for the first DLM; every expected result uses fixed input bytes."""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from .fixtures import Fixture, array


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    scale: str
    build: Callable[[], Fixture]


def matrix(name, k, n, *, dtype="F16", dynamic=False, batch=1, transpose=True, fcl=False):
    fixture = Fixture(name, "linear" if batch == 1 else "state_matrix")
    random = np.random.default_rng(101)
    inputs = [array(random.normal(0, 0.25, (batch, 1, k)), dtype) for _ in range(2)]
    weights = [array(random.normal(0, 1 / np.sqrt(k), (batch, n, k)), dtype) for _ in range(2)]
    if not dynamic:
        weights[1] = weights[0]
    if batch == 1:
        inputs = [value[0] for value in inputs]
        weights = [value[0] for value in weights]
    fixture.input("x", inputs, dtype)
    stored = weights if transpose else [np.swapaxes(value, -1, -2) for value in weights]
    if dynamic:
        fixture.input("w", stored, dtype)
    else:
        fixture.constant("w", stored[0], dtype)
    output_shape = (*inputs[0].shape[:-1], n)
    fixture.node(
        "FCL" if fcl else "MATRIXMUL",
        ["x", "w"],
        "y",
        output_shape,
        dtype,
        (n,) if fcl else (0, int(transpose)),
    )
    fixture.output(
        "y",
        [
            x.astype(np.float32) @ np.swapaxes(w.astype(np.float32), -1, -2)
            for x, w in zip(inputs, weights, strict=True)
        ],
    )
    return fixture


def normalization(name, width, *, mean=True):
    fixture = Fixture(name, "normalization")
    random = np.random.default_rng(101)
    values = [array(random.normal(0, 0.2, (2, width)), "F32"), np.zeros((2, width), np.float32)]
    fixture.input("x", values, "F32")
    fixture.constant("epsilon", [[1e-6]], "F32")
    fixture.node("MULTIPLY", ["x", "x"], "squared", (2, width), "F32")
    fixture.node(
        "REDUCE_MEAN" if mean else "REDUCE_SUM", ["squared"], "reduced", (2, 1), "F32", (0,)
    )
    fixture.node("ADD", ["reduced", "epsilon"], "shifted", (2, 1), "F32")
    fixture.node("RSQRT", ["shifted"], "inverse", (2, 1), "F32")
    fixture.node("MULTIPLY", ["x", "inverse"], "y", (2, width), "F32")
    reduction = np.mean if mean else np.sum
    fixture.output(
        "y", [x / np.sqrt(reduction(x * x, axis=-1, keepdims=True) + 1e-6) for x in values]
    )
    return fixture


def activation(name, operation, width, dtype):
    fixture = Fixture(name, "activation")
    # The second execution changes the input without rebuilding the graph.
    bound = 80 if operation == "SOFTRELU" else 8
    values = [
        array(np.linspace(-bound, bound, width).reshape(1, -1), dtype),
        array(np.linspace(bound, -bound, width).reshape(1, -1), dtype),
    ]
    fixture.input("x", values, dtype)
    fixture.node(operation, ["x"], "y", values[0].shape, dtype)
    functions = {
        "SWISH": lambda x: x / (1 + np.exp(-x)),
        "SIGMOID": lambda x: 1 / (1 + np.exp(-x)),
        "EXP": np.exp,
        "SOFTRELU": lambda x: np.logaddexp(0, x),
    }
    fixture.output("y", [functions[operation](x.astype(np.float32)) for x in values])
    return fixture


def gather(name, rows, width):
    fixture = Fixture(name, "lookup")
    # Distinct row and column signatures catch axis swaps and stale token indices.
    table = (np.arange(rows, dtype=np.float32)[:, None] % 113) / 128
    table = array(table + (np.arange(width, dtype=np.float32)[None, :] % 31) / 64, "F16")
    indices = [np.array([0, rows - 1], np.int32), np.array([rows - 1, rows // 2], np.int32)]
    fixture.constant("table", table)
    fixture.input("indices", indices, "I32")
    fixture.node("GATHER", ["table", "indices"], "y", (2, width), parameters=(1,))
    fixture.output("y", [table[index] for index in indices])
    return fixture


def state_feedback(name, width, dtype, handle):
    fixture = Fixture(name, "state_feedback", steps=3, reset_after=2)
    shape = (1, width) if isinstance(width, int) else width
    initial = array(np.full(shape, 0.125), dtype)
    updates = [array(np.full(shape, value), dtype) for value in (0.25, 0.5, 0.25)]
    fixture.input("update", updates, dtype)
    fixture.tensor("state", initial, dtype, "handle" if handle else "mutable", initialize=True)
    fixture.node("ADD", ["state", "update"], "next", initial.shape, dtype)
    fixture.tensors["next"]["storage"] = "handle" if handle else "mutable"
    fixture.feedback = ("next", "state")
    fixture.output(
        "next", [initial + updates[0], initial + updates[0] + updates[1], initial + updates[2]]
    )
    return fixture


def conversion(name, source, target):
    fixture = Fixture(name, "dtype_conversion")
    values = [
        array([[0, 0.125, -0.25, 3.5, 16]], source),
        array([[1, -0.125, 0.25, -3.5, 32]], source),
    ]
    fixture.input("x", values, source)
    fixture.node("DATACONVERT", ["x"], "y", values[0].shape, target)
    fixture.output("y", values)
    return fixture


def short_convolution(name, channels):
    fixture = Fixture(name, "short_convolution")
    random = np.random.default_rng(101)
    windows = [array(random.normal(0, 0.25, (channels, 4)), "F16") for _ in range(2)]
    weight = array(random.normal(0, 0.5, (channels, 4)), "F16")
    fixture.input("window", windows)
    fixture.constant("weight", weight)
    # Equivalent four-tap depthwise convolution, with a known FP16 boundary before reduction.
    fixture.node("MULTIPLY", ["window", "weight"], "products", (channels, 4))
    fixture.node("REDUCE_SUM", ["products"], "y", (channels, 1), parameters=(0,))
    fixture.output(
        "y", [array(x * weight, "F16").astype(np.float32).sum(-1, keepdims=True) for x in windows]
    )
    return fixture


def masked_softmax(name, length):
    fixture = Fixture(name, "attention_mask")
    random = np.random.default_rng(101)
    scores = [array(random.normal(0, 0.5, (8, length)), "F32") for _ in range(2)]
    lengths = [np.array([[1]], np.int32), np.array([[length - 1]], np.int32)]
    fixture.input("scores", scores, "F32")
    fixture.input("length", lengths, "I32")
    fixture.constant("positions", np.arange(length, dtype=np.int32).reshape(1, -1), "I32")
    fixture.constant("negative", [[-1e9]], "F32")
    fixture.node("LESS", ["positions", "length"], "mask", (1, length), "BOOL")
    fixture.node("SELECT", ["mask", "scores", "negative"], "masked", (8, length), "F32")
    fixture.node("SOFTMAX", ["masked"], "y", (8, length), "F32", (0,))
    outputs = []
    for score, valid in zip(scores, lengths, strict=True):
        value = np.where(np.arange(length)[None, :] < valid, score, -1e9)
        probabilities = np.exp(value - value.max(axis=-1, keepdims=True))
        outputs.append(probabilities / probabilities.sum(axis=-1, keepdims=True))
    fixture.output("y", outputs)
    return fixture


def scatter(name, length, width):
    fixture = Fixture(name, "cache_update")
    cache = array(np.arange(length * width).reshape(length, width) / (length * width), "F16")
    positions = [np.array([[0]], np.int32), np.array([[length - 1]], np.int32)]
    updates = [array(np.full((1, width), value), "F16") for value in (-0.5, 0.25)]
    fixture.input("cache", [cache, cache])
    fixture.input("positions", positions, "I32")
    fixture.input("updates", updates)
    fixture.node("SCATTER_ND_UPDATE", ["cache", "positions", "updates"], "y", cache.shape)
    results = []
    for position, update in zip(positions, updates, strict=True):
        result = cache.copy()
        result[position[0, 0]] = update[0]
        results.append(result)
    fixture.output("y", results)
    return fixture


def argmax(name, width):
    fixture = Fixture(name, "token_selection")
    first = np.zeros((1, width), np.float16)
    first[0, width - 2 :] = 5  # Equal maxima must select the smaller valid token ID.
    second = np.zeros_like(first)
    second[0, 0] = 6
    fixture.input("logits", [first, second])
    fixture.node("ARGMAX", ["logits"], "y", (1, 1), "I32", (0,))
    fixture.output("y", [np.argmax(value, axis=-1, keepdims=True) for value in (first, second)])
    return fixture


def layout(name):
    fixture = Fixture(name, "layout")
    values = [
        array(np.arange(24).reshape(4, 6) / 32, "F16"),
        array(np.arange(24, 48).reshape(4, 6) / 64, "F16"),
    ]
    fixture.input("x", values)
    fixture.node("SLICE", ["x"], "sliced", (4, 3), parameters=(1, 0, 3, 4))
    fixture.node("PERMUTE", ["sliced"], "permuted", (3, 4), parameters=(1, 0))
    fixture.node("RESHAPE", ["permuted"], "reshaped", (2, 6), parameters=(6, 2))
    fixture.node("CONCAT", ["reshaped", "reshaped"], "y", (2, 12), parameters=(0,))
    fixture.output("y", [np.concatenate([x[:, 1:4].T.reshape(2, 6)] * 2, axis=-1) for x in values])
    return fixture


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
    for handle in (False, True):
        for dtype in ("F16", "F32"):
            add(
                f"state_{dtype.lower()}_{'handle' if handle else 'ordinary'}",
                "state_feedback",
                "small",
                state_feedback,
                16,
                dtype,
                handle,
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
    return cases
