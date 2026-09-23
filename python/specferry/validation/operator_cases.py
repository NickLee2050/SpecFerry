"""Parameterized operator fixtures with independent fixed-input expectations."""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from .extended_cases import block_selection, weight_storage
from .fixtures import Fixture, array

# Fixed-input budgets are independent of any checkpoint's end-to-end calibration.
# Preserve the previously validated operator thresholds when moving this catalog.
FIXED_INPUT_TOLERANCES = {
    "fixed_input_fp16_output": {"atol": 0.01, "rtol": 0.02},
    "fixed_input_fp32_state": {"atol": 0.0001, "rtol": 0.001},
}


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    scale: str
    build: Callable[[], Fixture]
    tolerances: dict | None = None


def catalog() -> dict[str, Case]:
    """Small SDK contracts with explicit shapes and no model adapter dependency."""
    cases = {}

    def add(name, family, function, *args, **kwargs):
        cases[name] = Case(
            name, family, "small", partial(function, name, *args, **kwargs), FIXED_INPUT_TOLERANCES
        )

    add("matmul_fp16_small", "linear", matrix, 8, 5)
    add("fcl_fp16_small", "linear", matrix, 8, 5, fcl=True)
    add("matmul_dynamic_fp32", "state_matrix", matrix, 8, 5, dtype="F32", dynamic=True)
    add(
        "matmul_batched_fp32",
        "state_matrix",
        matrix,
        8,
        5,
        dtype="F32",
        dynamic=True,
        batch=2,
        transpose=False,
    )
    for mean, kind in ((True, "rms"), (False, "l2")):
        add(f"{kind}_fp32_small", "normalization", normalization, 8, mean=mean)
    add("short_conv_small", "short_convolution", short_convolution, 8)
    add("mask_softmax_small", "attention_mask", masked_softmax, 8)
    add("scatter_small", "cache_update", scatter, 8, 4)
    add("argmax_small", "token_selection", argmax, 8)
    for operation, dtype in (
        ("SWISH", "F16"),
        ("SIGMOID", "F16"),
        ("SIGMOID", "F32"),
        ("SWISH", "F32"),
        ("EXP", "F32"),
        ("SOFTRELU", "F32"),
    ):
        add(f"{operation.lower()}_{dtype.lower()}", "activation", activation, operation, 16, dtype)
    add("gather_small", "lookup", gather, 17, 8)
    add("layout", "layout", layout)
    for source, target in (("F16", "F32"), ("F32", "F16")):
        add(
            f"convert_{source.lower()}_{target.lower()}",
            "dtype_conversion",
            conversion,
            source,
            target,
        )
    for storage in ("constant", "mutable", "mixed"):
        add(f"weights_{storage}_fp16", "weight_storage", weight_storage, storage)
    add("block_token_selection", "token_selection", block_selection)
    return cases


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
    fixture.integer_bounds["indices"] = (0, rows - 1)
    fixture.node("GATHER", ["table", "indices"], "y", (2, width), parameters=(1,))
    fixture.output("y", [table[index] for index in indices])
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
    fixture.integer_bounds["length"] = (1, length)
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
    fixture.integer_bounds["positions"] = (0, length - 1)
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
