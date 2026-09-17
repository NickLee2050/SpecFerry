"""Model-shaped primitive probes, independent of full decoding."""

from functools import partial

import numpy as np

from .fixtures import Fixture, array


def weight_storage(name, storage):
    fixture = Fixture(name, "weight_storage", steps=3)
    random = np.random.default_rng(104)
    fixture.provenance["seed"] = 104
    first = array(random.normal(0, 0.2, (1, 8)), "F16")
    second = array(random.normal(0, 0.2, (1, 8)), "F16")
    inputs = [first, second, second]
    base = array(random.normal(0, 0.2, (5, 8)), "F16")
    weights = [base, base, array(base + 0.125, "F16")]
    fixture.input("x", inputs)
    if storage == "constant":
        weights = [base] * fixture.steps
        fixture.constant("weight", base)
    else:
        fixture.input("weight", weights)
    fixture.node("MATRIXMUL", ["x", "weight"], "projection", (1, 5), parameters=(0, 1))
    outputs = [
        array(x.astype(np.float32) @ weight.astype(np.float32).T, "F16")
        for x, weight in zip(inputs, weights, strict=True)
    ]
    output = "projection"
    if storage == "mixed":
        fixture.constant("fixed_weight", -base)
        fixture.node("MATRIXMUL", ["x", "fixed_weight"], "fixed", (1, 5), parameters=(0, 1))
        fixture.node("ADD", ["projection", "fixed"], "y", (1, 5))
        outputs = [
            value + array(x.astype(np.float32) @ (-base).astype(np.float32).T, "F16")
            for x, value in zip(inputs, outputs, strict=True)
        ]
        output = "y"
    fixture.output(output, outputs)
    return fixture


def delta_update(name):
    """Test the FP32 outer product, delta correction and per-head decay independently."""
    fixture = Fixture(name, "state_matrix")
    random = np.random.default_rng(105)
    fixture.provenance["seed"] = 105
    states = [array(random.normal(0, 0.03, (16, 128, 128)), "F32") for _ in range(2)]
    keys = [array(random.normal(0, 0.03, (16, 1, 128)), "F32") for _ in range(2)]
    values = [array(random.normal(0, 0.1, (16, 1, 128)), "F32") for _ in range(2)]
    fixture.input("state", states, "F32")
    fixture.input("key", keys, "F32")
    fixture.input("value", values, "F32")
    decay = np.linspace(0.5, 0.95, 16, dtype=np.float32).reshape(16, 1, 1)
    beta = np.linspace(0.125, 0.75, 16, dtype=np.float32).reshape(16, 1, 1)
    fixture.constant("decay", decay, "F32")
    fixture.constant("beta", beta, "F32")
    fixture.node("MULTIPLY", ["state", "decay"], "decayed", (16, 128, 128), "F32")
    fixture.node("MATRIXMUL", ["key", "decayed"], "prediction", (16, 1, 128), "F32", (0, 0))
    fixture.node("SUBTRACT", ["value", "prediction"], "error", (16, 1, 128), "F32")
    fixture.node("MULTIPLY", ["error", "beta"], "correction", (16, 1, 128), "F32")
    fixture.node("MATRIXMUL", ["key", "correction"], "outer", (16, 128, 128), "F32", (1, 0))
    fixture.node("ADD", ["decayed", "outer"], "next", (16, 128, 128), "F32")
    expected = []
    for state, key, value in zip(states, keys, values, strict=True):
        decayed = state * decay
        correction = beta * (value - key @ decayed)
        expected.append(decayed + np.swapaxes(key, -1, -2) @ correction)
    fixture.output("next", expected)
    return fixture


def attention_product(name, probability_value):
    fixture = Fixture(name, "attention_matrix")
    random = np.random.default_rng(106)
    fixture.provenance["seed"] = 106
    left_shape = (8, 1, 512 if probability_value else 256)
    right_shape = (8, 512, 256)
    left = [array(random.normal(0, 0.03, left_shape), "F32") for _ in range(2)]
    right = [array(random.normal(0, 0.03, right_shape), "F32") for _ in range(2)]
    fixture.input("left", left, "F32")
    fixture.input("right", right, "F32")
    shape = (8, 1, 256 if probability_value else 512)
    fixture.node("MATRIXMUL", ["left", "right"], "y", shape, "F32", (0, int(not probability_value)))
    fixture.output(
        "y",
        [
            x @ (value if probability_value else np.swapaxes(value, -1, -2))
            for x, value in zip(left, right, strict=True)
        ],
    )
    return fixture


def head_split(name):
    fixture = Fixture(name, "attention_layout")
    values = [array(np.arange(4096).reshape(1, -1) / 4096 + step / 8, "F16") for step in range(2)]
    fixture.input("packed", values)
    fixture.node("RESHAPE", ["packed"], "heads", (8, 512), parameters=(512, 8))
    fixture.node("SLICE", ["heads"], "query", (8, 256), parameters=(0, 0, 256, 8))
    fixture.node("SLICE", ["heads"], "gate", (8, 256), parameters=(256, 0, 256, 8))
    fixture.output("query", [value.reshape(8, 512)[:, :256] for value in values])
    fixture.output("gate", [value.reshape(8, 512)[:, 256:] for value in values])
    return fixture


def gqa_layout(name):
    fixture = Fixture(name, "attention_layout")
    values = [
        array(np.arange(2 * 3 * 256).reshape(2, 3, 256) / 2048 + step, "F16") for step in range(2)
    ]
    fixture.input("kv", values)
    indices = np.repeat(np.arange(2, dtype=np.int32), 4)
    fixture.constant("heads", indices, "I32")
    fixture.integer_bounds["heads"] = (0, 1)
    fixture.node("GATHER", ["kv", "heads"], "y", (8, 3, 256), parameters=(2,))
    fixture.output("y", [value[indices] for value in values])
    return fixture


def partial_rope(name):
    # Text uses the same position on all three mRoPE axes; only 64/256 dimensions rotate.
    fixture = Fixture(name, "position_encoding", steps=6)
    positions = [0, 1, 3, 7, 255, 511]
    random = np.random.default_rng(107)
    fixture.provenance["seed"] = 107
    inputs = [array(random.normal(0, 0.2, (8, 256)), "F32") for _ in positions]
    frequencies = 1 / (10000000 ** (np.arange(0, 64, 2, dtype=np.float32) / 64))
    angles = [np.concatenate([position * frequencies] * 2).reshape(1, 64) for position in positions]
    cosines = [np.cos(angle).astype(np.float32) for angle in angles]
    sines = [np.sin(angle).astype(np.float32) for angle in angles]
    fixture.input("x", inputs, "F32")
    fixture.input("cosine", cosines, "F32")
    fixture.input("sine", sines, "F32")
    fixture.node("SLICE", ["x"], "rotary", (8, 64), "F32", (0, 0, 64, 8))
    fixture.node("SLICE", ["x"], "unchanged", (8, 192), "F32", (64, 0, 192, 8))
    fixture.node("SLICE", ["rotary"], "first", (8, 32), "F32", (0, 0, 32, 8))
    fixture.node("SLICE", ["rotary"], "second", (8, 32), "F32", (32, 0, 32, 8))
    fixture.constant("minus_one", [[-1]], "F32")
    fixture.node("MULTIPLY", ["second", "minus_one"], "negative_second", (8, 32), "F32")
    fixture.node("CONCAT", ["negative_second", "first"], "turned", (8, 64), "F32", (0,))
    fixture.node("MULTIPLY", ["rotary", "cosine"], "cos_part", (8, 64), "F32")
    fixture.node("MULTIPLY", ["turned", "sine"], "sin_part", (8, 64), "F32")
    fixture.node("ADD", ["cos_part", "sin_part"], "rotated", (8, 64), "F32")
    fixture.node("CONCAT", ["rotated", "unchanged"], "y", (8, 256), "F32", (0,))
    outputs = []
    for x, cosine, sine in zip(inputs, cosines, sines, strict=True):
        rotated_half = np.concatenate([-x[:, 32:64], x[:, :32]], axis=-1)
        outputs.append(
            np.concatenate([x[:, :64] * cosine + rotated_half * sine, x[:, 64:]], axis=-1)
        )
    fixture.output("y", outputs)
    fixture.provenance.update(text_positions=positions, rotated_dimensions=64)
    return fixture


def weighted_norm(name):
    fixture = Fixture(name, "normalization")
    random = np.random.default_rng(108)
    fixture.provenance["seed"] = 108
    inputs = [random.normal(0, 0.2, (2, 1024)).astype(np.float32), np.zeros((2, 1024), np.float32)]
    weight = random.normal(0, 0.1, (1, 1024)).astype(np.float32)
    fixture.input("x", inputs, "F32")
    fixture.constant("weight", weight, "F32")
    fixture.constant("one", [[1]], "F32")
    fixture.constant("epsilon", [[1e-6]], "F32")
    fixture.node("MULTIPLY", ["x", "x"], "squared", (2, 1024), "F32")
    fixture.node("REDUCE_MEAN", ["squared"], "mean", (2, 1), "F32", (0,))
    fixture.node("ADD", ["mean", "epsilon"], "shifted", (2, 1), "F32")
    fixture.node("RSQRT", ["shifted"], "inverse", (2, 1), "F32")
    fixture.node("MULTIPLY", ["x", "inverse"], "normalized", (2, 1024), "F32")
    fixture.node("ADD", ["one", "weight"], "scale", (1, 1024), "F32")
    fixture.node("MULTIPLY", ["normalized", "scale"], "y", (2, 1024), "F32")
    fixture.output(
        "y",
        [x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * (1 + weight) for x in inputs],
    )
    return fixture


def block_selection(name):
    """Exercise a device-side merge with invalid tail rows and cross-block ties."""
    fixture = Fixture(name, "token_selection")
    first = np.array([[1, 2, 5, 4, 5, 3, 100, 100]], np.float16)
    second = np.array([[1, 2, 3, 4, 5, 6, 100, 100]], np.float16)
    fixture.input("logits", [first, second])
    fixture.constant("positions", np.arange(8, dtype=np.int32).reshape(1, 8), "I32")
    fixture.constant("valid_count", [[6]], "I32")
    fixture.constant("invalid", [[-65504]])
    fixture.node("LESS", ["positions", "valid_count"], "valid", (1, 8), "BOOL")
    fixture.node("SELECT", ["valid", "logits", "invalid"], "masked", (1, 8))
    for index in range(2):
        prefix = f"block{index}"
        fixture.node("SLICE", ["masked"], prefix, (1, 4), parameters=(index * 4, 0, 4, 1))
        fixture.node("REDUCE_MAX", [prefix], f"max{index}", (1, 1), parameters=(0,))
        fixture.node("ARGMAX", [prefix], f"index{index}", (1, 1), "I32", (0,))
    fixture.constant("offset", [[4]], "I32")
    fixture.node("ADD", ["index1", "offset"], "global1", (1, 1), "I32")
    fixture.node("LESS", ["max0", "max1"], "choose_second", (1, 1), "BOOL")
    fixture.node("SELECT", ["choose_second", "global1", "index0"], "token", (1, 1), "I32")
    fixture.output("token", [np.array([[2]], np.int32), np.array([[5]], np.int32)])
    return fixture


def extend_catalog(cases, case_type):
    def add(name, family, scale, function, *args):
        cases[name] = case_type(name, family, scale, partial(function, name, *args))

    for storage in ("constant", "mutable", "mixed"):
        add(f"weights_{storage}_fp16", "weight_storage", "small", weight_storage, storage)
    add("delta_update_fp32", "state_matrix", "model", delta_update)
    add("attention_qk_fp32", "attention_matrix", "model", attention_product, False)
    add("attention_pv_fp32", "attention_matrix", "model", attention_product, True)
    add("attention_head_split", "attention_layout", "small", head_split)
    add("attention_gqa_layout", "attention_layout", "small", gqa_layout)
    add("partial_rope_fp32", "position_encoding", "model", partial_rope)
    add("weighted_rms_fp32", "normalization", "model", weighted_norm)
    add("block_token_selection", "token_selection", "small", block_selection)
