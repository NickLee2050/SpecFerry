"""Model-independent storage and block-selection probes with fixed diagnostic inputs."""

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
