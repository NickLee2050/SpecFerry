"""Single-step DeltaNet export retaining the accepted FP32 recurrence contract."""

import numpy as np
from onnx import TensorProto

from specferry.export.onnx import OnnxGraph


def vocabulary_head_model(weights, rows):
    """Compile a bounded vocabulary prefix, preserving one FP16 output per row block."""
    table = weights["model.embed_tokens.weight"]
    if (
        table.shape[1] != 1024
        or table.shape[0] < rows
        or table.dtype != np.float16
        or not 4096 <= rows <= 65536
        or rows % 4096
    ):
        raise ValueError("compiled head requires 4096-aligned FP16 vocabulary rows, at most 65536")
    graph = OnnxGraph({})
    hidden = graph.port("hidden", TensorProto.FLOAT16, [1, 1024])
    for index, first in enumerate(range(0, rows, 4096)):
        weight = graph.constant(f"weight_{index}", table[first : first + 4096])
        name = f"logits_{index}"
        graph.node("Gemm", [hidden, weight], name, transB=1, alpha=1.0, beta=0.0)
        graph.port(name, TensorProto.FLOAT16, [1, 4096], output=True)
    return graph


class DeltaGraph(OnnxGraph):
    def cast(self, value, name, dtype):
        return self.node("Cast", [value], name, to=dtype)

    def slice(self, value, name, first, last):
        values = [[first], [last], [1], [1]]
        controls = [
            self.constant(name + suffix, np.array(data, "i8"))
            for suffix, data in zip(("_starts", "_ends", "_axes", "_steps"), values)
        ]
        return self.node("Slice", [value, *controls], name)

    def normalize(self, value, prefix, epsilon, mean=True):
        wide = self.cast(value, prefix + "_fp32", TensorProto.FLOAT)
        squared = self.node("Mul", [wide, wide], prefix + "_squares")
        variance = self.node(
            "ReduceMean" if mean else "ReduceSum",
            [squared],
            prefix + "_reduction",
            axes=[-1],
            keepdims=1,
        )
        regularized = self.node(
            "Add",
            [variance, self.constant(prefix + "_epsilon", np.array(epsilon, "f4"))],
            prefix + "_regularized",
        )
        root = self.node("Sqrt", [regularized], prefix + "_root")
        normalized = self.node("Div", [wide, root], prefix + "_normalized")
        narrow = self.cast(normalized, prefix + "_fp16", TensorProto.FLOAT16)
        return self.cast(narrow, prefix + "_rounded_fp32", TensorProto.FLOAT)

    def swish(self, value, prefix):
        sigmoid = self.node("Sigmoid", [value], prefix + "_sigmoid")
        return self.node("Mul", [value, sigmoid], prefix + "_swish")


def delta_model(weights, layer, config, *, outer_product="matmul", fuse_input_norm=False):
    if config["layer_types"][layer] != "linear_attention":
        raise ValueError("compiled mixer export requires a DeltaNet layer")
    hidden = config["hidden_size"]
    heads, key_dim, value_dim = (
        config[name]
        for name in ("linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim")
    )
    if config["linear_num_key_heads"] != heads:
        raise ValueError("DeltaNet export currently requires equal key/value head counts")
    if outer_product not in ("matmul", "multiply"):
        raise ValueError("unknown DeltaNet outer product implementation")
    channels, width, convolution = (
        heads * (2 * key_dim + value_dim),
        heads * value_dim,
        config["linear_conv_kernel_dim"],
    )
    graph = DeltaGraph(weights)
    prefix = f"model.layers.{layer}.linear_attn."
    incoming = graph.port("hidden", TensorProto.FLOAT16, [1, hidden])
    if fuse_input_norm:
        # Match GraphBuilder::rms_norm, including its FP32 weight offset and
        # final FP16 boundary before any projection reads the normalized input.
        wide = graph.cast(incoming, "input_norm_fp32", TensorProto.FLOAT)
        squares = graph.node("Mul", [wide, wide], "input_norm_squares")
        variance = graph.node("ReduceMean", [squares], "input_norm_variance", axes=[-1], keepdims=1)
        regularized = graph.node(
            "Add",
            [
                variance,
                graph.constant("input_norm_epsilon", np.array(config["rms_norm_eps"], "f4")),
            ],
            "input_norm_regularized",
        )
        inverse = graph.node(
            "Reciprocal",
            [graph.node("Sqrt", [regularized], "input_norm_root")],
            "input_norm_inverse",
        )
        normalized = graph.node("Mul", [wide, inverse], "input_norm_normalized")
        scale = graph.cast(
            graph.weight("input_norm_weight", f"model.layers.{layer}.input_layernorm.weight"),
            "input_norm_scale_fp32",
            TensorProto.FLOAT,
        )
        scale = graph.node(
            "Add",
            [scale, graph.constant("input_norm_one", np.array(1, "f4"))],
            "input_norm_shifted_scale",
        )
        incoming = graph.cast(
            graph.node("Mul", [normalized, scale], "input_norm_scaled"),
            "input_normalized",
            TensorProto.FLOAT16,
        )
    state = graph.port("past_recurrent", TensorProto.FLOAT, [heads, key_dim, value_dim])
    history = graph.port("past_convolution", TensorProto.FLOAT16, [channels, convolution])
    packed, gate, a, b = [
        graph.project(incoming, part, prefix + part, False)
        for part in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
    ]
    retained = graph.slice(history, "retained_history", 1, convolution)
    packed_column = graph.reshape(packed, "packed_column", [channels, 1])
    updated_history = graph.node("Concat", [retained, packed_column], "present_convolution", axis=1)
    kernel = graph.reshape(
        graph.weight("convolution_kernel", prefix + "conv1d.weight"),
        "flat_kernel",
        [channels, convolution],
    )
    products = graph.node(
        "Mul",
        [
            graph.cast(updated_history, "history_fp32", TensorProto.FLOAT),
            graph.cast(kernel, "kernel_fp32", TensorProto.FLOAT),
        ],
        "convolution_products_fp32",
    )
    reduced = graph.node("ReduceSum", [products], "convolution_sum_fp32", axes=[1], keepdims=0)
    convolved = graph.swish(
        graph.cast(reduced, "convolution_fp16", TensorProto.FLOAT16), "convolution"
    )
    convolved = graph.reshape(convolved, "convolved", [1, channels])
    query = graph.reshape(
        graph.slice(convolved, "query_slice", 0, heads * key_dim), "query", [heads, 1, key_dim]
    )
    key = graph.reshape(
        graph.slice(convolved, "key_slice", heads * key_dim, 2 * heads * key_dim),
        "key",
        [heads, 1, key_dim],
    )
    value = graph.reshape(
        graph.slice(convolved, "value_slice", 2 * heads * key_dim, channels),
        "value",
        [heads, 1, value_dim],
    )
    query = graph.normalize(query, "query_norm", 1e-6, False)
    key = graph.normalize(key, "key_norm", 1e-6, False)
    query = graph.node(
        "Mul",
        [query, graph.constant("query_scale", np.array(key_dim**-0.5, "f4"))],
        "scaled_query_fp32",
    )
    value = graph.cast(value, "value_fp32", TensorProto.FLOAT)
    bias = graph.cast(
        graph.weight("dt_bias", prefix + "dt_bias"), "dt_bias_fp32", TensorProto.FLOAT
    )
    time = graph.node(
        "Softplus",
        [graph.node("Add", [graph.cast(a, "a_fp32", TensorProto.FLOAT), bias], "biased_time_fp32")],
        "time_fp32",
    )
    rate = graph.node("Exp", [graph.weight("A_log", prefix + "A_log")], "rate_fp32")
    rate = graph.node(
        "Mul", [rate, graph.constant("negative_one", np.array(-1, "f4"))], "negative_rate_fp32"
    )
    decay = graph.node("Exp", [graph.node("Mul", [time, rate], "scaled_time_fp32")], "decay_fp32")
    decay = graph.reshape(decay, "decay_heads", [heads, 1, 1])
    beta = graph.cast(graph.node("Sigmoid", [b], "beta_fp16"), "beta_fp32", TensorProto.FLOAT)
    beta = graph.reshape(beta, "beta_heads", [heads, 1, 1])
    decayed = graph.node("Mul", [state, decay], "decayed_state_fp32")
    prediction = graph.node("MatMul", [key, decayed], "prediction_fp32")
    error = graph.node("Sub", [value, prediction], "error_fp32")
    correction = graph.node("Mul", [error, beta], "correction_fp32")
    transposed_key = graph.node("Transpose", [key], "transposed_key", perm=[0, 2, 1])
    update = graph.node(
        "MatMul" if outer_product == "matmul" else "Mul",
        [transposed_key, correction],
        "update_fp32",
    )
    updated_state = graph.node("Add", [decayed, update], "present_recurrent")
    core = graph.node("MatMul", [query, updated_state], "core_fp32")
    core = graph.reshape(core, "flat_core", [heads, value_dim])
    normalized = graph.normalize(core, "output_norm", config["rms_norm_eps"])
    scale = graph.weight("output_norm_weight", prefix + "norm.weight")
    scaled = graph.node("Mul", [normalized, scale], "scaled_core_fp32")
    gate = graph.cast(
        graph.reshape(gate, "gate_heads", [heads, value_dim]), "gate_fp32", TensorProto.FLOAT
    )
    gated = graph.node("Mul", [scaled, graph.swish(gate, "gate")], "gated_fp32")
    gated = graph.reshape(
        graph.cast(gated, "gated_fp16", TensorProto.FLOAT16), "flat_gated", [1, width]
    )
    output = graph.project(gated, "out_proj", prefix + "out_proj", False)
    graph.port(output, TensorProto.FLOAT16, [1, hidden], True)
    graph.port(updated_state, TensorProto.FLOAT, [heads, key_dim, value_dim], True)
    graph.port(updated_history, TensorProto.FLOAT16, [channels, convolution], True)
    return graph


def feed_forward_model(weights, layer, config):
    """Post-attention residual, FP32 RMSNorm and FP16 SwiGLU without runtime FCL packing."""
    graph = DeltaGraph(weights)
    hidden = config["hidden_size"]
    prefix = f"model.layers.{layer}."
    residual = graph.port("residual", TensorProto.FLOAT16, [1, hidden])
    mixer = graph.port("mixer", TensorProto.FLOAT16, [1, hidden])
    summed = graph.node("Add", [residual, mixer], "residual_sum")
    wide = graph.cast(summed, "residual_fp32", TensorProto.FLOAT)
    squares = graph.node("Mul", [wide, wide], "squares_fp32")
    variance = graph.node("ReduceMean", [squares], "variance_fp32", axes=[-1], keepdims=1)
    regularized = graph.node(
        "Add",
        [variance, graph.constant("epsilon", np.array(config["rms_norm_eps"], "f4"))],
        "regularized_fp32",
    )
    normalized = graph.node(
        "Div", [wide, graph.node("Sqrt", [regularized], "root_fp32")], "normalized_fp32"
    )
    scale = graph.node(
        "Add",
        [
            graph.weight("norm_weight", prefix + "post_attention_layernorm.weight", True),
            graph.constant("one", np.array(1, "f4")),
        ],
        "scale_fp32",
    )
    normalized = graph.cast(
        graph.node("Mul", [normalized, scale], "scaled_fp32"), "normalized", TensorProto.FLOAT16
    )
    gate = graph.project(normalized, "gate", prefix + "mlp.gate_proj", False)
    up = graph.project(normalized, "up", prefix + "mlp.up_proj", False)
    activated = graph.swish(gate, "gate_activation")
    product = graph.node("Mul", [activated, up], "gated")
    projected = graph.project(product, "down", prefix + "mlp.down_proj", False)
    output = graph.node("Add", [summed, projected], "output")
    graph.port(output, TensorProto.FLOAT16, [1, hidden], True)
    return graph
