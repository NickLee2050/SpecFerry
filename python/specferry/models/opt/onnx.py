"""Fixed OPT decoder/full-model exports with explicit FP16 and FP32 boundaries."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from specferry.export.onnx import OnnxGraph


class Graph(OnnxGraph):
    def __init__(self, weights, capacity):
        if capacity not in (64, 128):
            raise ValueError("compiled OPT export supports fixed C64/C128")
        super().__init__(weights)
        self.capacity = capacity

    def norm(self, value, prefix, source):
        value = self.node("Cast", [value], prefix + "_fp32", to=TensorProto.FLOAT)
        mean = self.node("ReduceMean", [value], prefix + "_mean", axes=[1], keepdims=1)
        centered = self.node("Sub", [value, mean], prefix + "_centered")
        squares = self.node("Mul", [centered, centered], prefix + "_squares")
        variance = self.node("ReduceMean", [squares], prefix + "_variance", axes=[1], keepdims=1)
        epsilon = self.constant(prefix + "_epsilon", np.array(1e-5, "f4"))
        regularized = self.node("Add", [variance, epsilon], prefix + "_regularized")
        root = self.node("Sqrt", [regularized], prefix + "_denominator")
        normalized = self.node("Div", [centered, root], prefix + "_normalized")
        scaled = self.node(
            "Mul",
            [normalized, self.weight(prefix + "_weight", source + ".weight", True)],
            prefix + "_scaled",
        )
        affine = self.node(
            "Add",
            [scaled, self.weight(prefix + "_bias", source + ".bias", True)],
            prefix + "_shifted",
        )
        return self.node("Cast", [affine], prefix + "_output", to=TensorProto.FLOAT16)

    def decoder(self, layer, hidden, indices, mask, *, names=""):
        p, source = names, f"decoder.layers.{layer}."
        past = [
            self.port(p + "past_" + part, TensorProto.FLOAT, [16, self.capacity, 64])
            for part in ("key", "value")
        ]
        query = self.project(hidden, p + "self_attn.q_proj", source + "self_attn.q_proj")
        query = self.node(
            "Mul",
            [query, self.constant(p + "query_scale", np.array(0.125, "f2"))],
            p + "scaled_query",
        )
        query = self.reshape(query, p + "query", [16, 1, 64])
        states, narrow = [], []
        for index, part in enumerate(("key", "value")):
            projection = "k_proj" if part == "key" else "v_proj"
            value = self.project(
                hidden, p + "self_attn." + projection, source + "self_attn." + projection
            )
            column = self.reshape(value, p + part + "_column", [16, 1, 64])
            update = self.reshape(column, p + part + "_update", [16, 64])
            wide = self.node("Cast", [update], p + part + "_update_fp32", to=TensorProto.FLOAT)
            state = self.node("ScatterND", [past[index], indices, wide], p + "present_" + part)
            states.append(state)
            narrow.append(
                self.node(
                    "Cast", [state], p + "present_" + part + "_attention", to=TensorProto.FLOAT16
                )
            )
        key = self.node("Transpose", [narrow[0]], p + "transposed_key", perm=[0, 2, 1])
        scores = self.node("MatMul", [query, key], p + "scores")
        scores = self.reshape(scores, p + "flat_scores", [16, self.capacity])
        scores = self.node("Cast", [scores], p + "scores_fp32", to=TensorProto.FLOAT)
        scores = self.node("Add", [scores, mask], p + "masked_scores")
        probability = self.node("Softmax", [scores], p + "probability_fp32", axis=1)
        probability = self.node(
            "Cast", [probability], p + "probability_fp16", to=TensorProto.FLOAT16
        )
        probability = self.reshape(probability, p + "probability", [16, 1, self.capacity])
        attended = self.node("MatMul", [probability, narrow[1]], p + "attended_heads")
        attended = self.reshape(attended, p + "attended", [1, 1024])
        projected = self.project(attended, p + "self_attn.out_proj", source + "self_attn.out_proj")
        residual = self.node("Add", [hidden, projected], p + "attention_residual")
        norm = self.norm(residual, p + "self_attn_layer_norm", source + "self_attn_layer_norm")
        fc1 = self.project(norm, p + "fc1", source + "fc1")
        activation = self.node("Relu", [fc1], p + "activation")
        fc2 = self.project(activation, p + "fc2", source + "fc2")
        residual = self.node("Add", [norm, fc2], p + "ffn_residual")
        output = self.norm(residual, p + "final_layer_norm", source + "final_layer_norm")
        return self.node("Identity", [output], p + "hidden_out"), states

    def model(self):
        result = helper.make_model(
            helper.make_graph(
                self.nodes, "OPT fixed inference", self.inputs, self.outputs, self.constants
            ),
            producer_name="SpecFerry OPT fixed inference",
            opset_imports=[helper.make_opsetid("", 11)],
            ir_version=7,
        )
        onnx.checker.check_model(result, full_check=True)
        return result


def decoder_model(weights, layer, capacity):
    graph = Graph(weights, capacity)
    hidden = graph.port("hidden", TensorProto.FLOAT16, [1, 1024])
    indices = graph.port("kv_indices", TensorProto.INT64, [16, 2])
    mask = graph.port("attention_mask", TensorProto.FLOAT, [1, capacity])
    output, states = graph.decoder(layer, hidden, indices, mask)
    roles = ("hidden", "past_key", "past_value", "kv_indices", "attention_mask")
    graph.inputs.sort(key=lambda port: roles.index(port.name))
    graph.port(output, TensorProto.FLOAT16, [1, 1024], True)
    for state in states:
        graph.port(state, TensorProto.FLOAT, [16, capacity, 64], True)
    return graph


def full_model(weights, capacity):
    graph = Graph(weights, capacity)
    token = graph.port("token", TensorProto.INT64, [1, 1])
    position = graph.port("position", TensorProto.INT64, [1, 1])
    indices = graph.port("kv_indices", TensorProto.INT64, [16, 2])
    mask = graph.port("attention_mask", TensorProto.FLOAT, [1, capacity])
    table = graph.weight("decoder.embed_tokens.weight", "decoder.embed_tokens.weight")
    embedded = graph.node("GatherND", [table, token], "token_embedding")
    projected = graph.project(embedded, "input_projection", "decoder.project_in", False)
    offset = graph.constant("position_offset", np.array([2], "i8"))
    position = graph.node("Add", [position, offset], "learned_position_index")
    position = graph.node(
        "GatherND",
        [
            graph.weight("decoder.embed_positions.weight", "decoder.embed_positions.weight"),
            position,
        ],
        "learned_position",
    )
    hidden = graph.node("Add", [projected, position], "embedded_hidden")
    states = []
    for layer in range(24):
        hidden, updated = graph.decoder(layer, hidden, indices, mask, names=f"layer{layer}.")
        states.extend(updated)
    hidden = graph.project(hidden, "projected_hidden", "decoder.project_out", False)
    logits = graph.node("Gemm", [hidden, table], "logits", transB=1, alpha=1.0, beta=0.0)
    scores = graph.node("Cast", [logits], "selection_scores", to=TensorProto.FLOAT)
    selected = graph.node("ArgMax", [scores], "selected_int64", axis=1, keepdims=0)
    selected = graph.node("Cast", [selected], "selected", to=TensorProto.INT32)
    graph.port(logits, TensorProto.FLOAT16, [1, 50272], True)
    graph.port(selected, TensorProto.INT32, [1], True)
    for state in states:
        graph.port(state, TensorProto.FLOAT, [16, capacity, 64], True)
    return graph
