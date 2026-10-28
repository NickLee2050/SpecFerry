"""Fixed OPT decoder/full-model exports with explicit FP16 and FP32 boundaries."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from specferry.export.onnx import OnnxGraph


def select_cache_updates(model):
    """Replace fixed OPT ScatterND updates in place without changing public ports.

    The runtime supplies one index per head, in head order: [head, position].
    Broadcasting a selection preserves FP32 values (including untouched slots),
    unlike adding sparse updates, which would require an initially empty slot.
    """
    ports = {port.name: port for port in model.graph.input}
    nodes, constants, masks = [], [], {}
    count = 0
    for node in model.graph.node:
        if node.op_type != "ScatterND":
            nodes.append(node)
            continue
        past, indices, update = node.input
        port = ports.get(past)
        if port is None or not past.endswith(("past_key", "past_value")):
            raise ValueError("selection update requires a public OPT cache input")
        shape = [dim.dim_value for dim in port.type.tensor_type.shape.dim]
        index_port = ports.get(indices)
        index_shape = (
            [dim.dim_value for dim in index_port.type.tensor_type.shape.dim]
            if index_port is not None
            else []
        )
        if (
            len(shape) != 3
            or min(shape) <= 0
            or port.type.tensor_type.elem_type != TensorProto.FLOAT
            or index_shape != [shape[0], 2]
        ):
            raise ValueError("selection update requires FP32 [heads, capacity, width] cache")
        key = (indices, tuple(shape))
        if key not in masks:
            prefix = "cache_selection_" + str(len(masks))
            constants.extend(
                numpy_helper.from_array(value, prefix + suffix)
                for suffix, value in (
                    ("_start", np.array([1], "i8")),
                    ("_end", np.array([2], "i8")),
                    ("_axis", np.array([1], "i8")),
                    ("_slots", np.arange(shape[1], dtype="i8").reshape(1, -1)),
                    ("_shape", np.array([shape[0], shape[1], 1], "i8")),
                )
            )
            nodes.extend(
                (
                    # This converter requires a newer SDK for Gather; Slice has
                    # the same fixed-column semantics on the pinned SDK.
                    helper.make_node(
                        "Slice",
                        [indices, prefix + "_start", prefix + "_end", prefix + "_axis"],
                        [prefix + "_positions"],
                    ),
                    helper.make_node(
                        "Equal", [prefix + "_positions", prefix + "_slots"], [prefix + "_mask"]
                    ),
                    helper.make_node(
                        "Reshape", [prefix + "_mask", prefix + "_shape"], [prefix + "_expanded"]
                    ),
                )
            )
            masks[key] = prefix + "_expanded"
        expanded = node.output[0] + "_update_column"
        nodes.extend(
            (
                helper.make_node("Unsqueeze", [update], [expanded], axes=[1]),
                helper.make_node("Where", [masks[key], expanded, past], node.output),
            )
        )
        count += 1
    if not count:
        raise ValueError("no OPT ScatterND cache updates found")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(constants)
    onnx.checker.check_model(model, full_check=True)
    return model


def narrow_cache_selection(model):
    """Narrow old KV before selection and publish only the FP32 current slots.

    Canonical input state remains FP32. The runtime must commit output slot zero
    to the requested destination position, preserving every untouched input bit.
    Only the fixed Select-based decode graph is supported, never M16 prefill.
    """
    producers = {value: node for node in model.graph.node for value in node.output}
    replacements, removed = {}, set()
    for port in model.graph.output:
        if not port.name.endswith(("present_key", "present_value")):
            continue
        state = port.name
        selector = producers[state]
        attention = producers.get(state + "_attention")
        shape = [dim.dim_value for dim in port.type.tensor_type.shape.dim]
        if (
            selector.op_type != "Where"
            or attention is None
            or attention.op_type != "Cast"
            or list(attention.input) != [state]
            or port.type.tensor_type.elem_type != TensorProto.FLOAT
            or len(shape) != 3
            or shape[0] != 16
            or shape[2] != 64
        ):
            raise ValueError("narrow selection requires fixed FP32 OPT decode cache outputs")
        condition, expanded, old = selector.input
        column = producers[expanded]
        wide = producers[column.input[0]]
        if (
            column.op_type != "Unsqueeze"
            or wide.op_type != "Cast"
            or not any(a.name == "to" and a.i == TensorProto.FLOAT16 for a in attention.attribute)
            or not any(a.name == "to" and a.i == TensorProto.FLOAT for a in wide.attribute)
        ):
            raise ValueError("narrow selection requires explicit FP16-to-FP32 current slots")
        current = state.rsplit("present_", 1)[0] + "current_" + state.rsplit("_", 1)[1]
        narrow_old, narrow_column = state + "_old_fp16", state + "_column_fp16"
        replacements[state] = [
            helper.make_node("Cast", [old], [narrow_old], to=TensorProto.FLOAT16),
            helper.make_node("Unsqueeze", [wide.input[0]], [narrow_column], axes=[1]),
            helper.make_node("Where", [condition, narrow_column, narrow_old], attention.output),
            helper.make_node("Identity", [expanded], [current]),
        ]
        removed.add(attention.output[0])
        port.CopyFrom(
            helper.make_tensor_value_info(current, TensorProto.FLOAT, [shape[0], 1, shape[2]])
        )
    if not replacements:
        raise ValueError("no public OPT Select cache outputs found")
    nodes = []
    for node in model.graph.node:
        if node.output[0] in replacements:
            nodes.extend(replacements[node.output[0]])
        elif node.output[0] not in removed:
            nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    return model


class Graph(OnnxGraph):
    def __init__(self, weights, capacity):
        if capacity not in (64, 128, 256):
            raise ValueError("compiled OPT export supports fixed C64/C128/C256")
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


def prefill_decoder_model(weights, layer, capacity=64):
    """Compile sixteen prompt tokens with an aligned, FP32 cache-block update.

    The caller supplies a causal mask and a block index in [0, capacity / 16).
    All sixteen rows are real input tokens; padded final blocks are unsupported.
    """
    graph = Graph(weights, capacity)
    hidden = graph.port("hidden", TensorProto.FLOAT16, [16, 1024])
    past = [
        graph.port("past_" + part, TensorProto.FLOAT, [16, capacity, 64])
        for part in ("key", "value")
    ]
    index = graph.port("block_index", TensorProto.INT64, [1])
    index = graph.reshape(index, "block_index_broadcast", [1, 1, 1, 1])
    mask = graph.port("attention_mask", TensorProto.FLOAT, [16, capacity])
    blocks = graph.constant(
        "cache_blocks", np.arange(capacity // 16, dtype="i8").reshape(1, -1, 1, 1)
    )
    selected = graph.node("Equal", [index, blocks], "selected_block")
    source = f"decoder.layers.{layer}."
    projections = {}
    for part in ("q", "k", "v"):
        value = graph.project(hidden, part + "_proj", source + "self_attn." + part + "_proj")
        if part == "q":
            value = graph.node(
                "Mul", [value, graph.constant("query_scale", np.array(0.125, "f2"))], "scaled_query"
            )
        value = graph.reshape(value, part + "_rows", [16, 16, 64])
        projections[part] = graph.node("Transpose", [value], part + "_heads", perm=[1, 0, 2])
    states, narrow = [], []
    for ordinal, (part, projection) in enumerate((("key", "k"), ("value", "v"))):
        old = graph.reshape(past[ordinal], part + "_blocks", [16, capacity // 16, 16, 64])
        update = graph.node(
            "Cast", [projections[projection]], part + "_update_fp32", to=TensorProto.FLOAT
        )
        update = graph.reshape(update, part + "_new_block", [16, 1, 16, 64])
        state = graph.node("Where", [selected, update, old], part + "_updated_blocks")
        state = graph.reshape(state, "present_" + part, [16, capacity, 64])
        states.append(state)
        narrow.append(graph.node("Cast", [state], part + "_attention", to=TensorProto.FLOAT16))
    key = graph.node("Transpose", [narrow[0]], "transposed_key", perm=[0, 2, 1])
    scores = graph.node("MatMul", [projections["q"], key], "scores")
    scores = graph.node("Cast", [scores], "scores_fp32", to=TensorProto.FLOAT)
    scores = graph.node("Add", [scores, mask], "masked_scores")
    scores = graph.reshape(scores, "flat_scores", [256, capacity])
    probability = graph.node("Softmax", [scores], "probability_fp32", axis=1)
    probability = graph.node("Cast", [probability], "probability_fp16", to=TensorProto.FLOAT16)
    probability = graph.reshape(probability, "probability", [16, 16, capacity])
    attended = graph.node("MatMul", [probability, narrow[1]], "attended_heads")
    attended = graph.node("Transpose", [attended], "attended_rows", perm=[1, 0, 2])
    attended = graph.reshape(attended, "attended", [16, 1024])
    projected = graph.project(attended, "out_proj", source + "self_attn.out_proj")
    residual = graph.node("Add", [hidden, projected], "attention_residual")
    norm = graph.norm(residual, "attention_norm", source + "self_attn_layer_norm")
    fc1 = graph.project(norm, "fc1", source + "fc1")
    activation = graph.node("Relu", [fc1], "activation")
    fc2 = graph.project(activation, "fc2", source + "fc2")
    residual = graph.node("Add", [norm, fc2], "ffn_residual")
    normalized = graph.norm(residual, "final_norm", source + "final_layer_norm")
    output = graph.node("Identity", [normalized], "hidden_out")
    graph.port(output, TensorProto.FLOAT16, [16, 1024], True)
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
