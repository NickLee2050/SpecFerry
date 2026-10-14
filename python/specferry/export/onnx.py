"""Small ONNX builder preserving explicit precision boundaries and raw weight identities."""

import hashlib

import numpy as np
import onnx
from onnx import helper, numpy_helper


class OnnxGraph:
    def __init__(self, weights):
        self.weights = weights
        self.nodes, self.constants, self.inputs, self.outputs = [], [], [], []
        self.identities = {}

    def node(self, op, inputs, name, **attributes):
        self.nodes.append(helper.make_node(op, inputs, [name], name=name, **attributes))
        return name

    def constant(self, name, value):
        self.constants.append(numpy_helper.from_array(np.ascontiguousarray(value), name))
        return name

    def weight(self, name, source, wide=False):
        value = self.weights[source]
        value = value.numpy() if hasattr(value, "numpy") else value
        self.identities[source] = hashlib.sha256(value.tobytes()).hexdigest()
        return self.constant(name, value.astype("f4") if wide else value)

    def port(self, name, dtype, shape, output=False):
        (self.outputs if output else self.inputs).append(
            helper.make_tensor_value_info(name, dtype, shape)
        )
        return name

    def reshape(self, value, name, shape):
        return self.node(
            "Reshape", [value, self.constant(name + "_shape", np.array(shape, "i8"))], name
        )

    def project(self, value, prefix, source, bias=True):
        product = self.node(
            "Gemm",
            [value, self.weight(prefix + "_weight", source + ".weight")],
            prefix + "_product",
            transB=1,
            alpha=1.0,
            beta=0.0,
        )
        if not bias:
            return product
        return self.node(
            "Add", [product, self.weight(prefix + "_bias", source + ".bias")], prefix + "_biased"
        )

    def model(self, name="SpecFerry fixed inference"):
        result = helper.make_model(
            helper.make_graph(self.nodes, name, self.inputs, self.outputs, self.constants),
            producer_name="SpecFerry",
            opset_imports=[helper.make_opsetid("", 11)],
            ir_version=7,
        )
        onnx.checker.check_model(result, full_check=True)
        return result
