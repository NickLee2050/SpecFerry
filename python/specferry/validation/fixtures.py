"""Write small, inspectable graph fixtures with independent NumPy expectations."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DTYPES = {
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "I32": np.dtype("<i4"),
    "BOOL": np.dtype("u1"),
}


def array(value, dtype: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=DTYPES[dtype])
    if not np.isfinite(result).all():
        raise ValueError("fixture contains NaN/Inf or overflows its storage dtype")
    return result


@dataclass
class Fixture:
    name: str
    family: str
    steps: int = 2
    tensors: dict = field(default_factory=dict)
    nodes: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    feedback: tuple[str, str] | None = None
    reset_after: int = 0
    provenance: dict = field(default_factory=lambda: {"kind": "synthetic", "seed": 101})

    def tensor(self, name, value, dtype="F16", storage="mutable", initialize=False):
        if name in self.tensors:
            raise ValueError(f"duplicate tensor: {name}")
        data = array(value, dtype)
        self.tensors[name] = {
            "data": data,
            "dtype": dtype,
            "storage": storage,
            "initialize": initialize or storage == "constant",
        }
        return name

    def constant(self, name, value, dtype="F16"):
        return self.tensor(name, value, dtype, storage="constant")

    def input(self, name, values, dtype="F16"):
        if len(values) != self.steps:
            raise ValueError("input sequence length differs from step count")
        self.tensor(name, values[0], dtype)
        self.inputs[name] = [array(value, dtype) for value in values]
        if any(value.shape != self.inputs[name][0].shape for value in self.inputs[name]):
            raise ValueError("input shape changes between steps")
        return name

    def node(self, operation, inputs, output, shape, dtype="F16", parameters=()):
        if any(name not in self.tensors for name in inputs):
            raise ValueError("node consumes an undefined tensor")
        self.tensor(output, np.zeros(shape), dtype)
        self.nodes.append((operation, inputs, output, parameters))
        return output

    def output(self, name, values):
        dtype = self.tensors[name]["dtype"]
        self.expected[name] = [array(value, dtype) for value in values]
        if len(values) != self.steps or any(
            value.shape != self.tensors[name]["data"].shape for value in self.expected[name]
        ):
            raise ValueError("expected output shape/step count mismatch")

    def write(self, directory: Path) -> dict:
        directory.mkdir(parents=True, exist_ok=False)
        lines = ["specferry-np101-case 1", f"steps {self.steps}"]
        for name, tensor in self.tensors.items():
            data = tensor["data"]
            filename = f"{name}.initial.bin" if tensor["initialize"] else "-"
            if filename != "-":
                data.tofile(directory / filename)
            # NumPy is row-major; the SDK lists the contiguous dimension first.
            shape = ",".join(str(size) for size in reversed(data.shape))
            lines.append(f"tensor {name} {tensor['dtype']} {tensor['storage']} {shape} {filename}")
        for operation, inputs, output, parameters in self.nodes:
            encoded = ",".join(map(str, parameters)) or "-"
            lines.append(f"node {operation} {','.join(inputs)} {output} {encoded}")
        for name, values in self.inputs.items():
            filenames = []
            for step, value in enumerate(values):
                filename = f"{name}.input.{step}.bin"
                value.tofile(directory / filename)
                filenames.append(filename)
            lines.append(f"input {name} {','.join(filenames)}")
        for name, values in self.expected.items():
            lines.append(f"output {name}")
            for step, value in enumerate(values):
                value.tofile(directory / f"{name}.expected.{step}.bin")
        if self.feedback:
            lines.append(f"feedback {self.feedback[0]} {self.feedback[1]}")
        if self.reset_after:
            lines.append(f"reset_after {self.reset_after}")
        (directory / "graph.txt").write_text("\n".join(lines) + "\n")
        return {
            "name": self.name,
            "family": self.family,
            "steps": self.steps,
            "provenance": self.provenance,
            "operators": [node[0] for node in self.nodes],
            "tensors": {
                name: {
                    "dtype": tensor["dtype"],
                    "shape": list(tensor["data"].shape),
                    "storage": tensor["storage"],
                }
                for name, tensor in self.tensors.items()
            },
            "outputs": list(self.expected),
            "feedback": self.feedback,
        }
