#!/usr/bin/env python3
"""Export per-layer OPT projections and the shared head through the isolated compiler."""

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from specferry.data.checkpoint import sha256
from specferry.export.compiled import compile_model, public_port_order
from specferry.models.opt.reference import capture_layers, load_model
from specferry.validation.device import write_json


def prepare(kind, directory, weights, captured, hidden, layers=(0,)):
    directory.mkdir(parents=True, exist_ok=False)
    nodes, inputs, outputs, constants, references, source_hashes = [], [], [], [], [], {}
    layers = (0,) if kind == "head" else tuple(layers)
    count = len(layers)

    def constant(name, value):
        constants.append(numpy_helper.from_array(value, name))

    def projection(value, matrix_name, bias_name, name):
        matrix = weights[matrix_name].numpy()
        constant(name + "_weight", matrix)
        source_hashes[matrix_name] = hashlib.sha256(matrix.tobytes()).hexdigest()
        nodes.append(
            helper.make_node(
                "Gemm",
                [value[0], name + "_weight"],
                [name + "_product"],
                name=name + "_product",
                transB=1,
                alpha=1.0,
                beta=0.0,
            )
        )
        actual = (value[1].astype("f4") @ matrix.astype("f4").T).astype("f2")
        if bias_name:
            bias = weights[bias_name].numpy().reshape(1, -1)
            constant(name + "_bias", bias)
            source_hashes[bias_name] = hashlib.sha256(bias.tobytes()).hexdigest()
            nodes.append(
                helper.make_node("Add", [name + "_product", name + "_bias"], [name], name=name)
            )
            actual = (actual + bias).astype("f2")
            return name, actual
        return name + "_product", actual

    for index, layer in enumerate(layers):
        array = (
            hidden
            if kind == "head"
            else captured[f"layer.{layer}." + ("input" if kind == "qkv" else "attended")][
                :, -1, :
            ].numpy()
        )
        array = np.ascontiguousarray(array, dtype="f2")
        array.tofile(directory / f"input{index}.bin")
        input_name = f"input{index}"
        inputs.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT16, [1, 1024]))
        if kind == "head":
            projected = projection(
                (input_name, array), "decoder.project_out.weight", None, "project_out"
            )
            values = [projection(projected, "decoder.embed_tokens.weight", None, "logits")]
        else:
            names = ("q_proj", "k_proj", "v_proj") if kind == "qkv" else ("out_proj",)
            values = []
            for name in names:
                prefix = f"decoder.layers.{layer}.self_attn.{name}"
                value = projection(
                    (input_name, array),
                    prefix + ".weight",
                    prefix + ".bias",
                    f"layer{index}_{name}",
                )
                if name == "q_proj":
                    constant(f"scale{index}", np.array([0.125], dtype="f2"))
                    scaled = f"layer{index}_q_scaled"
                    nodes.append(
                        helper.make_node("Mul", [value[0], f"scale{index}"], [scaled], name=scaled)
                    )
                    value = scaled, (value[1] * np.float16(0.125)).astype("f2")
                values.append(value)
        for name, expected in values:
            expected.tofile(directory / f"expected{len(references)}.bin")
            references.append({"name": name, "shape": list(expected.shape)})
            outputs.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT16, list(expected.shape))
            )
    model = helper.make_model(
        helper.make_graph(nodes, kind, inputs, outputs, constants),
        producer_name="SpecFerry OPT projection comparison",
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=7,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, directory / "model.onnx")
    source_bytes = sum(value.nbytes for value in (numpy_helper.to_array(c) for c in constants))
    write_json(
        directory / "source.json",
        {
            "kind": kind,
            "count": count,
            "outputs": references,
            "weight_sha256": source_hashes,
            "source_const_bytes": source_bytes,
            "onnx_sha256": sha256(directory / "model.onnx"),
        },
    )


def finish(directory):
    directory = directory.resolve()
    source = json.loads((directory / "source.json").read_text())
    exported = directory / "export_model_nbg_unify"
    binary = exported / "network_binary.nb"
    metadata = json.loads((exported / "nbg_meta.json").read_text())
    actual = list(metadata["Outputs"].values())
    input_names = [value["name"] for value in metadata["Inputs"].values()]
    if input_names != [f"input{index}" for index in range(source["count"])]:
        raise ValueError("compiled input ordering differs from the collection")
    output_order = []
    for expected in source["outputs"]:
        matches = [
            index
            for index, value in enumerate(actual)
            if value["name"].endswith(expected["name"] + "/out0")
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot identify compiled output: {expected['name']}")
        output_order.append(matches[0])
    if len(actual) != len(source["outputs"]) or any(
        value["shape"] != expected["shape"] or value["dtype"] != "float16"
        for value, expected in zip(
            (actual[index] for index in output_order), source["outputs"], strict=True
        )
    ):
        raise ValueError("compiled projection ports differ from exported outputs")
    # The separated raw database includes padded constants; take the larger bound.
    constant_bytes = max(
        source["source_const_bytes"],
        (directory / "model.data").stat().st_size,
        binary.stat().st_size,
    )
    header = (directory / "export_model/vnn_network.h").read_text()
    version = [
        int(re.search(rf"#define\s+VNN_VERSION_{part}\s+\(?([0-9]+)", header)[1])
        for part in ("MAJOR", "MINOR", "PATCH")
    ]
    write_json(
        directory / "artifact.json",
        {
            "nbg_sha256": sha256(binary),
            "metadata": metadata,
            "constant_bytes": constant_bytes,
            "graph_version": version,
            "output_order": output_order,
            "public_output_order": public_port_order(directory, "output"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True, help="fixed token-list text file")
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, metadata = load_model(args.checkpoint)
    captured = {}
    tokens = [int(value) for value in args.request.read_text().split()]
    handle = model.model.decoder.project_out.register_forward_pre_hook(
        lambda module, args: captured.update(head_input=args[0].detach().clone())
    )
    with (
        torch.inference_mode(),
        capture_layers(dict(enumerate(model.model.decoder.layers)), captured),
    ):
        model(torch.tensor([tokens]))
    handle.remove()
    hidden = captured["head_input"][:, -1, :].numpy()
    weights = {
        name.removeprefix("model."): value
        for name, value in model.state_dict().items()
        if name.startswith("model.")
    }
    jobs = []
    for layer in range(24):
        for kind in ("qkv", "out"):
            directory = args.output / f"layer{layer}" / kind
            prepare(kind, directory, weights, captured, hidden, (layer,))
            jobs.append(directory)
    directory = args.output / "head"
    prepare("head", directory, weights, captured, hidden)
    jobs.append(directory)
    write_json(
        args.output / "reference.json",
        {
            "metadata": metadata,
            "tokens": tokens,
            "scope": "last prompt column; all 24 layers and head",
        },
    )
    del model, weights
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(
            pool.map(
                lambda directory: compile_model(args.rootfs.resolve(), directory),
                jobs,
            )
        )
    for directory in jobs:
        finish(directory)


if __name__ == "__main__":
    main()
