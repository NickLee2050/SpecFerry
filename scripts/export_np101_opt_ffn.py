#!/usr/bin/env python3
"""Export or package source-bound OPT-350m M1 FFN/residual/normalization artifacts."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from specferry.data.checkpoint import sha256
from specferry.export.compiled import compile_model
from specferry.models.opt.checkpoint import load_checkpoint
from specferry.models.opt.compiled import PARTS, audit_ffn_export, package_ffn_exports
from specferry.validation.device import write_json


def build_layer(weights, layer, directory):
    """Keep separate FP16 product/bias/residual boundaries and FP32 normalization."""
    directory.mkdir(parents=True, exist_ok=False)
    prefix = f"decoder.layers.{layer}."
    constants, nodes, identities = [], [], {}
    names = {}
    for part in PARTS:
        name = ("decoder.layers.0." + part).replace(".", "_")
        array = weights[prefix + part].numpy()
        constants.append(numpy_helper.from_array(array, name))
        identities[prefix + part] = hashlib.sha256(array.tobytes()).hexdigest()
        names[part] = name
    constants.insert(4, numpy_helper.from_array(np.array([1e-5], dtype="f4"), "norm0_epsilon"))

    def node(op, inputs, output, **attributes):
        nodes.append(helper.make_node(op, inputs, [output], name=output, **attributes))
        return output

    value = "input0"
    for part in ("fc1", "fc2"):
        value = node(
            "Gemm",
            [value, names[part + ".weight"]],
            part + "_0_product",
            transB=1,
            alpha=1.0,
            beta=0.0,
        )
        value = node("Add", [value, names[part + ".bias"]], part + "_0_biased")
        if part == "fc1":
            value = node("Relu", [value], "activation0")
    residual = node("Add", ["input0", value], "residual0")
    wide = node("Cast", [residual], "norm0_fp32", to=TensorProto.FLOAT)
    mean = node("ReduceMean", [wide], "norm0_mean", axes=[1], keepdims=1)
    centered = node("Sub", [wide, mean], "norm0_centered")
    squares = node("Mul", [centered, centered], "norm0_squares")
    variance = node("ReduceMean", [squares], "norm0_variance", axes=[1], keepdims=1)
    regularized = node("Add", [variance, "norm0_epsilon"], "norm0_regularized")
    root = node("Sqrt", [regularized], "norm0_sqrt")
    inverse = node("Reciprocal", [root], "norm0_inverse")
    normalized = node("Mul", [centered, inverse], "norm0_normalized")
    scale = node(
        "Cast", [names["final_layer_norm.weight"]], "norm0_scale_fp32", to=TensorProto.FLOAT
    )
    bias = node("Cast", [names["final_layer_norm.bias"]], "norm0_bias_fp32", to=TensorProto.FLOAT)
    scaled = node("Mul", [normalized, scale], "norm0_scaled")
    affine = node("Add", [scaled, bias], "norm0_affine")
    node("Cast", [affine], "norm0_output", to=TensorProto.FLOAT16)
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "OPT FFN finish",
            [helper.make_tensor_value_info("input0", TensorProto.FLOAT16, [1, 1024])],
            [helper.make_tensor_value_info("norm0_output", TensorProto.FLOAT16, [1, 1024])],
            constants,
        ),
        producer_name="SpecFerry OPT FFN finish",
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=7,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, directory / "model.onnx")
    # Host-only export check; this is not hardware correctness or throughput evidence.
    x = np.linspace(-0.5, 0.5, 1024, dtype="f2").reshape(1, 1024)
    expected = x
    for part in ("fc1", "fc2"):
        expected = (
            expected.astype("f4") @ weights[prefix + part + ".weight"].numpy().astype("f4").T
        ).astype("f2")
        expected = (
            expected.astype("f4") + weights[prefix + part + ".bias"].numpy().astype("f4")
        ).astype("f2")
        if part == "fc1":
            expected = np.maximum(expected, np.float16(0))
    centered = (expected.astype("f4") + x.astype("f4")).astype("f2").astype("f4")
    centered -= centered.mean(axis=1, keepdims=True)
    expected = centered / np.sqrt(
        (centered * centered).mean(axis=1, keepdims=True) + np.float32(1e-5)
    )
    expected = (
        expected * weights[prefix + "final_layer_norm.weight"].numpy().astype("f4")
        + weights[prefix + "final_layer_norm.bias"].numpy().astype("f4")
    ).astype("f2")
    (actual,) = ReferenceEvaluator(model).run(None, {"input0": x})
    error = float(np.max(np.abs(actual.astype("f4") - expected.astype("f4"))))
    if not np.isfinite(actual).all() or error > 0.02:
        raise ValueError(f"FFN export CPU precision mismatch at layer {layer}: {error}")
    write_json(
        directory / "source.json",
        dict(
            kind="ffn-finish",
            layer=layer,
            weight_sha256=identities,
            onnx_sha256=sha256(directory / "model.onnx"),
            cpu_max_abs_error=error,
            constant_payload_bound=sum(len(t.raw_data) for t in constants) + 8192,
        ),
    )


def finish_layer(directory):
    audit = audit_ffn_export(directory)
    binary = directory / "export_model_nbg_unify/network_binary.nb"
    source = json.loads((directory / "source.json").read_text())
    write_json(
        directory / "artifact.json",
        dict(
            nbg_sha256=sha256(binary),
            constant_bytes=max(
                source["constant_payload_bound"],
                binary.stat().st_size,
                (directory / "model.data").stat().st_size,
            ),
            graph_version=audit["graph_version"],
            source=source,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--projection-artifacts", type=Path, help="all 24 per-layer QKV/out exports"
    )
    parser.add_argument(
        "--import-artifacts",
        type=Path,
        help="package existing exports after repeating source, precision and identity checks",
    )
    parser.add_argument(
        "--prefill-artifacts", type=Path, help="existing source-bound M16 FFN exports"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--head-artifacts",
        type=Path,
        help="optional previously compiled FP16 project_out/vocabulary head",
    )
    parser.add_argument("--rootfs", type=Path, help="device-free installed compiler image")
    args = parser.parse_args()
    if args.prefill_artifacts is not None and args.head_artifacts is None:
        parser.error("M16 prefill requires the single compiled head")
    if args.output.exists():
        parser.error("use a new output directory")
    artifacts = args.import_artifacts
    if artifacts is None:
        if args.checkpoint is None or args.rootfs is None:
            parser.error("fresh export requires --checkpoint and --rootfs")
        artifacts = args.output.resolve().with_name(args.output.name + "-compiler")
        if artifacts.exists():
            parser.error("use a new compiler evidence directory")
        weights, _ = load_checkpoint(args.checkpoint.resolve())
        for layer in range(24):
            build_layer(weights, layer, artifacts / f"layer{layer}")
        del weights

        def compile_layer(layer):
            directory = artifacts / f"layer{layer}"
            compile_model(args.rootfs.resolve(), directory, precision="float32")
            finish_layer(directory)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(compile_layer, range(24)))
    result = package_ffn_exports(
        artifacts.resolve(),
        args.deployment.resolve(),
        args.output.resolve(),
        Path("/usr/lib/ljmicro"),
        head_artifacts=args.head_artifacts.resolve() if args.head_artifacts else None,
        prefill_artifacts=args.prefill_artifacts.resolve() if args.prefill_artifacts else None,
        projection_artifacts=args.projection_artifacts.resolve()
        if args.projection_artifacts
        else None,
    )
    print("Packaged", len(result["layers"]), "source-bound FFN finish artifacts:", args.output)


if __name__ == "__main__":
    main()
