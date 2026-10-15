#!/usr/bin/env python3
"""Compile cached Qwen blocks or a vocabulary prefix without exposing a board device."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import onnx

from specferry.data.checkpoint import sha256
from specferry.export.compiled import compile_model, compiled_ports
from specferry.models.qwen3_5.compiled import combine_packages, package_blocks
from specferry.models.qwen3_5.export import verify_export
from specferry.models.qwen3_5.onnx import delta_model, feed_forward_model, vocabulary_head_model
from specferry.validation.device import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--part", choices=("mixer", "feed-forward", "vocabulary-head"), default="mixer"
    )
    parser.add_argument("--vocabulary-rows", type=int, default=65536)
    parser.add_argument("--outer-product", choices=("matmul", "multiply"), default="matmul")
    parser.add_argument("--fuse-input-norm", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument(
        "--reuse-first-layer", type=Path, help="validated layer-zero compiler evidence"
    )
    parser.add_argument("--package-output", type=Path)
    parser.add_argument("--mixers-package", type=Path, help="combine accepted mixers with all FFNs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.part != "mixer" and (args.outer_product != "matmul" or args.fuse_input_norm):
        parser.error("outer product and input normalization options require --part mixer")
    if args.reuse_first_layer and (args.outer_product != "matmul" or args.fuse_input_norm):
        parser.error("changed mixer contracts cannot reuse an unchecked layer-zero export")
    if args.part == "vocabulary-head" and (args.all_layers or args.reuse_first_layer or args.layer):
        parser.error("vocabulary head is a single model-level graph without a decoder layer")
    if args.mixers_package and (
        args.part != "feed-forward" or not args.all_layers or not args.package_output
    ):
        parser.error("--mixers-package requires all feed-forward layers and --package-output")
    verify_export(args.deployment)
    manifest = json.loads((args.deployment / "deployment-manifest.json").read_text())
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    layers = (
        [
            i
            for i, kind in enumerate(manifest["text_config"]["layer_types"])
            if args.part == "feed-forward" or kind == "linear_attention"
        ]
        if args.all_layers
        else [args.layer]
    )
    if args.reuse_first_layer and (not args.all_layers or args.part != "mixer"):
        parser.error("reusing layer zero requires --all-layers")

    def export_layer(layer):
        if layer == 0 and args.reuse_first_layer:
            return args.reuse_first_layer.resolve()
        target = directory / f"layer{layer}" if args.all_layers else directory
        if args.all_layers:
            target.mkdir()
        weights = {}
        with (args.deployment / "weights.bin").open("rb") as stream:
            for record in manifest["tensors"]:
                if args.part == "vocabulary-head" and record["name"] == "model.embed_tokens.weight":
                    weights[record["name"]] = np.memmap(
                        args.deployment / "weights.bin",
                        dtype="<f2",
                        mode="r",
                        offset=record["offset"],
                        shape=record["shape"],
                    )
                elif args.part != "vocabulary-head" and (
                    (
                        record["name"].startswith(f"model.layers.{layer}.linear_attn.")
                        or (
                            args.fuse_input_norm
                            and record["name"] == f"model.layers.{layer}.input_layernorm.weight"
                        )
                    )
                    if args.part == "mixer"
                    else record["name"].startswith(f"model.layers.{layer}.mlp.")
                    or record["name"] == f"model.layers.{layer}.post_attention_layernorm.weight"
                ):
                    stream.seek(record["offset"])
                    weights[record["name"]] = np.frombuffer(
                        stream.read(record["bytes"]),
                        dtype={"F16": "<f2", "F32": "<f4"}[record["dtype"]],
                    ).reshape(record["shape"])
        if args.part == "vocabulary-head":
            graph = vocabulary_head_model(weights, args.vocabulary_rows)
            graph.identities = {
                row["name"]: row["sha256"]
                for row in manifest["tensors"]
                if row["name"] == "model.embed_tokens.weight"
            }
        else:
            if args.part == "mixer":
                graph = delta_model(
                    weights,
                    layer,
                    manifest["text_config"],
                    outer_product=args.outer_product,
                    fuse_input_norm=args.fuse_input_norm,
                )
            else:
                graph = feed_forward_model(weights, layer, manifest["text_config"])
        model = graph.model("Qwen single-step " + args.part)
        onnx.save(model, target / "model.onnx")
        write_json(
            target / "source.json",
            dict(
                layer=layer,
                part=args.part,
                weight_sha256=graph.identities,
                deployment_sha256=sha256(args.deployment / "deployment-manifest.json"),
                onnx_sha256=sha256(target / "model.onnx"),
                constant_bytes=sum(len(t.raw_data) for t in graph.constants),
                vocabulary_rows=args.vocabulary_rows if args.part == "vocabulary-head" else 0,
                outer_product=args.outer_product if args.part == "mixer" else None,
                includes_input_norm=args.fuse_input_norm,
            ),
        )
        compile_model(args.rootfs.resolve(), target, precision="float32")
        binary = target / "export_model_nbg_unify/network_binary.nb"
        write_json(
            target / "artifact.json",
            dict(
                nbg_sha256=sha256(binary),
                constant_bytes=max(binary.stat().st_size, (target / "model.data").stat().st_size),
                # This head binary uses the generated inference_with_nbg/public
                # port order, not the wrapper's original reverse kernel order.
                **compiled_ports(target, model, saved_public_graph=args.part == "vocabulary-head"),
            ),
        )
        return target

    with ThreadPoolExecutor(max_workers=2) as pool:
        graphs = list(pool.map(export_layer, layers))
    if args.package_output:
        package_output = directory / "package" if args.mixers_package else args.package_output
        package_blocks(
            graphs, args.deployment, package_output, Path("/usr/lib/ljmicro"), part=args.part
        )
        if args.mixers_package:
            combine_packages(args.mixers_package, package_output, args.package_output)


if __name__ == "__main__":
    main()
