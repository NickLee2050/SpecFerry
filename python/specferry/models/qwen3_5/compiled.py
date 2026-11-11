"""Source-bound DeltaNet imports; recurrent state precision is part of the contract."""

import json
import re
import shutil

from specferry.data.checkpoint import sha256
from specferry.validation.device import write_json

from .export import verify_export

SDK_FILES = ("libovxlib.so", "libOpenVX.so", "libGAL.so", "libArchModelSw.so", "libNNArchPerf.so")
DRIVER = "F53A3E6EFBB7D0FD7D4617C"


def native_manifest(graphs, *, mixer, sdk):
    """Reconstruct the public port/source contract, preserving graph and parameter bytes."""
    version = 2 if mixer and any(g.get("includes_input_norm") for g in graphs) else 1
    kind = "delta" if mixer else "feed-forward"
    profile = sdk.get("profile", sdk.get("module_srcversion"))
    lines = [f"specferry-qwen-{kind} {version} {profile} 5"]
    lines.extend(f"{name} {sdk['sha256'][name]}" for name in SDK_FILES)
    lines.append(str(len(graphs)))
    for graph in graphs:
        lines.append(
            f"{graph['layer']} {json.dumps(graph['path'])} {graph['nbg_sha256']} "
            f"{graph['constant_bytes']} " + " ".join(map(str, graph["graph_version"]))
        )
        if mixer and version == 2:
            lines.append(str(int(graph["includes_input_norm"])))
        for direction in ("input", "output"):
            lines.append(" ".join(map(str, graph[direction + "_order"])))
            lines.extend(
                f"{port['dtype']} {len(port['shape'])} " + " ".join(map(str, port["shape"]))
                for port in graph[direction + "s"]
            )
        lines.extend(
            f"{json.dumps(name)} {digest}"
            for name, digest in sorted(graph["weight_sha256"].items())
        )
    return "\n".join(lines) + "\n"


def graph_version(directory):
    header = (directory / "export_model/vnn_network.h").read_text()
    return [
        int(re.search(rf"#define\s+VNN_VERSION_{part}\s+\(?([0-9]+)", header)[1])
        for part in ("MAJOR", "MINOR", "PATCH")
    ]


def audit_mixer(directory):
    text = (directory / "export_model/vnn_network.c").read_text()
    tensors = {
        name: dtype
        for name, dtype in re.findall(
            r"/\*\s*@([^*]+):(?:out0|data)\s*\*/.*?NEW_(?:VIRTUAL|NORM|CONST)_TENSOR\([^;]*VSI_NN_TYPE_(\w+)",
            text,
            re.S,
        )
    }
    wide = {
        name: dtype
        for name, dtype in tensors.items()
        if "_fp32" in name
        or "present_recurrent" in name
        or re.search(r"input_norm_(?:squares|variance|regularized|root|normalized|scaled)", name)
    }
    if not wide or any(dtype != "FLOAT32" for dtype in wide.values()):
        raise ValueError("compiled Qwen mixer narrowed required FP32 arithmetic")
    return graph_version(directory)


def validate_feed_forward_ports(artifact, *, batch_tokens=None):
    if batch_tokens is not None and (
        type(batch_tokens) is not int or batch_tokens not in (1, 4, 8, 16)
    ):
        raise ValueError("unsupported compiled Qwen feed-forward batch selection")
    width = None
    for direction, names in (("input", ["residual", "mixer"]), ("output", ["output"])):
        order, ports = artifact[direction + "_order"], artifact[direction + "s"]
        if len(ports) != len(names) or sorted(order) != list(range(len(names))):
            raise ValueError("compiled Qwen feed-forward port permutation differs")
        for index, name in zip(order, names):
            shape = ports[index]["shape"]
            if len(shape) != 2 or type(shape[1]) is not int or shape[1] not in (1, 4, 8, 16):
                raise ValueError("compiled Qwen feed-forward batch width differs")
            if width is None:
                width = shape[1]
            if ports[index] != dict(name=name, dtype="F16", shape=[1024, width]):
                raise ValueError("compiled Qwen feed-forward semantic/precision contract differs")
    declared = artifact.get("batch_tokens", width)
    if type(declared) is not int or declared != width:
        raise ValueError("compiled Qwen feed-forward batch metadata differs")
    if batch_tokens is not None and width != batch_tokens:
        raise ValueError("compiled Qwen feed-forward batch differs from selected configuration")
    return width


def package_blocks(directories, deployment, output, sdk, *, part="mixer"):
    if part not in ("mixer", "feed-forward"):
        raise ValueError("unknown compiled Qwen block")
    mixer = part == "mixer"
    verify_export(deployment)
    records = {
        row["name"]: row
        for row in json.loads((deployment / "deployment-manifest.json").read_text())["tensors"]
    }
    graphs, total = [], 0
    for directory in directories:
        source = json.loads((directory / "source.json").read_text())
        artifact = json.loads((directory / "artifact.json").read_text())
        layer = source["layer"]
        includes_input_norm = source.get("includes_input_norm", False)
        if type(includes_input_norm) is not bool or (includes_input_norm and not mixer):
            raise ValueError("input normalization belongs only to a DeltaNet mixer")
        expected = {
            name: row["sha256"]
            for name, row in records.items()
            if (
                (
                    name.startswith(f"model.layers.{layer}.linear_attn.")
                    or (
                        includes_input_norm
                        and name == f"model.layers.{layer}.input_layernorm.weight"
                    )
                )
                if mixer
                else name.startswith(f"model.layers.{layer}.mlp.")
                or name == f"model.layers.{layer}.post_attention_layernorm.weight"
            )
        }
        binary = directory / "export_model_nbg_unify/network_binary.nb"
        if (
            len(expected) != (9 + int(includes_input_norm) if mixer else 4)
            or source.get("part", "mixer") != part
            or source["weight_sha256"] != expected
            or source["deployment_sha256"] != sha256(deployment / "deployment-manifest.json")
            or source["onnx_sha256"] != sha256(directory / "model.onnx")
            or artifact["nbg_sha256"] != sha256(binary)
        ):
            raise ValueError("compiled Qwen mixer source/NBG identity differs")
        batch_tokens = 1 if mixer else validate_feed_forward_ports(artifact)
        if not mixer and source.get("batch_tokens", 1) != batch_tokens:
            raise ValueError("compiled Qwen feed-forward source batch differs")
        shapes = [[1024, 1], [128, 128, 16], [4, 6144]]
        dtypes = ["F16", "F32", "F16"]
        for direction in ("input", "output"):
            order, ports = artifact[direction + "_order"], artifact[direction + "s"]
            count = 3 if mixer else (2 if direction == "input" else 1)
            expected_shapes = shapes if mixer else [[1024, batch_tokens]] * count
            expected_dtypes = dtypes if mixer else ["F16"] * count
            if (
                len(ports) != count
                or sorted(order) != list(range(count))
                or any(
                    ports[index]["shape"] != shape or ports[index]["dtype"] != dtype
                    for index, shape, dtype in zip(order, expected_shapes, expected_dtypes)
                )
            ):
                raise ValueError("compiled Qwen mixer state ports differ")
        minimum = sum(records[name]["bytes"] for name in expected)
        bound = artifact["constant_bytes"]
        if type(bound) is not int or not max(minimum, binary.stat().st_size) <= bound <= 2**30:
            raise ValueError("compiled Qwen mixer constant budget differs")
        total += bound
        graphs.append(
            dict(
                artifact,
                path=f"{'delta' if mixer else 'ffn'}{layer:02d}.nb",
                layer=layer,
                graph_version=audit_mixer(directory),
                weight_sha256=expected,
                includes_input_norm=includes_input_norm,
                outer_product=source.get("outer_product", "matmul") if mixer else None,
                joint_state_reads=source.get("joint_state_reads", False) if mixer else False,
                batch_tokens=batch_tokens,
            )
        )
    if not graphs or len({entry["layer"] for entry in graphs}) != len(graphs) or total > 2**30:
        raise ValueError("compiled Qwen mixer layer/budget aggregate differs")
    if not mixer and len({entry["batch_tokens"] for entry in graphs}) != 1:
        raise ValueError("compiled Qwen feed-forward package mixes token batch widths")
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(
        directory=str(sdk),
        module_srcversion=DRIVER,
        sha256={name: sha256(sdk / name) for name in SDK_FILES},
    )
    for entry, directory in zip(graphs, directories):
        shutil.copyfile(
            directory / "export_model_nbg_unify/network_binary.nb", output / entry["path"]
        )
    kind = "delta" if mixer else "feed-forward"
    native = "compiled-mixers.txt" if mixer else "compiled-feed-forward.txt"
    (output / native).write_text(native_manifest(graphs, mixer=mixer, sdk=identity))
    manifest = dict(
        schema_version=1,
        kind="qwen-" + kind,
        sdk=identity,
        graphs=graphs,
        deployment_sha256=sha256(deployment / "deployment-manifest.json"),
        native_manifest_sha256=sha256(output / native),
    )
    if not mixer:
        manifest["batch_tokens"] = graphs[0]["batch_tokens"]
    write_json(output / "manifest.json", manifest)
    return manifest


def combine_packages(mixers, feed_forward, output):
    """Bind the validated mixer and FFN sources into one complete-model package."""
    left = json.loads((mixers / "manifest.json").read_text())
    right = json.loads((feed_forward / "manifest.json").read_text())
    if (
        left["kind"] != "qwen-delta"
        or right["kind"] != "qwen-feed-forward"
        or len(left["graphs"]) != 18
        or len(right["graphs"]) != 24
        or left["sdk"] != right["sdk"]
        or left["deployment_sha256"] != right["deployment_sha256"]
        or sum(entry["constant_bytes"] for entry in left["graphs"] + right["graphs"]) > 2**30
    ):
        raise ValueError("complete Qwen source/SDK/count/constant budget differs")
    widths = {validate_feed_forward_ports(artifact) for artifact in right["graphs"]}
    if len(widths) != 1 or right.get("batch_tokens", next(iter(widths))) not in widths:
        raise ValueError("complete Qwen feed-forward package batch differs")
    output.mkdir(parents=True, exist_ok=False)
    for source, manifest, native in (
        (mixers, left, "compiled-mixers.txt"),
        (feed_forward, right, "compiled-feed-forward.txt"),
    ):
        if sha256(source / native) != manifest["native_manifest_sha256"]:
            raise ValueError("compiled Qwen native manifest changed")
        shutil.copyfile(source / native, output / native)
        for entry in manifest["graphs"]:
            binary = (source / entry["path"]).resolve(strict=True)
            if binary.parent != source.resolve() or sha256(binary) != entry["nbg_sha256"]:
                raise ValueError("compiled Qwen block identity changed")
            shutil.copyfile(binary, output / entry["path"])
    manifest = dict(
        left,
        feed_forward=right["graphs"],
        feed_forward_manifest_sha256=right["native_manifest_sha256"],
        feed_forward_batch_tokens=next(iter(widths)),
    )
    write_json(output / "manifest.json", manifest)
    return manifest
