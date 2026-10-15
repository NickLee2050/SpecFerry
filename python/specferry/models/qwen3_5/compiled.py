"""Source-bound DeltaNet imports; recurrent state precision is part of the contract."""

import json
import re
import shutil

from specferry.data.checkpoint import sha256
from specferry.validation.device import write_json

from .export import verify_export

SDK_FILES = ("libovxlib.so", "libOpenVX.so", "libGAL.so", "libArchModelSw.so", "libNNArchPerf.so")
DRIVER = "F53A3E6EFBB7D0FD7D4617C"


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


def validate_feed_forward_ports(artifact):
    for direction, names in (("input", ["residual", "mixer"]), ("output", ["output"])):
        order, ports = artifact[direction + "_order"], artifact[direction + "s"]
        if len(ports) != len(names) or sorted(order) != list(range(len(names))):
            raise ValueError("compiled Qwen feed-forward port permutation differs")
        for index, name in zip(order, names):
            if ports[index] != dict(name=name, dtype="F16", shape=[1024, 1]):
                raise ValueError("compiled Qwen feed-forward semantic/precision contract differs")


def validate_vocabulary_head_ports(artifact):
    rows = artifact["vocabulary_rows"]
    if type(rows) is not int or not 4096 <= rows <= 65536 or rows % 4096:
        raise ValueError("compiled vocabulary head row bound differs")
    for direction in ("input", "output"):
        count = 1 if direction == "input" else rows // 4096
        ports, order = artifact[direction + "s"], artifact[direction + "_order"]
        names = ["hidden"] if direction == "input" else [f"logits_{i}" for i in range(count)]
        shape = [1024, 1] if direction == "input" else [4096, 1]
        if (
            len(ports) != count
            or sorted(order) != list(range(count))
            or any(
                ports[index] != dict(name=name, dtype="F16", shape=shape)
                for index, name in zip(order, names)
            )
        ):
            raise ValueError("compiled vocabulary head semantic/precision port order differs")


def load_vocabulary_head(package, deployment, sdk):
    """Reject a changed prefix package before model preparation or device access."""
    package = package.resolve(strict=True)
    manifest = json.loads((package / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "qwen-vocabulary-head"
        or manifest["sdk"] != sdk
        or len(manifest["graphs"]) != 1
        or manifest["native_manifest_sha256"] != sha256(package / "compiled-head.txt")
        or manifest["deployment_sha256"] != sha256(deployment / "deployment-manifest.json")
    ):
        raise ValueError("compiled vocabulary head package/source/SDK differs")
    artifact = manifest["graphs"][0]
    validate_vocabulary_head_ports(artifact)
    records = json.loads((deployment / "deployment-manifest.json").read_text())["tensors"]
    table = next(row for row in records if row["name"] == "model.embed_tokens.weight")
    binary = (package / artifact["path"]).resolve(strict=True)
    if (
        binary.parent != package
        or sha256(binary) != artifact["nbg_sha256"]
        or artifact["weight_sha256"] != {table["name"]: table["sha256"]}
        or artifact["graph_version"] not in ([1, 1, 34], [1, 1, 35])
        or not max(artifact["vocabulary_rows"] * 1024 * 2, binary.stat().st_size)
        <= artifact["constant_bytes"]
        <= 2**30
    ):
        raise ValueError("compiled vocabulary head artifact/budget differs")
    return manifest


def package_blocks(directories, deployment, output, sdk, *, part="mixer"):
    if part not in ("mixer", "feed-forward", "vocabulary-head"):
        raise ValueError("unknown compiled Qwen block")
    mixer = part == "mixer"
    head = part == "vocabulary-head"
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
        rows = source.get("vocabulary_rows", 0)
        includes_input_norm = source.get("includes_input_norm", False)
        if type(includes_input_norm) is not bool or (includes_input_norm and not mixer):
            raise ValueError("input normalization belongs only to a DeltaNet mixer")
        if head and (
            len(directories) != 1 or layer != 0 or not 4096 <= rows <= 65536 or rows % 4096
        ):
            raise ValueError("compiled vocabulary head prefix geometry differs")
        if head:
            expected = {"model.embed_tokens.weight": records["model.embed_tokens.weight"]["sha256"]}
        else:
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
            len(expected) != (1 if head else 9 + int(includes_input_norm) if mixer else 4)
            or source.get("part", "mixer") != part
            or source["weight_sha256"] != expected
            or source["deployment_sha256"] != sha256(deployment / "deployment-manifest.json")
            or source["onnx_sha256"] != sha256(directory / "model.onnx")
            or artifact["nbg_sha256"] != sha256(binary)
        ):
            raise ValueError("compiled Qwen mixer source/NBG identity differs")
        shapes = [[1024, 1], [128, 128, 16], [4, 6144]]
        dtypes = ["F16", "F32", "F16"]
        for direction in ("input", "output"):
            order, ports = artifact[direction + "_order"], artifact[direction + "s"]
            if head:
                count = 1 if direction == "input" else rows // 4096
                expected_shapes = [[1024, 1]] if direction == "input" else [[4096, 1]] * count
            else:
                count = 3 if mixer else (2 if direction == "input" else 1)
                expected_shapes = shapes if mixer else [[1024, 1]] * count
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
            if head:
                names = (
                    ["hidden"] if direction == "input" else [f"logits_{i}" for i in range(count)]
                )
                if [ports[index]["name"] for index in order] != names:
                    raise ValueError("compiled vocabulary head semantic port order differs")
        if not mixer and not head:
            validate_feed_forward_ports(artifact)
        minimum = rows * 1024 * 2 if head else sum(records[name]["bytes"] for name in expected)
        bound = artifact["constant_bytes"]
        if type(bound) is not int or not max(minimum, binary.stat().st_size) <= bound <= 2**30:
            raise ValueError("compiled Qwen mixer constant budget differs")
        total += bound
        graphs.append(
            dict(
                artifact,
                path="head.nb" if head else f"{'delta' if mixer else 'ffn'}{layer:02d}.nb",
                layer=layer,
                graph_version=graph_version(directory) if head else audit_mixer(directory),
                weight_sha256=expected,
                vocabulary_rows=rows,
                includes_input_norm=includes_input_norm,
                outer_product=source.get("outer_product", "matmul") if mixer else None,
            )
        )
    if not graphs or len({entry["layer"] for entry in graphs}) != len(graphs) or total > 2**30:
        raise ValueError("compiled Qwen mixer layer/budget aggregate differs")
    output.mkdir(parents=True, exist_ok=False)
    identity = dict(
        directory=str(sdk),
        module_srcversion=DRIVER,
        sha256={name: sha256(sdk / name) for name in SDK_FILES},
    )
    kind = "vocabulary-head" if head else "delta" if mixer else "feed-forward"
    version = 2 if any(entry["includes_input_norm"] for entry in graphs) else 1
    lines = [f"specferry-qwen-{kind} {version} {DRIVER} 5"]
    lines.extend(f"{name} {identity['sha256'][name]}" for name in SDK_FILES)
    lines.append(str(len(graphs)))
    for entry, directory in zip(graphs, directories):
        shutil.copyfile(
            directory / "export_model_nbg_unify/network_binary.nb", output / entry["path"]
        )
        lines.append(
            f"{entry['layer']} {json.dumps(entry['path'])} {entry['nbg_sha256']} {entry['constant_bytes']} "
            + " ".join(map(str, entry["graph_version"]))
        )
        if head:
            lines.append(str(entry["vocabulary_rows"]))
        elif mixer and version == 2:
            lines.append(str(int(entry["includes_input_norm"])))
        for direction in ("input", "output"):
            lines.append(" ".join(map(str, entry[direction + "_order"])))
            lines.extend(
                f"{port['dtype']} {len(port['shape'])} " + " ".join(map(str, port["shape"]))
                for port in entry[direction + "s"]
            )
        lines.extend(
            f"{json.dumps(name)} {digest}"
            for name, digest in sorted(entry["weight_sha256"].items())
        )
    native = (
        "compiled-head.txt"
        if head
        else "compiled-mixers.txt"
        if mixer
        else "compiled-feed-forward.txt"
    )
    (output / native).write_text("\n".join(lines) + "\n")
    manifest = dict(
        schema_version=1,
        kind="qwen-" + kind,
        sdk=identity,
        graphs=graphs,
        deployment_sha256=sha256(deployment / "deployment-manifest.json"),
        native_manifest_sha256=sha256(output / native),
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def package_mixers(directories, deployment, output, sdk):
    return package_blocks(directories, deployment, output, sdk)


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
    for artifact in right["graphs"]:
        validate_feed_forward_ports(artifact)
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
    )
    write_json(output / "manifest.json", manifest)
    return manifest
