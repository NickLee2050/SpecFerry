"""Portable, source-bound packages for the accepted OPT FFN finish contract."""

import json
import re
import shutil
from pathlib import Path

from specferry.data.checkpoint import sha256
from specferry.validation.device import write_json

PARTS = (
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "final_layer_norm.weight",
    "final_layer_norm.bias",
)
SDK_FILES = ("libovxlib.so", "libOpenVX.so", "libGAL.so", "libArchModelSw.so", "libNNArchPerf.so")
DRIVER = "F53A3E6EFBB7D0FD7D4617C"


def audit_ffn_export(directory):
    """Metadata dtype can be wrong; inspect primitive and sealed C declarations."""
    text = (directory / "export_model/vnn_network.c").read_text()
    tensors = {}
    for comment, block in re.findall(r"/\*\s*(.*?)\*/(.*?)(?=/\*|\Z)", text, flags=re.S):
        name = re.fullmatch(r"\s*@?([^\s:]+):(out0|data)\s*", comment)
        dtype = re.search(r"NEW_(?:VIRTUAL|NORM|CONST)_TENSOR\([^;]*VSI_NN_TYPE_(\w+)", block)
        if name and dtype:
            tensors[name[1]] = dtype[1]
    sensitive = (
        "_fp32",
        "_mean",
        "_centered",
        "_variance",
        "_sqrt",
        "_inverse",
        "_normalized",
        "_affine",
    )
    for marker in sensitive:
        selected = [dtype for name, dtype in tensors.items() if marker in name]
        if not selected or any(dtype != "FLOAT32" for dtype in selected):
            raise ValueError(f"generated FFN normalization precision differs: {marker}")
    for marker in ("fc1_0_product", "fc2_0_product", "residual0", "norm0_output"):
        selected = [dtype for name, dtype in tensors.items() if marker in name]
        if not selected or any(dtype != "FLOAT16" for dtype in selected):
            raise ValueError(f"generated FFN FP16 boundary differs: {marker}")
    sealed = directory / "export_model_nbg_unify"
    wrapper = (sealed / "vnn_network.c").read_text()
    if re.findall(r"NEW_NORM_TENSOR\([^;]*VSI_NN_TYPE_(\w+)", wrapper) != ["FLOAT16", "FLOAT16"]:
        raise ValueError("sealed FFN ports must remain FP16")
    metadata = json.loads((sealed / "nbg_meta.json").read_text())
    if any(len(metadata[part]) != 1 for part in ("Inputs", "Outputs")) or any(
        port["shape"] != [1, 1024]
        for part in ("Inputs", "Outputs")
        for port in metadata[part].values()
    ):
        raise ValueError("sealed FFN ports must have ONNX shape [1,1024]")
    header = (directory / "export_model/vnn_network.h").read_text()
    version = [
        int(re.search(rf"#define\s+VNN_VERSION_{p}\s+\(?([0-9]+)", header)[1])
        for p in ("MAJOR", "MINOR", "PATCH")
    ]
    if version != [1, 1, 34]:
        raise ValueError("unsupported compiled FFN graph version")
    return {
        "graph_version": version,
        "generated_C_sha256": sha256(directory / "export_model/vnn_network.c"),
        "tensors": tensors,
    }


def package_prefill_exports(artifacts, deployment, output, manifest):
    """Import the established M16 FFNs; four ordinary layers leave const headroom."""
    models = json.loads((artifacts / "compiled-layers.json").read_text())["models"]
    if [item["layer"] for item in models] != list(range(24)):
        raise ValueError("prefill exports require all ordered source layers")
    records = {
        item["name"]: item
        for item in json.loads((deployment / "deployment-manifest.json").read_text())["tensors"]
    }
    layers, lines = (
        [],
        [
            "specferry-opt-prefill 1 "
            + sha256(output / "compiled-ffn.txt")
            + " "
            + " ".join(
                manifest["deployment_sha256"][name]
                for name in ("weights.index", "deployment-manifest.json")
            )
            + " 16 20"
        ],
    )
    for item in models[4:]:
        layer = item["layer"]
        directory = artifacts / f"layer{layer}"
        binary = Path(item["nbg"]).resolve(strict=True)
        parts = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
        identities = {
            f"decoder.layers.{layer}.{part}": records[f"decoder.layers.{layer}.{part}"]["sha256"]
            for part in parts
        }
        if (
            item["rows"] != 16
            or item["graph_version"] != [1, 1, 34]
            or item["source_parameter_order"] != list(parts)
            or item["source_parameter_sha256"] != list(identities.values())
            or sha256(directory / "ffn.onnx") != item["onnx_sha256"]
            or binary.parent != (directory / "export_ffn_nbg_unify").resolve()
            or sha256(binary) != item["nbg_sha256"]
        ):
            raise ValueError("prefill export geometry/source/NBG identity differs")
        wrapper = (binary.parent / "vnn_network.c").read_text()
        if re.findall(r"NEW_NORM_TENSOR\([^;]*VSI_NN_TYPE_(\w+)", wrapper) != [
            "FLOAT16",
            "FLOAT16",
        ]:
            raise ValueError("prefill NBG ports must remain FP16")
        for direction in ("Inputs", "Outputs"):
            ports = list(item["nbg_metadata"][direction].values())
            if len(ports) != 1 or ports[0]["shape"] != [16, 1024] or ports[0]["dtype"] != "float16":
                raise ValueError("prefill NBG requires fixed [16,1024] ports")
        target = output / f"prefill{layer:02d}.nb"
        shutil.copyfile(binary, target)
        entry = dict(
            layer=layer,
            path=target.name,
            nbg_sha256=sha256(target),
            constant_bytes=item["constant_tensor_bytes"],
            weight_sha256=identities,
        )
        layers.append(entry)
        lines.append(
            f"{layer} {json.dumps(target.name)} {entry['nbg_sha256']} {entry['constant_bytes']} 1 1 34 "
            + " ".join(identities.values())
        )
    native = output / "compiled-prefill.txt"
    native.write_text("\n".join(lines) + "\n")
    return dict(
        tokens=16,
        ordinary_layers=[0, 1, 2, 3],
        layers=layers,
        supported_capacities=[64],
        native_manifest_sha256=sha256(native),
    )


def validate_prefill_package(package, manifest, records):
    prefill = manifest["prefill"]
    if (
        not manifest.get("head")
        or prefill["tokens"] != 16
        or prefill["ordinary_layers"] != [0, 1, 2, 3]
        or prefill["supported_capacities"] not in ([64], [64, 256])
        or [item["layer"] for item in prefill["layers"]] != list(range(4, 24))
        or sha256(package / "compiled-prefill.txt") != prefill["native_manifest_sha256"]
    ):
        raise ValueError("prefill package requires one head and four ordinary M16 layers")
    header = (package / "compiled-prefill.txt").read_text().splitlines()[0].split()
    expected_header = [
        "specferry-opt-prefill",
        "1",
        manifest["native_manifest_sha256"],
        manifest["deployment_sha256"]["weights.index"],
        manifest["deployment_sha256"]["deployment-manifest.json"],
        "16",
        "20",
    ]
    if header != expected_header:
        raise ValueError("prefill native header must bind the M1 package and deployment")
    total = 0
    for item in prefill["layers"]:
        path = (package / item["path"]).resolve(strict=True)
        prefix = f"decoder.layers.{item['layer']}."
        expected = {
            prefix + part: records[prefix + part]["sha256"]
            for part in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
        }
        bound = item["constant_bytes"]
        if (
            path.parent != package
            or not path.stat().st_size
            or sha256(path) != item["nbg_sha256"]
            or item["weight_sha256"] != expected
            or type(bound) is not int
            or not max(path.stat().st_size, sum(records[name]["bytes"] for name in expected))
            <= bound
            <= 2**30
        ):
            raise ValueError("prefill NBG identity/source/constant budget differs")
        total += bound
    # The caller counts optional NBG projections (at least their source weight payload).
    # Without them, reserve all four M1 FCL matrices. The shared head is counted once.
    return total + (0 if manifest.get("projections") else 24 * 4 * 1024 * 1024 * 2)


def package_ffn_exports(
    artifacts,
    deployment,
    output,
    sdk,
    *,
    head_artifacts=None,
    projection_artifacts=None,
    prefill_artifacts=None,
):
    """Copy only runtime NBGs; preserve compiler evidence separately at its source."""
    from .export import verify_export

    verify_export(deployment)
    source = json.loads((deployment / "deployment-manifest.json").read_text())
    records = {record["name"]: record for record in source["tensors"]}
    output.mkdir(parents=True, exist_ok=False)
    layers = []
    for layer in range(24):
        directory = artifacts / f"layer{layer}"
        artifact = json.loads((directory / "artifact.json").read_text())
        audit = audit_ffn_export(directory)
        identities = {
            "decoder.layers." + str(layer) + "." + part: records[
                "decoder.layers." + str(layer) + "." + part
            ]["sha256"]
            for part in PARTS
        }
        if artifact["source"]["weight_sha256"] != identities:
            raise ValueError(f"compiled FFN source weights differ at layer {layer}")
        if sha256(directory / "model.onnx") != artifact["source"]["onnx_sha256"]:
            raise ValueError("compiled FFN source ONNX identity differs")
        binary = directory / "export_model_nbg_unify/network_binary.nb"
        if sha256(binary) != artifact["nbg_sha256"]:
            raise ValueError("compiled FFN NBG identity differs")
        minimum = sum(
            records[name]["bytes"] * (2 if "layer_norm" in name else 1) for name in identities
        )
        bound = artifact["constant_bytes"]
        if type(bound) is not int or not minimum <= bound <= 2**30 or bound < binary.stat().st_size:
            raise ValueError("compiled FFN constant bound does not cover the package")
        target = output / f"layer{layer:02d}.nb"
        shutil.copyfile(binary, target)
        layers.append(
            dict(
                layer=layer,
                path=target.name,
                nbg_sha256=sha256(target),
                constant_bytes=bound,
                weight_sha256=identities,
                precision=audit,
            )
        )
    manifest = dict(
        schema_version=1,
        kind="opt-ffn-finish",
        repo_id=source["repo_id"],
        revision=source["revision"],
        shape={"capacity": 64, "tokens": 1, "hidden": 1024, "intermediate": 4096, "layers": 24},
        graph_version=[1, 1, 34],
        sdk={
            "directory": str(sdk),
            "module_srcversion": DRIVER,
            "sha256": {name: sha256(sdk / name) for name in SDK_FILES},
        },
        deployment_sha256={
            name: sha256(deployment / name)
            for name in ("weights.index", "deployment-manifest.json")
        },
        layers=layers,
        supported_capacities=[64, 128],
    )
    if head_artifacts is not None:
        artifact = json.loads((head_artifacts / "artifact.json").read_text())
        head_source = json.loads((head_artifacts / "source.json").read_text())
        expected = {
            name: records[name]["sha256"]
            for name in ("decoder.project_out.weight", "decoder.embed_tokens.weight")
        }
        binary = head_artifacts / "export_model_nbg_unify/network_binary.nb"
        metadata = artifact["metadata"]
        if (
            head_source["weight_sha256"] != expected
            or artifact["graph_version"] != [1, 1, 34]
            or sha256(binary) != artifact["nbg_sha256"]
            or sha256(head_artifacts / "model.onnx") != head_source["onnx_sha256"]
        ):
            raise ValueError("compiled head source/NBG/version identity differs")
        for group, shape in (("Inputs", [1, 1024]), ("Outputs", [1, 50272])):
            ports = list(metadata[group].values())
            if len(ports) != 1 or ports[0]["dtype"] != "float16" or ports[0]["shape"] != shape:
                raise ValueError("compiled head must have FP16 fixed OPT ports")
        wrapper = (head_artifacts / "export_model_nbg_unify/vnn_network.c").read_text()
        if re.findall(r"NEW_NORM_TENSOR\([^;]*VSI_NN_TYPE_(\w+)", wrapper) != [
            "FLOAT16",
            "FLOAT16",
        ]:
            raise ValueError("compiled head wrapper ports differ")
        shutil.copyfile(binary, output / "head.nb")
        manifest["head"] = dict(
            path="head.nb",
            nbg_sha256=sha256(output / "head.nb"),
            constant_bytes=artifact["constant_bytes"],
            weight_sha256=expected,
        )
    if projection_artifacts is not None:
        manifest["projections"] = []
        for layer in range(24):
            entry = dict(layer=layer)
            for kind, parts in (("qkv", ("q_proj", "k_proj", "v_proj")), ("out", ("out_proj",))):
                directory = projection_artifacts / f"layer{layer}" / kind
                artifact = json.loads((directory / "artifact.json").read_text())
                source = json.loads((directory / "source.json").read_text())
                binary = directory / "export_model_nbg_unify/network_binary.nb"
                identities = {
                    f"decoder.layers.{layer}.self_attn.{part}.{suffix}": records[
                        f"decoder.layers.{layer}.self_attn.{part}.{suffix}"
                    ]["sha256"]
                    for part in parts
                    for suffix in ("weight", "bias")
                }
                if (
                    source["weight_sha256"] != identities
                    or artifact["graph_version"] != [1, 1, 34]
                    or sha256(binary) != artifact["nbg_sha256"]
                    or sha256(directory / "model.onnx") != source["onnx_sha256"]
                ):
                    raise ValueError("compiled projection source/NBG/version identity differs")
                wrapper = (directory / "export_model_nbg_unify/vnn_network.c").read_text()
                if re.findall(r"NEW_NORM_TENSOR\([^;]*VSI_NN_TYPE_(\w+)", wrapper) != [
                    "FLOAT16"
                ] * (len(parts) + 1):
                    raise ValueError("compiled projection wrapper ports differ")
                name = f"layer{layer:02d}-{kind}.nb"
                shutil.copyfile(binary, output / name)
                entry[kind] = dict(
                    path=name,
                    nbg_sha256=sha256(binary),
                    constant_bytes=artifact["constant_bytes"],
                    weight_sha256=identities,
                    output_order=artifact["output_order"],
                )
            manifest["projections"].append(entry)
    version = "3" if "projections" in manifest else "2" if "head" in manifest else "1"
    lines = [
        "specferry-opt-ffn-finish " + version,
        " ".join(manifest["deployment_sha256"].values()),
        DRIVER + " 5",
    ]
    lines.extend(name + " " + digest for name, digest in manifest["sdk"]["sha256"].items())
    lines.append("24")
    for layer in layers:
        lines.append(
            " ".join(
                [
                    str(layer["layer"]),
                    json.dumps(layer["path"]),
                    layer["nbg_sha256"],
                    str(layer["constant_bytes"]),
                    "1 1 34",
                    *layer["weight_sha256"].values(),
                ]
            )
        )
    if "projections" in manifest:
        lines.append(str(int("head" in manifest)))
    if "head" in manifest:
        head = manifest["head"]
        lines.append(
            " ".join(
                [
                    json.dumps(head["path"]),
                    head["nbg_sha256"],
                    str(head["constant_bytes"]),
                    "1 1 34",
                    *head["weight_sha256"].values(),
                ]
            )
        )
    if "projections" in manifest:
        lines.append("24")
        for layer in manifest["projections"]:
            values = [str(layer["layer"])]
            for kind in ("qkv", "out"):
                part = layer[kind]
                values.extend(
                    (
                        json.dumps(part["path"]),
                        part["nbg_sha256"],
                        str(part["constant_bytes"]),
                        "1 1 34",
                    )
                )
            values.extend(map(str, layer["qkv"]["output_order"]))
            values.extend(layer["qkv"]["weight_sha256"].values())
            values.extend(layer["out"]["weight_sha256"].values())
            lines.append(" ".join(values))
    (output / "compiled-ffn.txt").write_text("\n".join(lines) + "\n")
    manifest["native_manifest_sha256"] = sha256(output / "compiled-ffn.txt")
    if prefill_artifacts is not None:
        manifest["prefill"] = package_prefill_exports(
            prefill_artifacts, deployment, output, manifest
        )
    write_json(output / "manifest.json", manifest)
    return load_ffn_package(output, deployment)


def load_ffn_package(package, deployment, *, capacity=64):
    """Validate before request preparation or SDK initialization; do not infer shapes."""
    package = package.resolve(strict=True)
    manifest = json.loads((package / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "opt-ffn-finish"
        or manifest.get("shape")
        != {"capacity": 64, "tokens": 1, "hidden": 1024, "intermediate": 4096, "layers": 24}
        or manifest.get("supported_capacities", [64]) not in ([64], [64, 128], [64, 128, 256])
        or capacity not in (64, 128, 256)
        or capacity not in manifest.get("supported_capacities", [64])
        or manifest.get("graph_version") != [1, 1, 34]
    ):
        raise ValueError("compiled FFN package requires its declared C64/C128/C256 geometry")
    if sha256(package / "compiled-ffn.txt") != manifest["native_manifest_sha256"]:
        raise ValueError("compiled FFN native manifest identity differs")
    if set(manifest["deployment_sha256"]) != {"weights.index", "deployment-manifest.json"}:
        raise ValueError("compiled FFN package must pin both deployment manifests")
    for name, digest in manifest["deployment_sha256"].items():
        if (
            name not in {"weights.index", "deployment-manifest.json"}
            or sha256(deployment / name) != digest
        ):
            raise ValueError("compiled FFN deployment identity differs")
    source = json.loads((deployment / "deployment-manifest.json").read_text())
    if (manifest["repo_id"], manifest["revision"]) != (source["repo_id"], source["revision"]):
        raise ValueError("compiled FFN checkpoint revision differs")
    records = {record["name"]: record for record in source["tensors"]}
    if manifest["sdk"].get("module_srcversion") != DRIVER or set(manifest["sdk"]["sha256"]) != set(
        SDK_FILES
    ):
        raise ValueError("compiled FFN package requires the pinned v105 SDK contract")
    if len(manifest["layers"]) != 24:
        raise ValueError("compiled FFN package requires all 24 layers")
    total = 0
    for index, layer in enumerate(manifest["layers"]):
        path = (package / layer["path"]).resolve(strict=True)
        if (
            layer["layer"] != index
            or path.parent != package
            or not path.stat().st_size
            or sha256(path) != layer["nbg_sha256"]
        ):
            raise ValueError("compiled FFN NBG order/path/identity differs")
        prefix = f"decoder.layers.{index}."
        expected = {prefix + part: records[prefix + part]["sha256"] for part in PARTS}
        minimum = sum(
            records[name]["bytes"] * (2 if "layer_norm" in name else 1) for name in expected
        )
        if layer["weight_sha256"] != expected:
            raise ValueError("compiled FFN source weight identities differ")
        bound = layer["constant_bytes"]
        if type(bound) is not int or not max(path.stat().st_size, minimum) <= bound <= 2**30:
            raise ValueError("compiled FFN package has an invalid constant bound")
        total += bound
    if "head" in manifest:
        head = manifest["head"]
        path = (package / head["path"]).resolve(strict=True)
        expected = {
            name: records[name]["sha256"]
            for name in ("decoder.project_out.weight", "decoder.embed_tokens.weight")
        }
        bound = head["constant_bytes"]
        if (
            path.parent != package
            or not path.stat().st_size
            or sha256(path) != head["nbg_sha256"]
            or head["weight_sha256"] != expected
            or type(bound) is not int
            or not max(path.stat().st_size, sum(records[n]["bytes"] for n in expected))
            <= bound
            <= 2**30
        ):
            raise ValueError("compiled head identity or constant bound differs")
        total += bound
    if "projections" in manifest:
        if len(manifest["projections"]) != 24:
            raise ValueError("compiled projections require all 24 layers")
        for index, layer in enumerate(manifest["projections"]):
            if layer["layer"] != index:
                raise ValueError("compiled projection layer order differs")
            for kind, parts in (("qkv", ("q_proj", "k_proj", "v_proj")), ("out", ("out_proj",))):
                part = layer[kind]
                path = (package / part["path"]).resolve(strict=True)
                expected = {
                    f"decoder.layers.{index}.self_attn.{name}.{suffix}": records[
                        f"decoder.layers.{index}.self_attn.{name}.{suffix}"
                    ]["sha256"]
                    for name in parts
                    for suffix in ("weight", "bias")
                }
                bound = part["constant_bytes"]
                if (
                    path.parent != package
                    or not path.stat().st_size
                    or sha256(path) != part["nbg_sha256"]
                    or part["weight_sha256"] != expected
                    or type(bound) is not int
                    or not max(path.stat().st_size, sum(records[n]["bytes"] for n in expected))
                    <= bound
                    <= 2**30
                    or sorted(part["output_order"]) != list(range(len(parts)))
                ):
                    raise ValueError("compiled projection identity/ports/constant bound differs")
                total += bound
    if "prefill" in manifest:
        total += validate_prefill_package(package, manifest, records)
    if total > 2**30:
        raise ValueError("compiled FFN package exceeds the constant payload limit")
    return manifest
