"""Fixed-state OPT packages with source identities and explicit public/SDK port order."""

import json
import shutil

from specferry.data.checkpoint import sha256
from specferry.validation.device import write_json

from .compiled import DRIVER, SDK_FILES
from .export import verify_export


def expected_ports(kind, capacity):
    cache = [64, capacity, 16]
    controls = [("kv_indices", "I32", [2, 16]), ("attention_mask", "F32", [capacity, 1])]
    if kind == "decoders":
        return (
            [
                ("hidden", "F16", [1024, 1]),
                ("past_key", "F32", cache),
                ("past_value", "F32", cache),
                *controls,
            ],
            [
                ("hidden_out", "F16", [1024, 1]),
                ("present_key", "F32", cache),
                ("present_value", "F32", cache),
            ],
        )
    if kind != "fullgraph":
        raise ValueError("unknown compiled OPT model scope")
    inputs = [("token", "I32", [1, 1]), ("position", "I32", [1, 1]), *controls]
    outputs = [("logits", "F16", [50272, 1]), ("selected", "I32", [1])]
    for index in range(24):
        for part in ("key", "value"):
            inputs.append((f"layer{index}.past_{part}", "F32", cache))
            outputs.append((f"layer{index}.present_{part}", "F32", cache))
    return inputs, outputs


def _validate_ports(entry, kind, capacity):
    for direction, expected in zip(("input", "output"), expected_ports(kind, capacity)):
        ports, order = entry[direction + "s"], entry[direction + "_order"]
        if sorted(order) != list(range(len(expected))) or len(ports) != len(expected):
            raise ValueError("compiled OPT ports require a complete permutation")
        for role, index in zip(expected, order):
            actual = ports[index]
            if (actual["name"], actual["dtype"], actual["shape"]) != role:
                raise ValueError(f"compiled OPT port contract differs: {role[0]}")


def package_model_exports(artifacts, deployment, output):
    verify_export(deployment)
    export = json.loads((artifacts / "export.json").read_text())
    kind, capacity = export["kind"], export["capacity"]
    if kind not in ("decoders", "fullgraph") or capacity not in (64, 128):
        raise ValueError("compiled OPT model requires fixed C64/C128")
    records = {
        row["name"]: row
        for row in json.loads((deployment / "deployment-manifest.json").read_text())["tensors"]
    }
    entries = []
    directories = (
        [artifacts / f"layer{i}" for i in range(24)]
        if kind == "decoders"
        else [artifacts / "model"]
    )
    for index, directory in enumerate(directories):
        entry = json.loads((directory / "artifact.json").read_text())
        source = json.loads((directory / "source.json").read_text())
        binary = directory / "export_model_nbg_unify/network_binary.nb"
        if (
            source["kind"] != kind
            or source["capacity"] != capacity
            or source["layer"] != (index if kind == "decoders" else None)
            or source.get("kv_update", "scatter") not in ("scatter", "select")
        ):
            raise ValueError("compiled OPT model source geometry differs")
        identities = source["weight_sha256"]
        expected_names = (
            set(records)
            if kind == "fullgraph"
            else {name for name in records if name.startswith(f"decoder.layers.{index}.")}
        )
        if set(identities) != expected_names or any(
            records[name]["sha256"] != digest for name, digest in identities.items()
        ):
            raise ValueError("compiled OPT model source weights differ")
        minimum = sum(
            records[name]["bytes"] * (2 if "layer_norm" in name else 1) for name in identities
        )
        if (
            entry["graph_version"] not in ([1, 1, 34], [1, 1, 35])
            or sha256(binary) != entry["nbg_sha256"]
            or sha256(directory / "model.onnx") != source["onnx_sha256"]
        ):
            raise ValueError("compiled OPT model NBG/source/version differs")
        if (
            type(entry["constant_bytes"]) is not int
            or not max(minimum, binary.stat().st_size) <= entry["constant_bytes"] <= 2**30
        ):
            raise ValueError("compiled OPT model constant bound is insufficient")
        _validate_ports(entry, kind, capacity)
        entry.update(
            path=f"decoder{index:02d}.nb" if kind == "decoders" else "model.nb",
            weight_sha256=identities,
            kv_update=source.get("kv_update", "scatter"),
        )
        entries.append(entry)
    if sum(entry["constant_bytes"] for entry in entries) > 2**30:
        raise ValueError("compiled OPT model exceeds constant payload budget")
    output.mkdir(parents=True, exist_ok=False)
    for entry, directory in zip(entries, directories):
        shutil.copyfile(
            directory / "export_model_nbg_unify/network_binary.nb", output / entry["path"]
        )
    manifest = dict(
        schema_version=1,
        kind=kind,
        capacity=capacity,
        sdk=export["sdk"],
        deployment_sha256={
            name: sha256(deployment / name)
            for name in ("weights.index", "deployment-manifest.json")
        },
        graphs=entries,
    )
    lines = [
        f"specferry-opt-model 1 {kind} {capacity}",
        " ".join(manifest["deployment_sha256"].values()),
        f"{DRIVER} 5",
    ]
    lines.extend(f"{name} {manifest['sdk']['sha256'][name]}" for name in SDK_FILES)
    lines.append(str(len(entries)))
    for entry in entries:
        lines.append(
            f"{json.dumps(entry['path'])} {entry['nbg_sha256']} {entry['constant_bytes']} "
            + " ".join(map(str, entry["graph_version"]))
        )
        for direction in ("input", "output"):
            ports = entry[direction + "s"]
            lines.append(str(len(ports)) + " " + " ".join(map(str, entry[direction + "_order"])))
            lines.extend(
                f"{json.dumps(port['name'])} {port['dtype']} {len(port['shape'])} "
                + " ".join(map(str, port["shape"]))
                for port in ports
            )
        lines.append(str(len(entry["weight_sha256"])))
        lines.extend(
            f"{json.dumps(name)} {digest}"
            for name, digest in sorted(entry["weight_sha256"].items())
        )
    (output / "compiled-model.txt").write_text("\n".join(lines) + "\n")
    write_json(output / "manifest.json", manifest)
    manifest["native_manifest_sha256"] = sha256(output / "compiled-model.txt")
    write_json(output / "manifest.json", manifest)
    return manifest


def load_model_package(package, deployment, *, capacity):
    verify_export(deployment)
    manifest = json.loads((package / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["capacity"] != capacity
        or capacity not in (64, 128)
    ):
        raise ValueError("compiled OPT model package capacity/schema differs")
    if sha256(package / "compiled-model.txt") != manifest["native_manifest_sha256"]:
        raise ValueError("compiled OPT native manifest differs")
    if any(
        sha256(deployment / name) != digest
        for name, digest in manifest["deployment_sha256"].items()
    ):
        raise ValueError("compiled OPT model deployment differs")
    if len(manifest["graphs"]) != (24 if manifest["kind"] == "decoders" else 1):
        raise ValueError("compiled OPT model graph count differs")
    total = 0
    for entry in manifest["graphs"]:
        binary = (package / entry["path"]).resolve(strict=True)
        if binary.parent != package.resolve() or sha256(binary) != entry["nbg_sha256"]:
            raise ValueError("compiled OPT model path/hash differs")
        _validate_ports(entry, manifest["kind"], capacity)
        bound = entry["constant_bytes"]
        if type(bound) is not int or not binary.stat().st_size <= bound <= 2**30:
            raise ValueError("compiled OPT model budget differs")
        total += bound
    if total > 2**30:
        raise ValueError("compiled OPT model exceeds constant budget")
    return manifest
