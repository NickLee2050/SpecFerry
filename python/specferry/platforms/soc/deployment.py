"""Build relocatable SoC OPT bundles without changing source NBGs or EP packages."""

import json
import shutil
import tempfile
from pathlib import Path

from specferry.models.opt.compiled import SDK_FILES, validate_prefill_package
from specferry.models.opt.compiled_model import load_model_package
from specferry.validation.recovery_controls import FAULT

from . import board_runner
from .board_runner import read_json, save, sha256, verify_bundle

DRIVER_SHA256 = "4c9a6ca280791ec23b13759ebee67759dd8a40bdb465536ba0f1dff3f78449bc"


def request_options(request):
    options = {}
    for line in (request / "run-options.txt").read_text().splitlines():
        key, value = line.split()
        if key in options:
            raise ValueError("duplicate request option: " + key)
        options[key] = value
    required = {
        "variant": "compiled-fullgraph",
        "kv_transfer": "mapped-slot",
    }
    if any(options.get(key) != value for key, value in required.items()) or (
        options.get("block"),
        options.get("prefill_head"),
    ) not in (("1", "every-token"), ("16", "last-token")):
        raise ValueError("SoC bundle requires fullgraph C64 mapped-slot with explicit M1/M16")
    if options["block"] == "16" and options.get("benchmark", "0") != "0":
        raise ValueError("fullgraph block prefill requires independent reset requests")
    if options.get("history_mode", "reset") != "reset" or options.get("lifecycle", "0") != "0":
        raise ValueError("SoC bundle requires reset requests without Context rebuilding")
    metadata = read_json(request / "reference.json")
    if (metadata["capacity"], metadata["layers"], metadata["heads"], metadata["width"]) != (
        64,
        24,
        16,
        64,
    ):
        raise ValueError("SoC reference geometry differs from OPT C64")
    requests = [
        [int(x) for x in line.split()]
        for line in (request / "requests.txt").read_text().splitlines()
    ]
    expected = [
        [int(x) for x in line.split()]
        for line in (request / "expected.txt").read_text().splitlines()
    ]
    if not 1 <= len(requests) == len(expected) == len(metadata["requests"]) <= 64:
        raise ValueError("missing or mismatched independent request/reference rows")
    for prompt, tokens, reference in zip(requests, expected, metadata["requests"], strict=True):
        if (
            not prompt
            or not tokens
            or tokens != reference["tokens"]
            or reference["prompt_length"] != len(prompt)
            or reference["consumed"] != len(prompt) + len(tokens) - 1
            or reference["consumed"] > 64
        ):
            raise ValueError("request differs from its independent reference")
    return options


def package_block_prefill(source, output, model, deployment):
    """Reuse pinned M16 FFNs with six ordinary layers to preserve the const budget.

    NBG internals cannot share weights with the M1 graph. Ordinary prefill weights
    keep their existing WeightBank ownership/classification; no const reclassification.
    """
    source = Path(source).resolve()
    original = read_json(source / "manifest.json")
    records = {
        row["name"]: row for row in read_json(deployment / "deployment-manifest.json")["tensors"]
    }
    if original.get("kind") == "opt-compiled-block-prefill":
        return package_compiled_prefill(source, output, model, original, records)
    if (
        original["deployment_sha256"] != model["deployment_sha256"]
        or sha256(source / "compiled-ffn.txt") != original["native_manifest_sha256"]
    ):
        raise ValueError("prefill source package/deployment differs")
    validate_prefill_package(source, original, records)
    layers = original["prefill"]["layers"][2:]
    bound = model["graphs"][0]["constant_bytes"] + sum(row["constant_bytes"] for row in layers)
    # Leave room for small operator constants. The runtime budget counts actual
    # allocations too; this bound alone is not physical-pool acceptance.
    if bound > 2**30 - 8 * 1024**2:
        raise ValueError("fullgraph and prefill leave insufficient constant headroom")
    lines = [
        "specferry-opt-prefill 2 "
        + sha256(output / "compiled-model.txt")
        + " "
        + " ".join(
            model["deployment_sha256"][name]
            for name in ("weights.index", "deployment-manifest.json")
        )
        + " 16 18"
    ]
    for item in layers:
        shutil.copyfile(source / item["path"], output / item["path"])
        prefix = f"decoder.layers.{item['layer']}."
        identities = [
            item["weight_sha256"][prefix + part]
            for part in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
        ]
        lines.append(
            f"{item['layer']} {json.dumps(item['path'])} {item['nbg_sha256']} "
            f"{item['constant_bytes']} 1 1 34 " + " ".join(identities)
        )
    native = output / "compiled-prefill.txt"
    native.write_text("\n".join(lines) + "\n")
    return {
        "tokens": 16,
        "ordinary_layers": list(range(6)),
        "layers": layers,
        "native_manifest_sha256": sha256(native),
        "combined_nbg_constant_bytes": bound,
        "cache_handoff": "mapped-fp16-slot-major-to-fp32-head-major-prefix",
        "final_prompt_token": "m1",
    }


def package_compiled_prefill(source, output, model, original, records):
    """Bind public M16 NBG ports and source identities before any SDK calls."""
    layers = original["layers"]
    if (
        original["deployment_sha256"] != model["deployment_sha256"]
        or original["tokens"] != 16
        or original["capacity"] != 64
        or [item["layer"] for item in layers] != list(range(24))
    ):
        raise ValueError("compiled prefill source/geometry differs")
    bound = model["graphs"][0]["constant_bytes"] + sum(x["constant_bytes"] for x in layers)
    if bound > 2**30 - 8 * 1024**2:
        raise ValueError("compiled prefill leaves insufficient constant headroom")
    lines = [
        "specferry-opt-compiled-block-prefill 1 "
        + sha256(output / "compiled-model.txt")
        + " "
        + " ".join(
            model["deployment_sha256"][name]
            for name in ("weights.index", "deployment-manifest.json")
        )
        + " 64 16 24"
    ]
    inputs = [
        ("hidden", "F16", [1024, 16]),
        ("past_key", "F32", [64, 64, 16]),
        ("past_value", "F32", [64, 64, 16]),
        ("block_index", "I32", [1]),
        ("attention_mask", "F32", [64, 16]),
    ]
    for item in layers:
        full = item["full_decoder"]
        parts = ["self_attn." + part for part in ("q_proj", "k_proj", "v_proj", "out_proj")]
        parts += ["self_attn_layer_norm"] + (["fc1", "fc2", "final_layer_norm"] if full else [])
        names = [
            f"decoder.layers.{item['layer']}.{part}.{suffix}"
            for part in parts
            for suffix in ("weight", "bias")
        ]
        expected = {name: records[name]["sha256"] for name in names}
        path = (source / item["path"]).resolve(strict=True)
        outputs = [
            ("hidden_out" if full else "attention_norm_output", "F16", [1024, 16]),
            ("present_key", "F32", [64, 64, 16]),
            ("present_value", "F32", [64, 64, 16]),
        ]

        def ports(direction):
            return [(p["name"], p["dtype"], p["shape"]) for p in item[direction]]

        if (
            type(full) is not bool
            or path.parent != source
            or not path.stat().st_size
            or sha256(path) != item["nbg_sha256"]
            or item["weight_sha256"] != expected
            or item["graph_version"] != [1, 1, 35]
            or ports("inputs") != inputs
            or ports("outputs") != outputs
            or item["input_order"] != list(range(5))
            or item["output_order"] != list(range(3))
            or item["constant_bytes"]
            < max(path.stat().st_size, sum(records[n]["bytes"] for n in names))
        ):
            raise ValueError("compiled prefill NBG identity/ports/weights differ")
        shutil.copyfile(path, output / item["path"])
        lines.append(
            f"{item['layer']} {int(full)} {json.dumps(item['path'])} {item['nbg_sha256']} {item['constant_bytes']} 1 1 35 {len(names)}"
        )
        lines.extend(f"{name} {digest}" for name, digest in sorted(expected.items()))
    native = output / "compiled-block-prefill.txt"
    native.write_text("\n".join(lines) + "\n")
    return dict(
        tokens=16,
        backend="compiled-attention",
        layers=layers,
        native_manifest_sha256=sha256(native),
        combined_nbg_constant_bytes=bound,
        cache_handoff="mapped-fp32-head-major-prefix",
        final_prompt_token="m1",
    )


def prepare_bundle(binary, deployment, package, request, sdk, output, *, prefill_package=None):
    binary, deployment, package, request, sdk, output = map(
        lambda path: Path(path).resolve(), (binary, deployment, package, request, sdk, output)
    )
    if output.exists():
        raise ValueError("use a new bundle directory")
    options = request_options(request)
    if (options["block"] == "16") != (prefill_package is not None):
        raise ValueError("block16 requires an explicit source-bound prefill package")
    model = load_model_package(package, deployment, capacity=64)
    if model["kind"] != "fullgraph":
        raise ValueError("SoC bundle currently supports one full-model NBG")
    graph = model["graphs"][0]
    records = read_json(deployment / "deployment-manifest.json")["tensors"]
    if graph["weight_sha256"] != {row["name"]: row["sha256"] for row in records}:
        raise ValueError("NBG source weight identities differ")
    if sha256(sdk / "drivers/galcore.ko") != DRIVER_SHA256:
        raise ValueError("SDK snapshot is not the pinned SoC module")
    sdk_hashes = {path.name: sha256(path) for path in sorted((sdk / "drivers").glob("*.so"))}
    sdk_hashes["galcore.ko"] = DRIVER_SHA256
    if not all(name in sdk_hashes for name in SDK_FILES):
        raise ValueError("incomplete SDK snapshot")
    if not (sdk / "include/CL/cl_viv_vx_ext.h").is_file():
        raise ValueError("SDK CL headers are missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".soc-bundle-", dir=output.parent) as temporary:
        root = Path(temporary)
        for name in ("bin", "weights", "model", "request", "runtime"):
            (root / name).mkdir()
        shutil.copy2(binary, root / "bin/specferry_opt_graph_generate")
        (root / "bin/specferry_opt_graph_generate").chmod(0o755)
        shutil.copyfile(board_runner.__file__, root / "board_runner.py")
        for name in ("weights.bin", "weights.index", "deployment-manifest.json"):
            shutil.copyfile(deployment / name, root / "weights" / name)
        shutil.copyfile(package / graph["path"], root / "model" / graph["path"])
        # Only the platform identity changes. The original graph, weights and ports stay pinned.
        native = (package / "compiled-model.txt").read_text().splitlines()
        if (
            native[0] != "specferry-opt-model 1 fullgraph 64"
            or len(native[2].split()) != 2
            or native[2].split()[1] != "5"
            or tuple(line.split()[0] for line in native[3:8]) != SDK_FILES
        ):
            raise ValueError("unexpected native compiled-model manifest layout")
        native[2] = "np101-soc-riscv64 5"
        native[3:8] = [f"{name} {sdk_hashes[name]}" for name in SDK_FILES]
        (root / "model/compiled-model.txt").write_text("\n".join(native) + "\n")
        model["sdk"] = {"profile": "np101-soc-riscv64", "sha256": sdk_hashes}
        model["native_manifest_sha256"] = sha256(root / "model/compiled-model.txt")
        if prefill_package is not None:
            model["prefill"] = package_block_prefill(
                prefill_package, root / "model", model, deployment
            )
        save(root / "model/manifest.json", model)
        for path in request.iterdir():
            if path.name in {
                "model.txt",
                "components.txt",
                "reference.json",
                "requests.txt",
                "expected.txt",
                "run-options.txt",
            } or path.name.startswith("expected."):
                if path.is_file() and not path.is_symlink():
                    shutil.copyfile(path, root / "request" / path.name)
        (root / "request/compiled-model-package.txt").write_text('"../model"\n')
        if model.get("prefill", {}).get("backend") == "compiled-attention":
            reference = read_json(root / "request/reference.json")
            reference["prefill_backend"] = "compiled-attention"
            save(root / "request/reference.json", reference)
        shutil.copytree(sdk / "include/CL", root / "runtime/include/CL")
        # The validated vendor shader compiler also resolves these headers from cwd.
        for path in (sdk / "include/CL").glob("*.h"):
            shutil.copyfile(path, root / "runtime" / path.name)
        files = {
            path.relative_to(root).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "executable": path == root / "bin/specferry_opt_graph_generate",
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "target": "np101-soc-riscv64",
            "files": files,
            "payload_bytes": sum(item["bytes"] for item in files.values()),
            "sdk": sdk_hashes,
            "request_options": options,
            "source_native_manifest_sha256": sha256(package / "compiled-model.txt"),
            "driver_parameters": {
                "contiguousBase": 0x80000000,
                "contiguousSize": 0x70000000,
                "stuckDump": 1,
            },
            "fault_pattern": FAULT.pattern + r"|Out of memory|I/O error|kernel panic",
        }
        save(root / "bundle.json", manifest)
        verify_bundle(root, check_sdk=False)
        root.rename(output)
    return manifest
