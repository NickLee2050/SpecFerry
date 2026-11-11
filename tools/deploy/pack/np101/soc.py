"""Build relocatable SoC OPT bundles without changing source NBGs or EP packages."""

import json
import shutil
import tempfile
from pathlib import Path

from tools.deploy.execute.np101.recovery_controls import FAULT
from tools.deploy.execute.np101.soc import board_runner
from tools.deploy.execute.np101.soc.board_runner import read_json, save, sha256, verify_bundle
from tools.deploy.pack.opt.compiled import SDK_FILES
from tools.deploy.pack.opt.compiled_model import load_model_package

DRIVER_SHA256 = "4c9a6ca280791ec23b13759ebee67759dd8a40bdb465536ba0f1dff3f78449bc"
SUPPORTED_DRIVER_SHA256 = {
    DRIVER_SHA256,
    "d19f6d06ba1b8017cb44fb3280f62f9c1a0a0ef04c299c00867040cc9ae8de77",
}


def sdk_identity(sdk):
    """Resolve the pinned board SDK without contacting or loading a device."""
    driver_digest = sha256(sdk / "drivers/galcore.ko")
    if driver_digest not in SUPPORTED_DRIVER_SHA256:
        raise ValueError("SDK snapshot is not the pinned SoC module")
    hashes = {path.name: sha256(path) for path in sorted((sdk / "drivers").glob("*.so"))}
    hashes["galcore.ko"] = driver_digest
    if not all(name in hashes for name in SDK_FILES):
        raise ValueError("incomplete SDK snapshot")
    if not (sdk / "include/CL/cl_viv_vx_ext.h").is_file():
        raise ValueError("SDK CL headers are missing")
    return hashes


def stage_runtime(root, binary, deployment, sdk, executable):
    """Stage shared transport, source weights and shader headers for either model."""
    for name in ("bin", "weights", "model", "request", "runtime"):
        (root / name).mkdir()
    shutil.copy2(binary, root / "bin" / executable)
    (root / "bin" / executable).chmod(0o755)
    shutil.copyfile(board_runner.__file__, root / "board_runner.py")
    for name in ("weights.bin", "weights.index", "deployment-manifest.json"):
        shutil.copyfile(deployment / name, root / "weights" / name)
    shutil.copytree(sdk / "include/CL", root / "runtime/include/CL")
    # The validated vendor shader compiler also resolves these headers from cwd.
    for path in (sdk / "include/CL").glob("*.h"):
        shutil.copyfile(path, root / "runtime" / path.name)


def finish_bundle(root, sdk_hashes, options, *, model_family="opt", **provenance):
    """Bind every staged file to the shared board supervisor and recovery policy."""
    executable = (
        "specferry_qwen_generate" if model_family == "qwen3_5" else "specferry_opt_graph_generate"
    )
    files = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "executable": path == root / "bin" / executable,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "target": "np101-soc-riscv64",
        "model_family": model_family,
        "files": files,
        "payload_bytes": sum(item["bytes"] for item in files.values()),
        "sdk": sdk_hashes,
        "request_options": options,
        # Match the validated 512MiB Linux layout; older bundles keep their pinned policy.
        "linux_memory_reserve_mib": 64,
        "driver_parameters": {
            "contiguousBase": 0x60000000,
            "contiguousSize": 0x90000000,
            "stuckDump": 1,
        },
        "fault_pattern": FAULT.pattern + r"|Out of memory|I/O error|kernel panic",
        **provenance,
    }
    save(root / "bundle.json", manifest)
    verify_bundle(root, check_sdk=False)
    return manifest


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
        raise ValueError("SoC bundle requires fullgraph mapped-slot with explicit M1/M16")
    if options["block"] == "16" and options.get("benchmark", "0") != "0":
        raise ValueError("fullgraph block prefill requires independent reset requests")
    if options.get("history_mode", "reset") != "reset" or options.get("lifecycle", "0") != "0":
        raise ValueError("SoC bundle requires reset requests without Context rebuilding")
    metadata = read_json(request / "reference.json")
    capacity = metadata["capacity"]
    if capacity not in (64, 128, 256, 512) or (
        metadata["layers"],
        metadata["heads"],
        metadata["width"],
    ) != (
        24,
        16,
        64,
    ):
        raise ValueError("SoC reference geometry differs from OPT C64/C128/C256/C512")
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
            or reference["consumed"] > capacity
        ):
            raise ValueError("request differs from its independent reference")
    return options


def package_block_prefill(source, output, model, deployment):
    """Package source-bound attention and decoder artifacts for shared M16 KV."""
    source = Path(source).resolve()
    original = read_json(source / "manifest.json")
    records = {
        row["name"]: row for row in read_json(deployment / "deployment-manifest.json")["tensors"]
    }
    if original.get("kind") != "opt-compiled-block-prefill":
        raise ValueError("SoC M16 requires a compiled attention prefill package")
    layers = original["layers"]
    capacity = model["capacity"]
    if (
        original["deployment_sha256"] != model["deployment_sha256"]
        or original["tokens"] != 16
        or capacity not in (64, 128, 256, 512)
        or original["capacity"] != capacity
        or [item["layer"] for item in layers] != list(range(24))
    ):
        raise ValueError("compiled prefill source/geometry differs")
    bound = model["graphs"][0]["constant_bytes"] + sum(x["constant_bytes"] for x in layers)
    if bound > 2**30 - 8 * 1024**2:
        raise ValueError("compiled prefill leaves insufficient constant headroom")
    cache_dtype = model["graphs"][0].get("cache_dtype", "F32")
    if cache_dtype not in ("F16", "F32") or original.get("cache_dtype", "F32") != cache_dtype:
        raise ValueError("prefill and decode KV precision differs")
    version = 2 if cache_dtype == "F16" else 1
    lines = [
        f"specferry-opt-compiled-block-prefill {version} "
        + sha256(output / "compiled-model.txt")
        + " "
        + " ".join(
            model["deployment_sha256"][name]
            for name in ("weights.index", "deployment-manifest.json")
        )
        + f" {capacity} 16 24"
        + (f" {cache_dtype}" if version == 2 else "")
    ]
    inputs = [
        ("hidden", "F16", [1024, 16]),
        ("past_key", cache_dtype, [64, capacity, 16]),
        ("past_value", cache_dtype, [64, capacity, 16]),
        ("block_index", "I32", [1]),
        ("attention_mask", "F32", [capacity, 16]),
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
            ("present_key", cache_dtype, [64, capacity, 16]),
            ("present_value", cache_dtype, [64, capacity, 16]),
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
        cache_handoff=f"shared-{cache_dtype.lower()}-input",
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
    capacity = read_json(request / "reference.json")["capacity"]
    model = load_model_package(package, deployment, capacity=capacity)
    if model["kind"] != "fullgraph":
        raise ValueError("SoC bundle currently supports one full-model NBG")
    graph = model["graphs"][0]
    records = read_json(deployment / "deployment-manifest.json")["tensors"]
    if graph["weight_sha256"] != {row["name"]: row["sha256"] for row in records}:
        raise ValueError("NBG source weight identities differ")
    sdk_hashes = sdk_identity(sdk)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".soc-bundle-", dir=output.parent) as temporary:
        root = Path(temporary)
        stage_runtime(root, binary, deployment, sdk, "specferry_opt_graph_generate")
        shutil.copyfile(package / graph["path"], root / "model" / graph["path"])
        # Only the platform identity changes. The original graph, weights and ports stay pinned.
        native = (package / "compiled-model.txt").read_text().splitlines()
        if (
            native[0] != f"specferry-opt-model 1 fullgraph {capacity}"
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
        reference = read_json(root / "request/reference.json")
        reference["cache_dtype"] = model["graphs"][0].get("cache_dtype", "F32")
        save(root / "request/reference.json", reference)
        if model.get("prefill", {}).get("backend") == "compiled-attention":
            reference = read_json(root / "request/reference.json")
            reference["prefill_backend"] = "compiled-attention"
            reference["prefill_cache_policy"] = "shared-input"
            save(root / "request/reference.json", reference)
        manifest = finish_bundle(
            root,
            sdk_hashes,
            options,
            source_native_manifest_sha256=sha256(package / "compiled-model.txt"),
        )
        root.rename(output)
    return manifest
