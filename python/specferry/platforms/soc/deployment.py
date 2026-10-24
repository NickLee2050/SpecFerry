"""Build relocatable SoC OPT bundles without changing source NBGs or EP packages."""

import shutil
import tempfile
from pathlib import Path

from specferry.models.opt.compiled import SDK_FILES
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
        "block": "1",
        "kv_transfer": "mapped-slot",
        "prefill_head": "every-token",
    }
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError("SoC bundle requires explicit compiled fullgraph C64/M1 mapped-slot")
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


def prepare_bundle(binary, deployment, package, request, sdk, output):
    binary, deployment, package, request, sdk, output = map(
        lambda path: Path(path).resolve(), (binary, deployment, package, request, sdk, output)
    )
    if output.exists():
        raise ValueError("use a new bundle directory")
    options = request_options(request)
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
