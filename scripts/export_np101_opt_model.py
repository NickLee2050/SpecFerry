#!/usr/bin/env python3
"""Export complete fixed OPT models or all decoders with explicit state precision."""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import onnx

from specferry.data.checkpoint import sha256
from specferry.export.compiled import SIMULATOR, compile_model, compiled_ports
from specferry.models.opt.checkpoint import load_checkpoint
from specferry.models.opt.compiled import DRIVER, SDK_FILES
from specferry.models.opt.export import verify_export
from specferry.models.opt.onnx import decoder_model, full_model, select_cache_updates
from specferry.validation.device import write_json


def repair_selection(rootfs, directory):
    """Rebuild the known converter's narrowed ArgMax intermediate in a device-free SDK."""
    source = directory / "export_model/vnn_network.c"
    text = source.read_text()
    blocks = re.findall(r"NEW_VXNODE\(node\[(\d+)\], VSI_NN_OP_ARGMAX[^;]+;", text)
    if len(blocks) != 1:
        raise ValueError("full OPT export requires one ArgMax")
    tensor = "node[" + blocks[0] + "]->output.tensors[0]"
    pattern = (
        r"(NEW_(?:VIRTUAL|NORM)_TENSOR\(" + re.escape(tensor) + r", attr, VSI_NN_TYPE_)INT16(\))"
    )
    changed, count = re.subn(pattern, r"\g<1>INT32\2", text)
    if count != 1 and not re.search(pattern.replace("INT16", "INT32"), text):
        raise ValueError("ArgMax intermediate differs from the known narrowing contract")
    source.write_text(changed)
    # Reuse the generated application, including tensor preprocessing and one
    # simulator Run. A Verify-only re-save returned success but its NBG was
    # rejected at import on v105; it must not be treated as a usable artifact.
    model = onnx.load(directory / "model.onnx")
    inputs = []
    for port in model.graph.input:
        shape = [dimension.dim_value for dimension in port.type.tensor_type.shape.dim]
        values = np.zeros(shape, dtype="f4")
        if port.name == "token":
            values.fill(2)
        elif port.name == "kv_indices":
            values[:, 0] = np.arange(16)
        elif port.name == "attention_mask":
            values[:, 1:] = -1e9
        filename = port.name + ".tensor"
        np.savetxt(directory / "export_model" / filename, values.reshape(-1), fmt="%.9g")
        inputs.append(filename)
    prefix = [
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--ro-bind",
        str(rootfs),
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(directory),
        "/work",
        "--chdir",
        "/work/export_model",
        "--setenv",
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--setenv",
        "VIVANTE_SDK_DIR",
        SIMULATOR,
        "--setenv",
        "VSIMULATOR_CONFIG",
        SIMULATOR + "/../common/cfg/VIP9000LD_PLUS_PID0X1000001D.config",
        "--setenv",
        "LD_LIBRARY_PATH",
        SIMULATOR + "/lib:" + SIMULATOR + "/lib/x64_linux:" + SIMULATOR + "/../common/lib",
        "--setenv",
        "VIV_VX_ENABLE_SAVE_NETWORK_BINARY",
        "1",
        "--setenv",
        "VIV_VX_SAVE_NETWORK_BINARY_PATH",
        "/work/repaired-run.nb",
        "--setenv",
        "VIV_VX_PROFILE",
        "0",
        "--",
    ]
    include = ["-I."] + [
        "-I" + SIMULATOR + "/include/" + part for part in ("", "VX", "CL", "ovxlib", "jpeg")
    ]
    libraries = [
        "-L" + SIMULATOR + "/lib",
        "-L" + SIMULATOR + "/lib/x64_linux",
        "-L" + SIMULATOR + "/../common/lib",
    ]
    libraries += [
        "-l" + name
        for name in (
            "OpenVX",
            "OpenVXU",
            "CLC",
            "VSC",
            "GAL",
            "ovxlib",
            "Emulator",
            "vdtproxy",
            "jpeg",
            "m",
        )
    ]
    for label, command in (
        (
            "selection-build",
            [
                "gcc",
                "-O2",
                "-DLINUX",
                "-D__linux__",
                *include,
                "main.c",
                "vnn_network.c",
                "vnn_pre_process.c",
                "vnn_post_process.c",
                *libraries,
                "-o",
                "compile_binary",
            ],
        ),
        ("selection-rebuild", ["./compile_binary", "network.export.data", *inputs]),
    ):
        with (directory / (label + ".log")).open("w") as log:
            subprocess.run(
                prefix + command, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True
            )
    if not (directory / "repaired-run.nb").is_file():
        raise ValueError("simulator application did not produce the corrected NBG")
    (directory / "export_model_nbg_unify/network_binary.nb").write_bytes(
        (directory / "repaired-run.nb").read_bytes()
    )
    write_json(
        directory / "selection-repair.json",
        dict(
            tensor=tensor,
            change="ArgMax intermediate INT16 to INT32",
            header_unchanged=True,
            simulator_run=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--rootfs", type=Path, required=True)
    parser.add_argument("--scope", choices=("fullgraph", "decoders"), required=True)
    parser.add_argument("--capacity", type=int, choices=(64, 128), required=True)
    parser.add_argument(
        "--kv-update",
        choices=("scatter", "select"),
        default="scatter",
        help="graph-internal FP32 cache update; select requires indices in head order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--package-output", type=Path, help="write a portable source-bound runtime package"
    )
    args = parser.parse_args()
    verify_export(args.deployment)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    weights, checkpoint = load_checkpoint(args.checkpoint)
    graphs = []
    for layer in [None] if args.scope == "fullgraph" else range(24):
        graph = (
            full_model(weights, args.capacity)
            if layer is None
            else decoder_model(weights, layer, args.capacity)
        )
        directory = output / ("model" if layer is None else f"layer{layer}")
        directory.mkdir()
        model = graph.model()
        if args.kv_update == "select":
            select_cache_updates(model)
        onnx.save(model, directory / "model.onnx")
        write_json(
            directory / "source.json",
            dict(
                kind=args.scope,
                layer=layer,
                capacity=args.capacity,
                kv_update=args.kv_update,
                weight_sha256=graph.identities,
                onnx_sha256=sha256(directory / "model.onnx"),
                constant_bytes=sum(len(t.raw_data) for t in graph.constants),
            ),
        )
        graphs.append((directory, model))
    del weights

    def compile_graph(entry):
        directory, model = entry
        compile_model(args.rootfs.resolve(), directory, precision="float32")
        if args.scope == "fullgraph":
            repair_selection(args.rootfs.resolve(), directory)
        binary = directory / "export_model_nbg_unify/network_binary.nb"
        header = (directory / "export_model/vnn_network.h").read_text()
        version = [
            int(re.search(rf"#define\s+VNN_VERSION_{part}\s+\(?([0-9]+)", header)[1])
            for part in ("MAJOR", "MINOR", "PATCH")
        ]
        write_json(
            directory / "artifact.json",
            dict(
                nbg_sha256=sha256(binary),
                graph_version=version,
                constant_bytes=max(
                    binary.stat().st_size,
                    (directory / "model.data").stat().st_size,
                    json.loads((directory / "source.json").read_text())["constant_bytes"],
                ),
                **compiled_ports(directory, model, saved_public_graph=args.scope == "fullgraph"),
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(compile_graph, graphs))
    sdk = Path("/usr/lib/ljmicro")
    write_json(
        output / "export.json",
        dict(
            kind=args.scope,
            capacity=args.capacity,
            checkpoint=checkpoint,
            sdk=dict(
                directory=str(sdk),
                module_srcversion=DRIVER,
                sha256={name: sha256(sdk / name) for name in SDK_FILES},
            ),
            graphs=[str(path) for path, _ in graphs],
        ),
    )
    if args.package_output:
        from specferry.models.opt.compiled_model import package_model_exports

        package_model_exports(output, args.deployment, args.package_output)


if __name__ == "__main__":
    main()
