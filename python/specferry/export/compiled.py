"""Device-free compilation and generated-wrapper port contracts shared by model adapters."""

import re
import subprocess

import yaml

from specferry.validation.device import write_json

PEGASUS = "/root/Quantize_toolkit/quantize-toolkit-binary/bin/pegasus"
SIMULATOR = "/usr/local/VeriSilicon/VivanteIDE5.6.0/cmdtools/vsimulator"


def compile_model(rootfs, directory, precision="float"):
    """Expose no galcore; all writable compiler outputs remain in the requested directory."""
    if precision not in ("float", "float32"):
        raise ValueError("unsupported compiler precision policy")
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
        "/work",
        "--setenv",
        "PATH",
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "TF_CPP_MIN_LOG_LEVEL",
        "2",
        "--setenv",
        "OMP_NUM_THREADS",
        "2",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "2",
        "--",
        PEGASUS,
    ]

    def invoke(label, arguments):
        command = prefix + arguments
        with (directory / f"{label}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
        write_json(directory / f"{label}.json", {"command": command})
        print(directory.name, label, "completed", flush=True)

    invoke(
        "import",
        [
            "import",
            "onnx",
            "--model",
            "model.onnx",
            "--output-model",
            "model.json",
            "--output-data",
            "model.data",
        ],
    )
    invoke(
        "inputmeta",
        [
            "generate",
            "inputmeta",
            "--model",
            "model.json",
            "--separated-database",
            "--input-meta-output",
            "inputmeta.yml",
        ],
    )
    invoke(
        "postprocess",
        [
            "generate",
            "postprocess-file",
            "--model",
            "model.json",
            "--postprocess-file-output",
            "postprocess.yml",
        ],
    )
    meta_path = directory / "inputmeta.yml"
    meta = yaml.safe_load(meta_path.read_text())
    for database in meta["input_meta"]["databases"]:
        for port in database["ports"]:
            port["category"] = "undefined"
            port["preprocess"]["reverse_channel"] = False
            params = port["preprocess"]["preproc_node_params"]
            if params["add_preproc_node"] or params["preproc_type"] != "TENSOR":
                raise ValueError("unexpected image preprocessing in projection export")
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    post_path = directory / "postprocess.yml"
    post = yaml.safe_load(post_path.read_text())
    for port in post["postprocess"]["app_postprocs"]:
        port["postproc_params"]["force_float32"] = False
    post_path.write_text(yaml.safe_dump(post, sort_keys=False))
    invoke(
        "export",
        [
            "export",
            "ovxlib",
            "--model",
            "model.json",
            "--model-data",
            "model.data",
            "--dtype",
            precision,
            "--target-ide-project",
            "linux64",
            "--with-input-meta",
            "inputmeta.yml",
            "--output-path",
            "export_model/network",
            "--pack-nbg-unify",
            "--postprocess-file",
            "postprocess.yml",
            "--optimize",
            "VIP9000LD_PLUS_PID0X1000001D",
            "--viv-sdk",
            SIMULATOR,
        ],
    )


def public_port_order(directory, direction):
    """Return node-port indices in the generated wrapper's public graph order."""
    if direction not in ("input", "output"):
        raise ValueError("unsupported port direction")
    wrapper = (directory / "export_model_nbg_unify/vnn_network.c").read_text()
    node = re.findall(
        r"node\[0\]->" + direction + r"\.tensors\[(\d+)\] = norm_tensor\[(\d+)\]", wrapper
    )
    public = re.findall(
        r"graph->" + direction + r"\.tensors\[(\d+)\] = norm_tensor\[(\d+)\]", wrapper
    )
    count = len(public)
    node = node[:count]
    if not count or [int(slot) for slot, _ in node] != list(range(count)):
        raise ValueError("compiled wrapper node port contract differs")
    lookup = {int(tensor): int(slot) for slot, tensor in node}
    order = [lookup[int(tensor)] for _, tensor in sorted(public, key=lambda pair: int(pair[0]))]
    if sorted(order) != list(range(count)):
        raise ValueError("compiled wrapper public ports are not a permutation")
    return order


def compiled_ports(directory, source, *, saved_public_graph=False):
    """Use generated C dtype/shape and binding order; metadata dtype alone is insufficient."""
    text = (directory / "export_model_nbg_unify/vnn_network.c").read_text()
    definitions = {}
    roles = list(source.graph.input) + list(source.graph.output)
    pattern = r"/\*\s*@([^*]+):out0\s*\*/(.*?NEW_NORM_TENSOR\(norm_tensor\[(\d+)\], attr, VSI_NN_TYPE_(\w+)\))"
    for name, body, tensor, dtype in re.findall(pattern, text, re.S):
        matches = [
            port
            for port in roles
            if name.startswith(port.name + "_")
            or name == port.name
            or "_" + port.name + "/out0" in name
        ]
        if len(matches) != 1:
            raise ValueError(f"unresolved generated wrapper port: {name}")
        rank = int(re.search(r"attr.dim_num\s*=\s*(\d+)", body)[1])
        shape = [
            int(value) for _, value in re.findall(r"attr.size\[(\d+)\]\s*=\s*(\d+)", body)[:rank]
        ]
        definitions[int(tensor)] = dict(
            name=matches[0].name,
            dtype={"FLOAT16": "F16", "FLOAT32": "F32", "INT32": "I32"}[dtype],
            shape=shape,
        )
    result = {}
    for direction in ("input", "output"):
        assignments = re.findall(
            r"node\[0\]->" + direction + r"\.tensors\[(\d+)\] = norm_tensor\[(\d+)\]", text
        )
        count = len(source.graph.input if direction == "input" else source.graph.output)
        result[direction + "s"] = [definitions[int(tensor)] for _, tensor in assignments[:count]]
        result[direction + "_order"] = public_port_order(directory, direction)
        if len(result[direction + "s"]) != count:
            raise ValueError("generated port count differs")
        if saved_public_graph:
            # Saving the generated graph creates a new NBG whose kernel ports follow
            # that graph's public IO, rather than the original NBG node's bindings.
            result[direction + "s"] = [
                result[direction + "s"][index] for index in result[direction + "_order"]
            ]
            result[direction + "_order"] = list(range(count))
    return result
