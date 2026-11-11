"""Offline SoC adaptation of the existing single-token Qwen hybrid package."""

import math
import shutil
import tempfile
from pathlib import Path

from specferry.data.checkpoint import sha256
from specferry.models.qwen3_5.compiled import (
    DRIVER,
    SDK_FILES,
    native_manifest,
    validate_feed_forward_ports,
)
from specferry.models.qwen3_5.components import Components
from specferry.models.qwen3_5.export import verify_export

from .board_runner import bundle_file, read_json, save
from .deployment import finish_bundle, sdk_identity, stage_runtime

DECODER_LAYOUTS = ("separate", "fused-attention-norm")


def validate_package(package, deployment, *, ffn_batch_tokens=1):
    """Verify the original EP package once while assembling a new deployment."""
    verify_export(deployment)
    weights = read_json(deployment / "deployment-manifest.json")
    manifest = read_json(package / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "qwen-delta"
        or manifest.get("deployment_sha256") != sha256(deployment / "deployment-manifest.json")
        or manifest["sdk"].get("module_srcversion") != DRIVER
        or set(manifest["sdk"]["sha256"]) != set(SDK_FILES)
    ):
        raise ValueError("Qwen source package/deployment/SDK identity differs")
    records = {row["name"]: row for row in weights["tensors"]}
    delta_layers = [
        i
        for i, kind in enumerate(weights["text_config"]["layer_types"])
        if kind == "linear_attention"
    ]
    bound = 0
    for mixer, key, native, digest_key in (
        (True, "graphs", "compiled-mixers.txt", "native_manifest_sha256"),
        (False, "feed_forward", "compiled-feed-forward.txt", "feed_forward_manifest_sha256"),
    ):
        graphs = manifest[key]
        if [g["layer"] for g in graphs] != (delta_layers if mixer else list(range(24))):
            raise ValueError("Qwen package requires all ordered DeltaNet/FFN layers")
        for graph in graphs:
            layer = graph["layer"]
            prefix = f"model.layers.{layer}."
            normalized = graph.get("includes_input_norm", False)
            if type(normalized) is not bool or (normalized and not mixer):
                raise ValueError("Qwen input normalization contract differs")
            if mixer and (not normalized or graph.get("outer_product") != "multiply"):
                raise ValueError("initial Qwen SoC requires the accepted normalized mixer package")
            names = {
                name
                for name in records
                if (
                    name.startswith(prefix + "linear_attn.")
                    or (normalized and name == prefix + "input_layernorm.weight")
                    if mixer
                    else name.startswith(prefix + "mlp.")
                    or name == prefix + "post_attention_layernorm.weight"
                )
            }
            binary = bundle_file(package, graph["path"])
            minimum = sum(records[name]["bytes"] for name in names)
            if (
                binary.parent != package
                or graph["weight_sha256"] != {name: records[name]["sha256"] for name in names}
                or len(names) != (9 + normalized if mixer else 4)
                or graph["graph_version"] not in ([1, 1, 34], [1, 1, 35])
                or type(graph["constant_bytes"]) is not int
                or not max(minimum, binary.stat().st_size) <= graph["constant_bytes"] <= 2**30
                or sha256(binary) != graph["nbg_sha256"]
            ):
                raise ValueError("Qwen NBG source/path/hash/constant contract differs")
            if not mixer:
                validate_feed_forward_ports(graph, batch_tokens=ffn_batch_tokens)
            else:
                expected = [("F16", [1024, 1]), ("F32", [128, 128, 16]), ("F16", [4, 6144])]
                for direction in ("input", "output"):
                    order, ports = graph[direction + "_order"], graph[direction + "s"]
                    if (
                        sorted(order) != [0, 1, 2]
                        or len(ports) != 3
                        or [(ports[i]["dtype"], ports[i]["shape"]) for i in order] != expected
                    ):
                        raise ValueError("Qwen DeltaNet port precision/order differs")
            bound += graph["constant_bytes"]
        if sha256(package / native) != manifest[digest_key] or (
            package / native
        ).read_text() != native_manifest(graphs, mixer=mixer, sdk=manifest["sdk"]):
            raise ValueError("Qwen native manifest disagrees with its source contract")
    if bound > 2**30:
        raise ValueError("Qwen embedded constants exceed the separate 1 GiB budget")
    return manifest, weights


def validate_request(request, weights, deployment_digest):
    """Validate a C64 or C128 greedy fixture and its complete coarse oracle."""
    from transformers import GenerationConfig

    metadata, reference = read_json(request / "request.json"), read_json(request / "reference.json")
    policy = [int(value) for value in (request / "policy.txt").read_text().split()]
    tokens = [int(value) for value in (request / "tokens.txt").read_text().split()]
    if len(policy) != 6:
        raise ValueError("Qwen policy requires six integers")
    vocabulary, capacity, bos, eos, maximum, repeats = policy
    effective = metadata["selection"]["effective"]
    defaults = GenerationConfig(**GenerationConfig._get_default_generation_params())
    handled = {
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "max_length",
        "max_new_tokens",
        "transformers_version",
        "output_attentions",
        "output_hidden_states",
    }
    if (
        any(
            not name.startswith("_")
            and name not in handled
            and value != getattr(defaults, name, None)
            for name, value in effective.items()
        )
        or effective.get("output_attentions")
        or effective.get("output_hidden_states")
    ):
        raise ValueError("unsupported Qwen generation policy")
    if (
        vocabulary != 248320
        or capacity not in (64, 128, 512)
        or not 0 <= bos < vocabulary
        or type(effective.get("eos_token_id")) is not int
        or eos != effective["eos_token_id"]
        or not 0 <= eos < vocabulary
        or effective.get("do_sample") is not False
        or effective.get("num_beams") != 1
        or effective.get("repetition_penalty") != 1.0
        or metadata["deployment_sha256"] != deployment_digest
        or metadata["capacity"] != capacity
        or metadata["maximum"] != maximum
        or effective.get("max_new_tokens") != maximum
        or metadata["repeats"] != repeats
        or not 1 <= maximum <= 32
        or not 1 <= repeats <= 3
        or not tokens
        or tokens != metadata["tokens"]
        or len(tokens) > capacity
        or any(not 0 <= token < vocabulary for token in tokens)
        or reference["prompt_length"] != len(tokens)
        or metadata["reference"] != reference["reference"]
    ):
        raise ValueError("Qwen request/generation policy differs")
    generated = reference["tokens"]
    consumed = reference["consumed"]
    if (
        not 1 <= len(generated) <= maximum
        or any(type(t) is not int or not 0 <= t < vocabulary for t in generated)
        or consumed != len(tokens) + len(generated) - 1
        or consumed > capacity
        or eos in generated[:-1]
        or reference["stop_reason"]
        != (
            "eos"
            if generated[-1] == eos
            else "max_new_tokens"
            if len(generated) == maximum
            else "capacity"
        )
        or (reference["stop_reason"] == "capacity" and consumed != capacity)
    ):
        raise ValueError("Qwen oracle token/stop/capacity contract differs")
    expected = {
        "hidden": ("<f2", [1, 1, 1024]),
        "recurrent": ("<f4", [1, 16, 128, 128]),
        "keys": ("<f2", [1, 2, consumed, 256]),
        "values": ("<f2", [1, 2, consumed, 256]),
    }
    if set(reference["snapshots"]) != set(expected):
        raise ValueError("Qwen oracle requires hidden/recurrent/K/V snapshots")
    for name, (dtype, shape) in expected.items():
        if reference["snapshots"][name] != {"dtype": dtype, "shape": shape} or (
            request / f"expected.{name}.bin"
        ).stat().st_size != math.prod(shape) * (4 if dtype == "<f4" else 2):
            raise ValueError("Qwen oracle snapshot precision/size differs")
    components = Components.from_text_config(weights["text_config"], capacity=capacity)
    # Reuse the model adapter's exact native geometry, including all layer kinds.
    with tempfile.TemporaryDirectory() as temporary:
        components.write(Path(temporary))
        if (request / "components.txt").read_text() != (
            Path(temporary) / "components.txt"
        ).read_text():
            raise ValueError("Qwen native component geometry differs")
    batch_file = request / "ffn-batch-tokens.txt"
    batch_name = batch_file.read_text().strip() if batch_file.exists() else "1"
    if batch_name not in ("1", "8", "16"):
        raise ValueError("unsupported Qwen FFN batch width")
    ffn_batch_tokens = int(batch_name)
    layout_file = request / "decoder-layout.txt"
    decoder_layout = layout_file.read_text().strip() if layout_file.exists() else "separate"
    if (
        decoder_layout not in DECODER_LAYOUTS
        or metadata.get("decoder_layout", "separate") != decoder_layout
    ):
        raise ValueError("Qwen decoder layout metadata/option differs")
    if decoder_layout != "separate" and ffn_batch_tokens == 1:
        raise ValueError("optimized Qwen decoder layout requires batched FFN")
    state_commit = (request / "state-commit.txt").read_text().strip()
    if state_commit not in ("combined", "ffn"):
        raise ValueError("unsupported Qwen state commit policy")
    if state_commit == "ffn" and ffn_batch_tokens == 1:
        raise ValueError("FFN state commit requires batched FFN")
    if metadata.get("state_commit", "combined") != state_commit:
        raise ValueError("Qwen state commit metadata/option differs")
    fallback_layer = 9 if ffn_batch_tokens == 1 else -1
    options = {
        "embedding-lookup.txt": "gather-nd",
        "matmul-feed-forward-layer.txt": str(fallback_layer),
        "token-readback.txt": "mapped",
        "state-commit.txt": state_commit,
    }
    for name, value in options.items():
        if (request / name).read_text().strip() != value:
            raise ValueError("Qwen conservative execution option differs: " + name)
    head = (request / "head-backend.txt").read_text().strip()
    if head not in ("matmul", "shared-fcl", "shared-fcl-split"):
        raise ValueError("Qwen head must reuse the existing shared vocabulary")
    if ffn_batch_tokens > 1 and (
        capacity not in (128, 512)
        or head != "shared-fcl-split"
        or (request / "projections.txt").read_text().strip() != "shared-fcl"
        or (request / "kv-write-mode.txt").read_text().strip() != "mapped"
    ):
        raise ValueError("batched Qwen FFN requires a C128/C512 mapped shared FCL chain")
    return dict(
        capacity=capacity,
        maximum=maximum,
        repeats=repeats,
        head_backend=head,
        state_commit=state_commit,
        matmul_feed_forward_layer=fallback_layer,
        ffn_batch_tokens=ffn_batch_tokens,
        decoder_layout=decoder_layout,
        token_readback="mapped",
    )


def prepare_bundle(
    binary,
    deployment,
    package,
    request,
    sdk,
    output,
    *,
    projections="shared-fcl",
    kv_write_mode="copy-graph",
    ffn_batch_tokens=1,
    decoder_layout="separate",
    state_commit="combined",
):
    binary, deployment, package, request, sdk, output = map(
        lambda p: Path(p).resolve(), (binary, deployment, package, request, sdk, output)
    )
    if output.exists():
        raise ValueError("use a new bundle directory")
    if projections not in ("shared-fcl", "matmul"):
        raise ValueError("unsupported Qwen projection backend")
    if state_commit not in ("combined", "ffn"):
        raise ValueError("unsupported Qwen state commit policy")
    if state_commit == "ffn" and ffn_batch_tokens == 1:
        raise ValueError("FFN state commit requires batched FFN")
    if kv_write_mode not in ("copy-graph", "mapped"):
        raise ValueError("unsupported Qwen KV write mode")
    if kv_write_mode == "mapped" and projections != "shared-fcl":
        raise ValueError("mapped Qwen KV requires shared FCL projections")
    if type(ffn_batch_tokens) is not int or ffn_batch_tokens not in (1, 8, 16):
        raise ValueError("unsupported Qwen FFN batch width")
    if decoder_layout not in DECODER_LAYOUTS:
        raise ValueError("unsupported Qwen decoder layout")
    if decoder_layout != "separate" and ffn_batch_tokens == 1:
        raise ValueError("optimized Qwen decoder layout requires batched FFN")
    if ffn_batch_tokens > 1 and (projections != "shared-fcl" or kv_write_mode != "mapped"):
        raise ValueError("batched Qwen FFN requires mapped shared FCL projections")
    hashes = sdk_identity(sdk)
    manifest, weights = validate_package(package, deployment, ffn_batch_tokens=ffn_batch_tokens)
    options = validate_request(request, weights, manifest["deployment_sha256"])
    if ffn_batch_tokens > 1 and options["capacity"] not in (128, 512):
        raise ValueError("batched Qwen FFN requires C128 or C512")
    options.update(
        head_backend="shared-fcl-split",
        state_commit=state_commit,
        projections=projections,
        kv_write_mode=kv_write_mode,
        ffn_batch_tokens=ffn_batch_tokens,
        decoder_layout=decoder_layout,
        matmul_feed_forward_layer=9 if ffn_batch_tokens == 1 else -1,
    )
    source_sdk = manifest["sdk"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qwen-soc-bundle-", dir=output.parent) as temporary:
        root = Path(temporary)
        stage_runtime(root, binary, deployment, sdk, "specferry_qwen_generate")
        manifest["source_sdk"] = source_sdk
        manifest["sdk"] = {"profile": "np101-soc-riscv64", "sha256": hashes}
        for mixer, key, native, digest_key in (
            (True, "graphs", "compiled-mixers.txt", "native_manifest_sha256"),
            (False, "feed_forward", "compiled-feed-forward.txt", "feed_forward_manifest_sha256"),
        ):
            for graph in manifest[key]:
                shutil.copyfile(package / graph["path"], root / "model" / graph["path"])
            (root / "model" / native).write_text(
                native_manifest(manifest[key], mixer=mixer, sdk=manifest["sdk"])
            )
            manifest[digest_key] = sha256(root / "model" / native)
        save(root / "model/manifest.json", manifest)
        for name in (
            "request.json",
            "reference.json",
            "components.txt",
            "tokens.txt",
            "policy.txt",
            "embedding-lookup.txt",
            "matmul-feed-forward-layer.txt",
            "token-readback.txt",
            "state-commit.txt",
            *(f"expected.{name}.bin" for name in ("hidden", "recurrent", "keys", "values")),
        ):
            shutil.copyfile(bundle_file(request, name), root / "request" / name)
        request_dir = root / "request"
        (request_dir / "head-backend.txt").write_text(options["head_backend"] + "\n")
        (request_dir / "projections.txt").write_text(projections + "\n")
        (request_dir / "kv-write-mode.txt").write_text(kv_write_mode + "\n")
        (request_dir / "ffn-batch-tokens.txt").write_text(str(ffn_batch_tokens) + "\n")
        (request_dir / "decoder-layout.txt").write_text(decoder_layout + "\n")
        (request_dir / "state-commit.txt").write_text(state_commit + "\n")
        (request_dir / "matmul-feed-forward-layer.txt").write_text(
            str(options["matmul_feed_forward_layer"]) + "\n"
        )
        metadata = read_json(request_dir / "request.json")
        metadata.update(
            head_backend=options["head_backend"],
            state_commit=state_commit,
            projections=projections,
            kv_write_mode=kv_write_mode,
            ffn_batch_tokens=ffn_batch_tokens,
            decoder_layout=decoder_layout,
        )
        save(request_dir / "request.json", metadata)
        result = finish_bundle(
            root,
            hashes,
            options,
            model_family="qwen3_5",
            driver_parameters=dict(
                contiguousBase=0x70000000, contiguousSize=0x80000000, stuckDump=1
            ),
            source_package_manifest_sha256=sha256(package / "manifest.json"),
            source_native_manifest_sha256={
                name: sha256(package / name)
                for name in ("compiled-mixers.txt", "compiled-feed-forward.txt")
            },
        )
        root.rename(output)
    return result
