"""Bit-preserving OPT export through the common bounded weight-pack writer."""

import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from specferry.data.checkpoint import sha256
from specferry.export.memory import memory_budget
from specferry.export.weights import verify_weight_pack, write_native_index, write_weight_pack
from specferry.validation.device import write_json

from .checkpoint import load_checkpoint
from .config import MODEL_ID, REVISION, checkpoint_shapes, validate_config


def verify_export(root):
    result = verify_weight_pack(root)
    manifest = json.loads((root / "deployment-manifest.json").read_text())
    if (manifest.get("repo_id"), manifest.get("revision")) != (MODEL_ID, REVISION):
        raise ValueError("export is not the locked OPT-350M checkpoint")
    expected = checkpoint_shapes(manifest["config"])
    records = {entry["name"]: entry for entry in manifest["tensors"]}
    if set(records) != set(expected) or manifest.get("aliases") != {
        "lm_head.weight": "decoder.embed_tokens.weight"
    }:
        raise ValueError("OPT export tensor names or tied-head alias disagree")
    for name, shape in expected.items():
        entry = records[name]
        if (
            entry["shape"] != shape
            or entry["dtype"] != "F16"
            or entry["source_dtype"] != "F16"
            or entry["transformed"]
        ):
            raise ValueError(f"OPT export must preserve source FP16 tensors: {name}")
    return result


def compare_source(weights, deployment):
    """Compare directly with original PyTorch storage, without the packer's conversion."""
    manifest = json.loads((deployment / "deployment-manifest.json").read_text())
    with (deployment / "weights.bin").open("rb") as packed:
        for record in manifest["tensors"]:
            source = memoryview(weights[record["name"]].numpy()).cast("B")
            packed.seek(record["offset"])
            for offset in range(0, len(source), 8 * 1024**2):
                expected = source[offset : offset + 8 * 1024**2]
                if packed.read(len(expected)) != expected:
                    raise ValueError(
                        f"OPT exported bytes differ from the checkpoint: {record['name']}"
                    )
    return {
        "status": "verified",
        "tensors": len(manifest["tensors"]),
        "source_bytes_preserved": True,
        "precision_conversion": False,
    }


def export_checkpoint(model, output, capacity=512):
    if output.exists():
        raise ValueError("use a new export directory, or --verify-only")
    weights, metadata = load_checkpoint(model)
    config = replace(validate_config(metadata["config"]), capacity=capacity)
    entries = [
        {
            "source_name": name,
            "text_name": name,
            "file": "pytorch_model.bin",
            "dtype": "F16",
            "target_dtype": "F16",
            "shape": list(value.shape),
        }
        for name, value in weights.items()
    ]

    def blocks(root, entry, chunk_bytes):
        source = memoryview(weights[entry["source_name"]].numpy()).cast("B")
        for offset in range(0, len(source), chunk_bytes):
            yield source[offset : offset + chunk_bytes]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        records = write_weight_pack(model, entries, staging, block_reader=blocks)
        write_native_index(staging, records)
        budget = memory_budget(
            records,
            {"kv_cache": config.state_bytes(range(config.layers))},
            capacity=capacity,
            pool_bytes=1024**3,
        )
        manifest = {
            "format": "specferry-np101-weights-v1",
            "status": "complete",
            "repo_id": MODEL_ID,
            "revision": REVISION,
            "config": metadata["config"],
            "source_sha256": metadata["source_sha256"],
            "source_manifest_sha256": metadata["download_manifest_sha256"],
            "tensors": records,
            "aliases": {"lm_head.weight": "decoder.embed_tokens.weight"},
            "files": {
                name: {"bytes": (staging / name).stat().st_size, "sha256": sha256(staging / name)}
                for name in ("weights.bin", "weights.index")
            },
            "precision_policy": "Original FP16 bytes; no dtype conversion or norm folding",
            "memory_budget": budget,
            "deployment_ready": False,
        }
        write_json(staging / "deployment-manifest.json", manifest)
        write_json(staging / "memory-budget.json", budget)
        verify_export(staging)
        result = compare_source(weights, staging)
        write_json(staging / "source-verification.json", result)
        staging.rename(output)
    return result


def read_layer_weights(deployment, records, layer):
    """Load only a selected layer into the independent official reference."""
    import numpy as np
    import torch

    prefix = f"decoder.layers.{layer}."
    result = {}
    with (deployment / "weights.bin").open("rb") as source:
        for record in records:
            if record["name"].startswith(prefix):
                source.seek(record["offset"])
                data = source.read(record["bytes"])
                if hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError("OPT layer payload digest mismatch")
                value = np.frombuffer(data, dtype="<f2").reshape(record["shape"]).copy()
                result[record["name"].removeprefix(prefix)] = torch.from_numpy(value)
    return result
