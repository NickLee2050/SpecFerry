"""Strict local FP16 PyTorch checkpoint loading without executing pickle globals."""

import json

import torch

from specferry.data.checkpoint import sha256, verify_download

from .config import CHECKPOINT_SHA256, MODEL_ID, REVISION, checkpoint_shapes


def load_checkpoint(root):
    manifest = verify_download(root, repo_id=MODEL_ID, revision=REVISION)
    files = {entry["path"]: entry for entry in manifest["files"]}
    if not {"config.json", "pytorch_model.bin"} <= files.keys():
        raise ValueError("OPT download does not contain the required config and weights")
    if files["pytorch_model.bin"].get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("OPT checkpoint does not match the locked source digest")
    config = json.loads((root / "config.json").read_text())
    expected = checkpoint_shapes(config)
    weights = torch.load(
        root / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("OPT checkpoint contains missing or unexpected tensors")
    inventory = []
    for name, shape in expected.items():
        value = weights[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float16
            or list(value.shape) != shape
        ):
            raise ValueError(f"OPT weight shape/dtype mismatch: {name}")
        if not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError(f"OPT weight must be finite and contiguous: {name}")
        inventory.append({"name": name, "shape": shape, "dtype": "F16", "bytes": value.numel() * 2})
    return weights, {
        "repo_id": MODEL_ID,
        "revision": REVISION,
        "config": config,
        "source_sha256": CHECKPOINT_SHA256,
        "download_manifest_sha256": sha256(root / "download-manifest.json"),
        "tensors": inventory,
        "weight_bytes": sum(item["bytes"] for item in inventory),
    }
