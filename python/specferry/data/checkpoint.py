"""Validate local download manifests and indexed safetensors payloads."""

import hashlib
import json
import math
import struct
from pathlib import Path

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def local_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"unsafe checkpoint path: {name}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"checkpoint path escapes model directory: {name}")
    return path


def inventory(root: Path, *, repo_id: str, revision: str):
    manifest = json.loads((root / "download-manifest.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("repo_id") != repo_id
        or manifest.get("resolved_revision") != revision
    ):
        raise ValueError("a complete download manifest matching the requested identity is required")
    files = manifest["files"]
    for entry in files:
        path = local_path(root, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"missing or wrong-size checkpoint file: {path}")
        if entry.get("sha256") and sha256(path) != entry["sha256"]:
            raise ValueError(f"checkpoint SHA256 mismatch: {path}")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    selected = {entry["path"] for entry in files}
    if not set(index["weight_map"].values()) <= selected:
        raise ValueError("index references an unverified weight file")
    tensors = []
    names = set()
    for filename in sorted(set(index["weight_map"].values())):
        path = local_path(root, filename)
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError("truncated safetensors header")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if not 2 <= header_size <= min(100 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError("invalid safetensors header length")
            header = json.loads(stream.read(header_size))
        intervals = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in names or index["weight_map"].get(name) != filename:
                raise ValueError(f"duplicate tensor or index/header disagreement: {name}")
            names.add(name)
            shape = meta["shape"]
            if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
                raise ValueError(f"invalid shape: {name}")
            count = math.prod(shape)
            size = count * DTYPE_BYTES[meta["dtype"]]
            start, end = meta["data_offsets"]
            if start < 0 or end - start != size or end > path.stat().st_size - 8 - header_size:
                raise ValueError(f"invalid tensor offsets: {name}")
            intervals.append((start, end))
            tensors.append(
                {
                    "source_name": name,
                    "file": filename,
                    "shape": shape,
                    "dtype": meta["dtype"],
                    "parameters": count,
                    "bytes": size,
                    "data_offsets": [start, end],
                }
            )
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("overlapping or non-contiguous safetensors payload")
            cursor = end
        if cursor + 8 + header_size != path.stat().st_size:
            raise ValueError("unaccounted bytes in safetensors file")
    if names != set(index["weight_map"]):
        raise ValueError("some indexed tensors are missing from weight headers")
    if sum(tensor["bytes"] for tensor in tensors) != index["metadata"]["total_size"]:
        raise ValueError("index total_size does not match tensor payloads")
    config = json.loads((root / "config.json").read_text())
    return {
        "status": "verified",
        "repo_id": repo_id,
        "revision": revision,
        "download_manifest_sha256": sha256(root / "download-manifest.json"),
        "config": config,
        "tensors": sorted(tensors, key=lambda item: item["source_name"]),
    }
