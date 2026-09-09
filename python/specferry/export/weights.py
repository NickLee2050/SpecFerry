"""Stream validated tensors into an aligned, little-endian deployment weight pack."""

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

from specferry.reference.checkpoint import MODEL_ID, REVISION, local_path, sha256

from .schema import expected_text_tensors

ALIGNMENT = 64


def convert_bytes(raw: bytes, source_dtype: str) -> tuple[bytes, str]:
    if source_dtype == "BF16":
        if len(raw) % 2:
            raise ValueError("unaligned BF16 input")
        values = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
        target = "F16"
        with np.errstate(over="raise", invalid="raise"):
            converted = values.astype("<f2")
    elif source_dtype == "F32":
        if len(raw) % 4:
            raise ValueError("unaligned FP32 input")
        converted = np.frombuffer(raw, dtype="<f4")
        target = "F32"
    else:
        raise ValueError(f"unsupported source precision: {source_dtype}")
    if not np.isfinite(converted).all():
        raise ValueError("nonfinite weight or FP16 overflow")
    return converted.tobytes(), target


def tensor_blocks(root: Path, entry: dict, chunk_bytes: int):
    path = local_path(root, entry["file"])
    with path.open("rb") as source:
        header_bytes = source.read(8)
        if len(header_bytes) != 8:
            raise ValueError("truncated weight header")
        header_size = struct.unpack("<Q", header_bytes)[0]
        start, end = entry["data_offsets"]
        source.seek(8 + header_size + start)
        remaining = end - start
        while remaining:
            block = source.read(min(chunk_bytes, remaining))
            if not block:
                raise ValueError(f"truncated source tensor: {entry['text_name']}")
            remaining -= len(block)
            yield block


def write_weight_pack(
    root: Path, entries: list[dict], output: Path, chunk_bytes=8 * 1024 * 1024, head_block_rows=4096
) -> list[dict]:
    """Call only after checkpoint inventory validation; this function never loads a model."""
    if chunk_bytes < 4 or chunk_bytes % 4 or head_block_rows < 1:
        raise ValueError(
            "chunk bytes must be a positive multiple of four; block rows must be positive"
        )
    records = []
    with (output / "weights.bin").open("xb") as destination:
        for entry in entries:
            padding = (-destination.tell()) % ALIGNMENT
            destination.write(bytes(padding))
            offset = destination.tell()
            digest = hashlib.sha256()
            target_dtype = None
            for block in tensor_blocks(root, entry, chunk_bytes):
                converted, target_dtype = convert_bytes(block, entry["dtype"])
                destination.write(converted)
                digest.update(converted)
            size = destination.tell() - offset
            if size != math.prod(entry["shape"]) * (4 if target_dtype == "F32" else 2):
                raise ValueError(f"converted size mismatch: {entry['text_name']}")
            record = {
                "name": entry["text_name"],
                "source_name": entry["source_name"],
                "source_file": entry["file"],
                "source_dtype": entry["dtype"],
                "source_data_offsets": entry["data_offsets"],
                "dtype": target_dtype,
                "shape": entry["shape"],
                "sdk_shape": list(reversed(entry["shape"])),
                "offset": offset,
                "bytes": size,
                "sha256": digest.hexdigest(),
                "layout": "row_major_source_axes",
                "transformed": entry["dtype"] != target_dtype,
            }
            if len(entry["shape"]) == 2:
                rows, columns = entry["shape"]
                block_rows = min(head_block_rows, rows)
                element_bytes = 4 if target_dtype == "F32" else 2
                record["row_blocks"] = [
                    {
                        "first_row": first,
                        "rows": min(block_rows, rows - first),
                        "offset": offset + first * columns * element_bytes,
                        "bytes": min(block_rows, rows - first) * columns * element_bytes,
                    }
                    for first in range(0, rows, block_rows)
                ]
                record["projection"] = "Y[1,N] = X[1,K] @ W[N,K].T"
            records.append(record)
    return records


def write_native_index(output: Path, records: list[dict]) -> None:
    lines = ["specferry-np101-weights 1"]
    for record in records:
        shape = ",".join(map(str, record["sdk_shape"]))
        lines.append(
            f"{record['name']} {record['dtype']} {shape} {record['offset']} "
            f"{record['bytes']} {record['sha256']}"
        )
    (output / "weights.index").write_text("\n".join(lines) + "\n")


def verify_weight_pack(root: Path) -> dict:
    """Verify both complete pack integrity and each tensor's range/digest."""
    manifest = json.loads((root / "deployment-manifest.json").read_text())
    if (
        manifest.get("format") != "specferry-np101-weights-v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("incomplete or unsupported deployment manifest")
    for name in ("weights.bin", "weights.index"):
        record = manifest["files"][name]
        path = local_path(root, name)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"export file integrity failure: {name}")
    names = set()
    end = 0
    with (root / "weights.bin").open("rb") as packed:
        for tensor in manifest["tensors"]:
            shape = tensor["shape"]
            if (
                not shape
                or len(shape) > 8
                or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            ):
                raise ValueError("invalid export tensor shape")
            size = math.prod(tensor["shape"]) * {"F16": 2, "F32": 4}[tensor["dtype"]]
            if (
                tensor["name"] in names
                or tensor["offset"] != end + (-end) % ALIGNMENT
                or tensor["bytes"] != size
                or tensor["sdk_shape"] != tensor["shape"][::-1]
            ):
                raise ValueError("duplicate tensor, invalid shape, offset, or byte count")
            names.add(tensor["name"])
            packed.seek(tensor["offset"])
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                block = packed.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("truncated export tensor")
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != tensor["sha256"]:
                raise ValueError(f"tensor digest mismatch: {tensor['name']}")
            if len(shape) == 2:
                row_bytes = shape[1] * {"F16": 2, "F32": 4}[tensor["dtype"]]
                next_row = 0
                for block in tensor.get("row_blocks", []):
                    rows = block["rows"]
                    if (
                        type(rows) is not int
                        or rows <= 0
                        or block["first_row"] != next_row
                        or block["offset"] != tensor["offset"] + next_row * row_bytes
                        or block["bytes"] != rows * row_bytes
                        or next_row + rows > shape[0]
                    ):
                        raise ValueError("invalid or overlapping weight row block")
                    next_row += rows
                if next_row != shape[0]:
                    raise ValueError("weight row blocks do not cover every row")
            end = tensor["offset"] + size
    if end != (root / "weights.bin").stat().st_size:
        raise ValueError("unexpected trailing export data")
    if manifest["aliases"] != {"lm_head.weight": "model.embed_tokens.weight"}:
        raise ValueError("embedding/head sharing contract changed")
    if "model.embed_tokens.weight" not in names:
        raise ValueError("shared embedding/head target is missing")
    return {"status": "verified", "tensors": len(names), "bytes": end}


def verify_export(root: Path) -> dict:
    result = verify_weight_pack(root)
    manifest = json.loads((root / "deployment-manifest.json").read_text())
    if manifest.get("repo_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise ValueError("deployment does not identify the fixed first DLM revision")
    expected = expected_text_tensors(manifest["text_config"])
    records = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    if set(records) != set(expected):
        raise ValueError("deployment is missing text tensors or contains unexpected tensors")
    lines = ["specferry-np101-weights 1"]
    for record in manifest["tensors"]:
        shape, source_dtype = expected[record["name"]]
        dtype = "F32" if source_dtype == "F32" else "F16"
        if (
            record["shape"] != shape
            or record["dtype"] != dtype
            or record["source_dtype"] != source_dtype
        ):
            raise ValueError("deployment tensor violates the fixed shape/precision contract")
        sdk_shape = ",".join(map(str, record["sdk_shape"]))
        lines.append(
            f"{record['name']} {record['dtype']} {sdk_shape} {record['offset']} "
            f"{record['bytes']} {record['sha256']}"
        )
    if (root / "weights.index").read_text() != "\n".join(lines) + "\n":
        raise ValueError("native index disagrees with deployment manifest")
    return result
