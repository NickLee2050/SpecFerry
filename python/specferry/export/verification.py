"""Independent, chunked PyTorch verification of all exported parameter bytes."""

import json
import math
from pathlib import Path

from specferry.reference.checkpoint import inventory

from .schema import validate_text_entries
from .weights import verify_export


def compare_checkpoint(model: Path, deployment: Path) -> dict:
    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)
    verify_export(deployment)
    source = inventory(model)
    entries = {entry["text_name"]: entry for entry in validate_text_entries(source)}
    manifest = json.loads((deployment / "deployment-manifest.json").read_text())
    verified_bytes = 0
    with (deployment / "weights.bin").open("rb") as packed:
        for record in manifest["tensors"]:
            entry = entries[record["name"]]
            target = torch.float32 if entry["dtype"] == "F32" else torch.float16
            row_elements = math.prod(entry["shape"][1:])
            rows_per_chunk = max(1, (4 * 1024 * 1024) // row_elements)
            packed.seek(record["offset"])
            with safe_open(model / entry["file"], framework="pt", device="cpu") as weights:
                tensor = weights.get_slice(entry["source_name"])
                for first in range(0, entry["shape"][0], rows_per_chunk):
                    values = tensor[first : first + rows_per_chunk].to(target).contiguous()
                    expected = values.view(torch.uint8).numpy().tobytes()
                    actual = packed.read(len(expected))
                    if actual != expected:
                        raise ValueError(
                            f"export differs from PyTorch conversion: {record['name']} "
                            f"starting at row {first}"
                        )
                    verified_bytes += len(expected)
    return {
        "status": "passed",
        "comparison": "bitwise_against_pytorch_dtype_conversion",
        "tensors": len(entries),
        "verified_payload_bytes": verified_bytes,
        "torch": torch.__version__,
        "source_revision": source["revision"],
    }
