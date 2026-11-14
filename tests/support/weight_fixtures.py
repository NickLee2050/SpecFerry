"""Small model-independent checkpoint and weight-pack fixtures."""

import hashlib
import json
import struct

import torch


def source_tensor(root, name, values, dtype):
    raw = values.view(torch.uint8).numpy().tobytes()
    header = json.dumps(
        {name: {"dtype": dtype, "shape": list(values.shape), "data_offsets": [0, len(raw)]}}
    ).encode()
    filename = name.replace(".", "_") + ".safetensors"
    (root / filename).write_bytes(struct.pack("<Q", len(header)) + header + raw)
    return {
        "source_name": name,
        "text_name": name,
        "file": filename,
        "dtype": dtype,
        "target_dtype": "F32" if dtype == "F32" else "F16",
        "shape": list(values.shape),
        "data_offsets": [0, len(raw)],
    }


def complete_manifest(root, records, aliases=None):
    manifest = {
        "format": "specferry-np101-weights-v1",
        "status": "complete",
        "tensors": records,
        "aliases": aliases or {},
        "files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
            }
            for name in ("weights.bin", "weights.index")
        },
    }
    (root / "deployment-manifest.json").write_text(json.dumps(manifest))
    return manifest
