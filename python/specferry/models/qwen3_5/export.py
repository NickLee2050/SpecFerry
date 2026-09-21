"""Strict Qwen deployment validation and tensor precision selection."""

import json
from pathlib import Path

from specferry.export.weights import verify_weight_pack

from .checkpoint import MODEL_ID, REVISION
from .schema import expected_text_tensors, validate_text_entries


def deployment_entries(metadata):
    return [
        dict(entry, target_dtype="F32" if entry["dtype"] == "F32" else "F16")
        for entry in validate_text_entries(metadata)
    ]


def verify_export(root: Path) -> dict:
    result = verify_weight_pack(root)
    manifest = json.loads((root / "deployment-manifest.json").read_text())
    if manifest.get("repo_id") != MODEL_ID or manifest.get("revision") != REVISION:
        raise ValueError("deployment does not identify the fixed first DLM revision")
    if manifest["aliases"] != {"lm_head.weight": "model.embed_tokens.weight"}:
        raise ValueError("embedding/head sharing contract changed")
    expected = expected_text_tensors(manifest["text_config"])
    records = {tensor["name"]: tensor for tensor in manifest["tensors"]}
    if set(records) != set(expected):
        raise ValueError("deployment is missing text tensors or contains unexpected tensors")
    for record in manifest["tensors"]:
        shape, source_dtype = expected[record["name"]]
        dtype = "F32" if source_dtype == "F32" else "F16"
        if (
            record["shape"] != shape
            or record["dtype"] != dtype
            or record["source_dtype"] != source_dtype
        ):
            raise ValueError("deployment tensor violates the fixed shape/precision contract")
    return result
