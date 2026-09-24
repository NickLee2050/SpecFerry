"""Memory test input checks and separate allocation/integrity outcomes."""

import json
import math
from pathlib import Path

from specferry.data.checkpoint import sha256 as fingerprint
from specferry.export.weights import verify_weight_pack

from .device import clean_execution

MIB = 1024**2
BLOCK_BYTES = 8 * MIB
SEGMENT_LIMIT_BYTES = 1024 * MIB
STORAGE_MODES = ("constant", "mutable")


def check_segment_budget(weight_bytes: int, state_bytes: int, storage: str) -> dict:
    """Preflight known payloads; SDK workspace and hidden copies remain unknown."""
    if storage not in STORAGE_MODES or min(weight_bytes, state_bytes) < 0:
        raise ValueError("invalid storage mode or payload size")
    constant = weight_bytes if storage == "constant" else 0
    mutable = state_bytes + (weight_bytes if storage == "mutable" else 0)
    if max(constant, mutable) > SEGMENT_LIMIT_BYTES:
        raise ValueError(
            f"application segment budget exceeded: const={constant}, nonconst={mutable}, "
            f"limit={SEGMENT_LIMIT_BYTES} bytes each; device submission blocked"
        )
    return {
        "segment_limit_bytes": SEGMENT_LIMIT_BYTES,
        "const_bytes": constant,
        "nonconst_bytes": mutable,
        "sdk_internal_allocations_included": False,
    }


def state_payload_bytes(path: Path) -> int:
    """Validate an explicit state fixture before opening the device."""
    lines = path.read_text().splitlines()
    if len(lines) < 2 or lines[0] != "specferry-allocation-states 1":
        raise ValueError("empty or unsupported allocation state specification")
    total = 0
    widths = {"F16": 2, "F32": 4, "I32": 4, "BOOL": 1}
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != 3 or fields[0] not in widths:
            raise ValueError("invalid allocation state record")
        dtype, shape, copies = fields
        numbers = [*shape.split(","), copies]
        if any(not value.isascii() or not value.isdecimal() for value in numbers):
            raise ValueError("state dimensions and copies must be positive integers")
        dimensions = [int(value) for value in numbers[:-1]]
        count = int(copies)
        size = math.prod(dimensions) * widths[dtype]
        if (
            not 1 <= len(dimensions) <= 8
            or any(not 0 < dimension < 2**32 for dimension in dimensions)
            or not 1 <= count <= 1024
            or size > BLOCK_BYTES
        ):
            raise ValueError("state shape, copies or payload exceeds the supported bounds")
        total += size * count
    return total


def prepare_weight_check(deployment: Path, state_spec: Path | None, output: Path) -> dict:
    """Bind a model-independent readback check to a verified pack and optional states."""
    verified = verify_weight_pack(deployment)
    if not verified["tensors"]:
        raise ValueError("weight check requires a nonempty pack")
    manifest_path = deployment / "deployment-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    state_bytes = 0
    if state_spec is not None:
        snapshot = output / "allocation-states.txt"
        snapshot.write_bytes(state_spec.read_bytes())
        state_bytes = state_payload_bytes(snapshot)
    return {
        "deployment": str(deployment.resolve()),
        "repo_id": manifest.get("repo_id"),
        "revision": manifest.get("revision"),
        "manifest_sha256": fingerprint(manifest_path),
        "files": manifest["files"],
        "expected": {
            "expected_weights": verified["tensors"],
            "expected_weight_bytes": sum(record["bytes"] for record in manifest["tensors"]),
            "expected_state_bytes": state_bytes,
        },
        "state_spec_sha256": fingerprint(output / "allocation-states.txt") if state_spec else None,
        "architecture_validated": False,
    }


def capacity_result(report, evidence, target, storage, dtype, readback):
    mismatch = report.get("status") == "readback_mismatch"
    uploaded = report.get("uploaded_bytes", 0)
    allocated = (
        clean_execution(evidence, expected_returncode=int(mismatch))
        and report.get("status") in ("target_reached", "allocation_limit", "readback_mismatch")
        and report.get("released") is True
        and report.get("phase") == "complete"
        and not report.get("error")
        and not report.get("release_error")
        and report.get("target_bytes") == target
        and 0 < uploaded <= target
        and report.get("storage") == storage
        and report.get("dtype") == dtype
        and report.get("readback_mode") == readback
    )
    intact = (
        allocated
        and readback != "none"
        and not mismatch
        and report.get("verified_bytes") == uploaded
        and not report.get("readback_error")
    )
    return {
        "allocation_and_release_passed": allocated,
        "readback_and_release_passed": intact,
        "retained_payload_bytes": uploaded,
        "physical_memory_limit_proven": False,
    }


def weights_complete(report, evidence, storage, expected):
    return (
        clean_execution(evidence)
        and report.get("status") == "allocation_pass"
        and report.get("released") is True
        and not report.get("release_error")
        and report.get("weight_storage") == storage
        and report.get("verified_weight_bytes") == expected["expected_weight_bytes"]
        and report.get("verified_state_bytes") == expected["expected_state_bytes"]
        and report.get("completed_weights") == expected["expected_weights"]
    )
