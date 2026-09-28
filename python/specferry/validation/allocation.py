"""Memory test input checks and separate allocation/integrity outcomes."""

import json
import math
from pathlib import Path

from specferry.data.checkpoint import sha256 as fingerprint

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
    from specferry.export.weights import verify_weight_pack

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
    full_scan = (
        allocated
        and readback == "all"
        and report.get("scanned_bytes") == uploaded
        and report.get("scanned_blocks", 0) == report.get("retained_tensors", -1)
        and report.get("size_mismatch_blocks", 0) == 0
    )
    intact = intact and (readback != "all" or full_scan)
    return {
        "allocation_and_release_passed": allocated,
        "readback_and_release_passed": intact,
        "target_reached": allocated and uploaded == target,
        "next_block_rejected": allocated and report.get("rejected_block_bytes", 0) > 0,
        "full_scan_completed": full_scan,
        "retained_payload_bytes": uploaded,
        "scanned_bytes": report.get("scanned_bytes", 0),
        "mismatched_bytes": report.get("mismatched_bytes") if readback == "all" else None,
        "mismatched_blocks": report.get("mismatched_blocks") if readback == "all" else None,
        "mismatched_ranges": report.get("mismatched_ranges") if readback == "all" else None,
        "matching_bytes": uploaded - report["mismatched_bytes"] if full_scan else None,
        "physical_memory_limit_proven": False,
    }


def print_capacity_summary(result, report):
    retained = result["retained_payload_bytes"]
    print(f"Retained payload: {retained / MIB:g} / {report.get('target_bytes', 0) / MIB:g} MiB")
    if result["next_block_rejected"]:
        print(f"Next block rejected during {report['rejection_phase']}; see SDK error in sdk.log")
    else:
        print("No next-block rejection recorded; the allocation limit is not established.")
    if report.get("readback_mode") == "all":
        print(
            f"Readback coverage: {result['scanned_bytes']} / {retained} bytes; "
            f"{report.get('scanned_blocks', 0)} / {report.get('retained_tensors', 0)} blocks"
        )
        print(f"Full scan completed: {'yes' if result['full_scan_completed'] else 'no'}")
        print(
            f"Differences: {result['mismatched_bytes']} bytes in "
            f"{result['mismatched_blocks']} blocks, {result['mismatched_ranges']} byte ranges"
        )
        print("All scanned block results: readback-blocks.jsonl (including blocks with no errors)")
    print("Coverage describes retained application tensors, not all physical board memory.")


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
