"""Shared execution and acceptance contracts for allocation-only diagnostics."""

import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path

from specferry.export.weights import verify_weight_pack

from .device import (
    RECOVERY_ROOT,
    clean_execution,
    device_lock,
    fingerprint,
    run_device,
    snapshot_binary,
)

MIB = 1024**2
BLOCK_BYTES = 8 * MIB
MAXIMUM_CAPACITY_MIB = 4096
# Keep historical report validation separate from the current submission ceiling.
SEGMENT_LIMIT_MIB = 1024
SEGMENT_LIMIT_BYTES = SEGMENT_LIMIT_MIB * MIB
REPORT_VERSION = 2
PATTERN = "splitmix64-finite-v1"
STORAGE_MODES = ("constant", "mutable")
FLOAT_DTYPES = ("F16", "F32")


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


def add_run_arguments(parser: ArgumentParser, binary: Path, timeout: int) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=binary)
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--timeout", type=int, default=timeout)


def run_allocation(
    binary: Path, arguments: list[str], output: Path, sdk_lib: Path, timeout: int
) -> dict:
    with device_lock(RECOVERY_ROOT):
        if not os.access("/dev/galcore", os.R_OK | os.W_OK):
            raise RuntimeError("/dev/galcore is unavailable or not readable/writable")
        snapshot = snapshot_binary(binary, output / binary.name)
        # run_device records the snapshot hash and SDK fingerprints in one place.
        return run_device(snapshot, arguments, output, sdk_lib, timeout)


def clean_readback(report: dict) -> bool:
    return (
        report.get("report_version") == REPORT_VERSION
        and report.get("phase") == "complete"
        and report.get("released") is True
        and all(
            report.get(field) == ""
            for field in ("error", "release_error", "readback_error", "failure_phase")
        )
        and "first_mismatch" in report
        and report["first_mismatch"] is None
    )


def positive_integer(value) -> bool:
    return type(value) is int and value > 0


def capacity_payload_matches(report: dict, storage: str, dtype: str, target_mib: int) -> bool:
    """Check allocation counts and rejection boundaries without judging readback."""
    uploaded = report.get("uploaded_bytes")
    rejected = report.get("rejected_block_bytes")
    target = target_mib * MIB
    if not (
        storage in STORAGE_MODES
        and dtype in FLOAT_DTYPES
        and 1 <= target_mib <= MAXIMUM_CAPACITY_MIB
        and report.get("storage") == storage
        and report.get("dtype") == dtype
        and report.get("pattern") == PATTERN
        and report.get("target_bytes") == target
        and report.get("block_bytes") == BLOCK_BYTES
        and positive_integer(uploaded)
        and uploaded <= target
        and report.get("retained_tensors") == (uploaded + BLOCK_BYTES - 1) // BLOCK_BYTES
        and type(rejected) is int
    ):
        return False
    if rejected == 0:
        return (
            uploaded == target
            and report.get("allocation_error") == ""
            and report.get("status") in ("target_reached", "readback_mismatch")
        )
    return (
        report.get("status") in ("allocation_limit", "readback_mismatch")
        and uploaded < target
        and uploaded % BLOCK_BYTES == 0
        and rejected == min(BLOCK_BYTES, target - uploaded)
        and isinstance(report.get("allocation_error"), str)
        and bool(report["allocation_error"])
    )


def capacity_complete(
    report: dict, evidence: dict, storage: str, dtype: str, target_mib: int
) -> bool:
    return (
        capacity_payload_matches(report, storage, dtype, target_mib)
        and report.get("status") in ("target_reached", "allocation_limit")
        and clean_execution(evidence)
        and clean_readback(report)
        and report.get("verified_bytes") == report.get("uploaded_bytes")
    )


def capacity_allocation_complete(
    report: dict, evidence: dict, storage: str, dtype: str, target_mib: int, readback: str
) -> bool:
    """Accept allocation/teardown separately from the contents of retained bytes.

    Exit 1 is allowed only for the native checker's explicit readback-mismatch
    result. Signals, API read failures and incomplete cleanup never establish a bound.
    """
    mismatch = report.get("status") == "readback_mismatch"
    expected_code = 1 if mismatch else 0
    if not (
        readback in ("first", "all", "none")
        and report.get("status") in ("target_reached", "allocation_limit", "readback_mismatch")
        and report.get("readback_mode") == readback
        and clean_execution(evidence, expected_returncode=expected_code)
        and report.get("report_version") == REPORT_VERSION
        and report.get("error") == ""
        and report.get("release_error") == ""
        and report.get("phase") == "complete"
        and report.get("released") is True
    ):
        return False
    if mismatch:
        if not (
            readback != "none"
            and report.get("failure_phase") == "readback"
            and report.get("readback_error") in ("byte_mismatch", "size_mismatch")
            and isinstance(report.get("first_mismatch"), dict)
        ):
            return False
    elif not clean_readback(report):
        return False
    if readback == "none" and any(
        report.get(field) != 0
        for field in (
            "verified_bytes",
            "scanned_bytes",
            "scanned_blocks",
            "mismatched_bytes",
            "mismatched_blocks",
        )
    ):
        return False
    allowed_rejection_phases = ("allocate", "upload") if storage == "mutable" else ("allocate",)
    if report.get("rejection_phase") not in (
        ("",) if report.get("rejected_block_bytes") == 0 else allowed_rejection_phases
    ):
        return False
    return capacity_payload_matches(report, storage, dtype, target_mib)


def capacity_scan_complete(
    report: dict, evidence: dict, storage: str, dtype: str, target_mib: int, output: Path
) -> bool:
    """Validate coverage/counters against the per-block scan, including corrupt blocks."""
    if not capacity_allocation_complete(report, evidence, storage, dtype, target_mib, "all"):
        return False
    try:
        with (output / "readback-blocks.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        uploaded = report["uploaded_bytes"]
        if len(rows) != report["retained_tensors"]:
            return False
        for index, row in enumerate(rows):
            size = min(BLOCK_BYTES, uploaded - index * BLOCK_BYTES)
            if not (
                row["block_index"] == index
                and row["payload_offset_bytes"] == index * BLOCK_BYTES
                and row["expected_bytes"] == size
                and row["actual_bytes"] == size
                and type(row["mismatched_bytes"]) is int
                and 0 <= row["mismatched_bytes"] <= size
                and row["logical_page_size"] == 4096
                and type(row["byte_range_count"]) is int
                and 0 <= row["byte_range_count"] <= row["mismatched_bytes"]
            ):
                return False
            ranges = row["byte_ranges"]
            if len(ranges) != min(256, row["byte_range_count"]):
                return False
            if row["byte_ranges_truncated"] is not (row["byte_range_count"] > 256):
                return False
            for field, limit in (("byte_ranges", size), ("bad_page_ranges", (size + 4095) // 4096)):
                previous = -1
                for begin, end in row[field]:
                    if not (
                        type(begin) is int and type(end) is int and previous < begin < end <= limit
                    ):
                        return False
                    previous = end
            if (
                not row["byte_ranges_truncated"]
                and sum(end - begin for begin, end in ranges) != row["mismatched_bytes"]
            ):
                return False
            if bool(row["bad_page_ranges"]) != bool(row["mismatched_bytes"]):
                return False
        return (
            report["scanned_bytes"] == uploaded
            and report["scanned_blocks"] == len(rows)
            and report["mismatched_bytes"] == sum(row["mismatched_bytes"] for row in rows)
            and report["mismatched_blocks"] == sum(row["mismatched_bytes"] > 0 for row in rows)
            and report["verified_bytes"]
            == sum(row["expected_bytes"] for row in rows if row["mismatched_bytes"] == 0)
            and (report["status"] == "readback_mismatch") is (report["mismatched_blocks"] > 0)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def weights_complete(
    report: dict, evidence: dict, storage: str, expected: dict | None = None
) -> bool:
    weights = report.get("expected_weights")
    weight_bytes = report.get("expected_weight_bytes")
    state_bytes = report.get("expected_state_bytes")
    return (
        storage in STORAGE_MODES
        and clean_execution(evidence)
        and clean_readback(report)
        and report.get("status") == "allocation_pass"
        and report.get("weight_storage") == storage
        and report.get("is_const") is (storage == "constant")
        and positive_integer(weights)
        and report.get("completed_weights") == weights
        and positive_integer(weight_bytes)
        and report.get("uploaded_weight_bytes") == weight_bytes
        and report.get("verified_weight_bytes") == weight_bytes
        and type(state_bytes) is int
        and state_bytes >= 0
        and report.get("uploaded_state_bytes") == state_bytes
        and report.get("verified_state_bytes") == state_bytes
        and report.get("state_allocation_complete") is True
        and report.get("payload_bytes") == weight_bytes + state_bytes
        and (expected is None or all(report.get(key) == value for key, value in expected.items()))
    )


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
