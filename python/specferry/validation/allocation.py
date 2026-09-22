"""Shared execution and acceptance contracts for allocation-only diagnostics."""

import os
from argparse import ArgumentParser
from pathlib import Path

from .device import RECOVERY_ROOT, device_lock, run_device, snapshot_binary

MIB = 1024**2
BLOCK_BYTES = 8 * MIB
MAXIMUM_CAPACITY_MIB = 4096
REPORT_VERSION = 2
PATTERN = "splitmix64-finite-v1"
STORAGE_MODES = ("constant", "mutable")
FLOAT_DTYPES = ("F16", "F32")


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


def clean_execution(evidence: dict) -> bool:
    return (
        evidence.get("returncode") == 0
        and evidence.get("timeout") is False
        and evidence.get("process_group_exited") is True
        and evidence.get("remaining_processes") == []
        and evidence.get("device_recovery_required") is False
    )


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


def capacity_complete(
    report: dict, evidence: dict, storage: str, dtype: str, target_mib: int
) -> bool:
    uploaded = report.get("uploaded_bytes")
    rejected = report.get("rejected_block_bytes")
    target = target_mib * MIB
    if not (
        storage in STORAGE_MODES
        and dtype in FLOAT_DTYPES
        and 1 <= target_mib <= MAXIMUM_CAPACITY_MIB
        and clean_execution(evidence)
        and clean_readback(report)
        and report.get("storage") == storage
        and report.get("dtype") == dtype
        and report.get("pattern") == PATTERN
        and report.get("target_bytes") == target
        and report.get("block_bytes") == BLOCK_BYTES
        and positive_integer(uploaded)
        and uploaded <= target
        and report.get("verified_bytes") == uploaded
        and report.get("retained_tensors") == (uploaded + BLOCK_BYTES - 1) // BLOCK_BYTES
        and type(rejected) is int
    ):
        return False
    if report.get("status") == "target_reached":
        return uploaded == target and rejected == 0 and report.get("allocation_error") == ""
    return (
        report.get("status") == "allocation_limit"
        and uploaded < target
        and uploaded % BLOCK_BYTES == 0
        and rejected == min(BLOCK_BYTES, target - uploaded)
        and isinstance(report.get("allocation_error"), str)
        and bool(report["allocation_error"])
    )


def weights_complete(report: dict, evidence: dict, storage: str) -> bool:
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
        and positive_integer(state_bytes)
        and report.get("uploaded_state_bytes") == state_bytes
        and report.get("verified_state_bytes") == state_bytes
        and report.get("state_allocation_complete") is True
        and report.get("payload_bytes") == weight_bytes + state_bytes
    )
