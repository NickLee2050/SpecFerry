"""Compare probe outputs and preserve separate numerical and hardware acceptance."""

import json
from pathlib import Path

import numpy as np

from .device import fingerprint, run_device, write_json
from .fixtures import DTYPES


def compare_arrays(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape mismatch"}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "reason": "NaN/Inf"}
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    limits = atol + rtol * np.abs(expected.astype(np.float64))
    failures = np.flatnonzero(difference > limits)
    return {
        "passed": failures.size == 0,
        "max_abs_error": float(difference.max(initial=0)),
        "failed_elements": int(failures.size),
        "first_failed_index": int(failures[0]) if failures.size else None,
        "atol": atol,
        "rtol": rtol,
    }


def compare_outputs(
    metadata: dict, fixture: Path, output: Path, policy: dict | None = None
) -> dict:
    policy = metadata["tolerances"] if policy is None else policy
    results = {}
    for name in metadata["outputs"]:
        tensor = metadata["tensors"][name]
        dtype = DTYPES[tensor["dtype"]]
        tolerance = (
            {"atol": 0, "rtol": 0}
            if tensor["dtype"] in ("I32", "BOOL")
            else policy[
                "fixed_input_fp16_output" if tensor["dtype"] == "F16" else "fixed_input_fp32_state"
            ]
        )
        steps = (
            [metadata["steps"] - 1]
            if metadata.get("readback_mode") == "final"
            else range(metadata["steps"])
        )
        cycles = metadata.get("cycles", 1)
        for cycle in range(cycles):
            prefix = f"cycle.{cycle}." if cycles > 1 else ""
            for step in steps:
                key = f"{prefix}{name}.{step}"
                actual_path = output / f"{key}.bin"
                expected_path = fixture / f"{name}.expected.{step}.bin"
                expected_bytes = int(np.prod(tensor["shape"])) * dtype.itemsize
                if not actual_path.is_file() or actual_path.stat().st_size != expected_bytes:
                    results[key] = {"passed": False, "reason": "missing or wrong-size output"}
                    continue
                expected = np.fromfile(expected_path, dtype=dtype).reshape(tensor["shape"])
                actual = np.fromfile(actual_path, dtype=dtype).reshape(tensor["shape"])
                results[key] = compare_arrays(actual, expected, **tolerance)
    return results


def check_case(
    case,
    directory: Path,
    binary: Path,
    sdk_lib: Path,
    timeout: int,
    prepare_only=False,
    readback="each-step",
    cycles=1,
    max_payload_bytes=768 * 1024**2,
    shader_header=None,
) -> dict:
    fixture_dir = directory / "fixture"
    metadata = case.build().write(fixture_dir)
    if case.tolerances is None:
        raise ValueError("operator case requires an explicit tolerance policy")
    metadata["tolerances"] = case.tolerances
    metadata["scale"] = case.scale
    metadata["readback_mode"] = readback
    metadata["cycles"] = cycles
    metadata["fixture_sha256"] = {
        path.name: fingerprint(path) for path in sorted(fixture_dir.iterdir())
    }
    write_json(fixture_dir / "case.json", metadata)
    result = {
        "name": case.name,
        "family": case.family,
        "scale": case.scale,
        "header_and_link_available": binary.is_file(),
        "status": "prepared",
        "hardware_execution_proven": False,
        "fixture": str(fixture_dir),
        "readback_mode": readback,
        "cycles": cycles,
        "tensor_payload_bytes": metadata["tensor_payload_bytes"],
        "constant_payload_bytes": metadata["constant_payload_bytes"],
        "acceptance": {
            "numerical": "not_run",
            "lifecycle": "not_run",
            "hardware": "unverified",
            "device_residency": "unverified",
        },
    }
    if prepare_only:
        return result
    if metadata["tensor_payload_bytes"] > max_payload_bytes:
        result.update(status="blocked", blockers=["tensor_payload_budget_exceeded"])
        write_json(directory / "result.json", result)
        return result
    execution_dir = directory / "execution"
    evidence = run_device(
        binary,
        [
            str(fixture_dir / "graph.txt"),
            str(execution_dir),
            "--readback",
            readback,
            "--cycles",
            str(cycles),
        ],
        execution_dir,
        sdk_lib,
        timeout,
        shader_header=shader_header,
    )
    report_path = execution_dir / "execution.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    comparisons = compare_outputs(metadata, fixture_dir, execution_dir)
    lifecycle_pass = (
        evidence["returncode"] == 0
        and not evidence["device_recovery_required"]
        and report.get("status") == "executed"
        and report.get("completed_steps") == metadata["steps"]
        and report.get("completed_cycles") == cycles
        and report.get("readback_mode") == readback
    )
    outputs_pass = bool(comparisons) and all(item["passed"] for item in comparisons.values())
    numeric_pass = lifecycle_pass and outputs_pass
    blockers = ["individual_node_execution_unverified"]
    sdk_log = (
        (execution_dir / "sdk.log").read_text(errors="replace")
        if (execution_dir / "sdk.log").is_file()
        else ""
    )
    compile_errors = [
        line
        for line in sdk_log.splitlines()
        if any(
            marker in line
            for marker in (
                "Cannot find the header",
                "Failed to compile",
                "Build program fail",
                "Register client kernel",
            )
        )
    ]
    if compile_errors:
        blockers.append("sdk_kernel_compilation_failed")
    if not numeric_pass:
        blockers.append("execution_or_numeric_failure")
    if evidence["device_recovery_required"]:
        blockers.append("device_recovery_required")
    if report.get("argmax_execute_on_sw"):
        blockers.append("argmax_software_path_selected")
    acceptance = {
        "numerical": "passed" if outputs_pass else "failed",
        "lifecycle": "passed" if lifecycle_pass else "failed",
        "hardware": "software_path_observed"
        if report.get("argmax_execute_on_sw")
        else "unverified",
        "device_residency": "unverified",
    }
    result.update(
        {
            "status": "numerical_pass" if numeric_pass else "failed",
            "execution": report,
            "acceptance": acceptance,
            "execution_directory": str(execution_dir),
            "kernel_compilation_errors": compile_errors,
            "comparisons": comparisons,
            "blockers": blockers,
            "driver_activity_observed": evidence["driver_activity_observed"],
            "tolerances": case.tolerances,
        }
    )
    write_json(directory / "result.json", result)
    return result
