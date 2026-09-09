"""Compare probe outputs and preserve separate numerical and hardware acceptance."""

import json
from pathlib import Path

import numpy as np

from .device import fingerprint, run_device, write_json
from .fixtures import DTYPES

TOLERANCE_FILE = Path(__file__).parents[1] / "reference/tolerances.json"


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


def compare_outputs(metadata: dict, fixture: Path, output: Path) -> dict:
    policy = json.loads(TOLERANCE_FILE.read_text())
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
        for step in range(metadata["steps"]):
            key = f"{name}.{step}"
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
    case, directory: Path, binary: Path, sdk_lib: Path, timeout: int, prepare_only=False
) -> dict:
    fixture_dir = directory / "fixture"
    metadata = case.build().write(fixture_dir)
    metadata["scale"] = case.scale
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
    }
    if prepare_only:
        return result
    execution_dir = directory / "execution"
    evidence = run_device(
        binary,
        [str(fixture_dir / "graph.txt"), str(execution_dir)],
        execution_dir,
        sdk_lib,
        timeout,
    )
    report_path = execution_dir / "execution.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    comparisons = compare_outputs(metadata, fixture_dir, execution_dir)
    numeric_pass = (
        evidence["returncode"] == 0
        and not evidence["device_recovery_required"]
        and bool(comparisons)
        and all(item["passed"] for item in comparisons.values())
    )
    blockers = ["individual_node_execution_unverified"]
    if not numeric_pass:
        blockers.append("execution_or_numeric_failure")
    if evidence["device_recovery_required"]:
        blockers.append("device_recovery_required")
    if report.get("rnn_connection") and not report.get("rnn_tensor_swappable"):
        blockers.append("rnn_uses_host_state_buffer")
    if report.get("argmax_execute_on_sw"):
        blockers.append("argmax_software_path_selected")
    result.update(
        {
            "status": "numerical_pass" if numeric_pass else "failed",
            "execution": report,
            "comparisons": comparisons,
            "blockers": blockers,
            "driver_activity_observed": evidence["driver_activity_observed"],
            "tolerances_sha256": fingerprint(TOLERANCE_FILE),
        }
    )
    write_json(directory / "result.json", result)
    return result
