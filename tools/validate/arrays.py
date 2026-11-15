"""Reference file IO and comparison shared by component diagnostics."""

from pathlib import Path

import numpy as np
import torch


def save_tensor(path: Path, value: torch.Tensor):
    array = value.detach().contiguous().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"nonfinite reference: {path.name}")
    path.write_bytes(array.tobytes())
    return {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.nbytes}


def compare_file(actual: Path, expected: Path, spec: dict, tolerance: dict):
    for path in (actual, expected):
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            return {"passed": False, "reason": f"missing or wrong-size file: {path.name}"}
    return compare_arrays(
        np.fromfile(actual, dtype=spec["dtype"]),
        np.fromfile(expected, dtype=spec["dtype"]),
        **tolerance,
    )


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
