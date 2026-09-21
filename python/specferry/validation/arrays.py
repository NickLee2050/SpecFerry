"""Reference file IO and comparison shared by component diagnostics."""

from pathlib import Path

import numpy as np
import torch

from .capabilities import compare_arrays


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
