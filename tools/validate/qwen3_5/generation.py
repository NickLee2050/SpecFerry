"""Shared EP/SoC acceptance for complete, fixed-precision Qwen generation."""

import json
from pathlib import Path

import numpy as np

from tools.benchmark.generation_metrics import summarize
from tools.validate.arrays import compare_arrays

TOLERANCES = json.loads(Path(__file__).with_name("tolerances.json").read_text())


def evaluate_generation(fixture, device):
    request = json.loads((fixture / "request.json").read_text())
    reference = json.loads((fixture / "reference.json").read_text())
    generations, checks = [], {}
    for index in range(request["repeats"]):
        run = json.loads((device / f"measured.{index}.json").read_text())
        generations.append(run)
        checks[f"request.{index}"] = (
            run["tokens"] == reference["tokens"]
            and run["consumed"] == reference["consumed"]
            and run["stop_reason"] == reference["stop_reason"]
        )
        for name, contract in reference["snapshots"].items():
            expected = np.fromfile(
                fixture / f"expected.{name}.bin", dtype=contract["dtype"]
            ).reshape(contract["shape"])
            actual = np.fromfile(device / f"measured.{index}.{name}.bin", dtype=contract["dtype"])
            if name in ("keys", "values"):
                full = actual.reshape(1, 2, request["capacity"], 256)
                checks[f"request.{index}.{name}.unwritten_bit_zero"] = not np.any(
                    full[:, :, reference["consumed"] :].copy().view("u1")
                ).item()
                actual = full[:, :, : reference["consumed"]]
            else:
                actual = actual.reshape(expected.shape)
            tolerance = TOLERANCES[
                "fp32_state_from_fp16_pipeline" if name == "recurrent" else "fp16_pipeline"
            ]
            checks[f"request.{index}.{name}"] = compare_arrays(actual, expected, **tolerance)[
                "passed"
            ]
            if index:
                checks[f"request.{index}.{name}.reset"] = (
                    device / f"measured.{index}.{name}.bin"
                ).read_bytes() == (device / f"measured.0.{name}.bin").read_bytes()
    native = json.loads((device / "execution.json").read_text())
    checks["release"] = (
        native["released"]
        and native["const_live_bytes"] == native["nonconst_live_bytes"] == 0
        and all(
            type(native[part + "_peak_bytes"]) is int and 0 <= native[part + "_peak_bytes"] <= 2**30
            for part in ("const", "nonconst")
        )
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "performance": summarize(generations) if all(checks.values()) else None,
        "generations": generations,
    }
