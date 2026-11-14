"""Shared EP/SoC acceptance for complete, fixed-precision Qwen generation."""

import json
from pathlib import Path

import numpy as np

from tools.benchmark.generation_metrics import summarize
from tools.validate.arrays import compare_arrays

TOLERANCES = json.loads(Path(__file__).with_name("tolerances.json").read_text())


def check_kv_reset(current, consumed, previous=None, previous_consumed=0):
    """Reset changes visibility; previously written masked slots need not be zero."""
    capacity = current.shape[2]
    if not 0 <= consumed <= capacity or not 0 <= previous_consumed <= capacity:
        raise ValueError("invalid KV reset extent")
    if previous is None and previous_consumed:
        raise ValueError("previous KV evidence is required for reused storage")
    if previous is not None and previous.shape != current.shape:
        raise ValueError("KV reset snapshots have different shapes")
    high_water = max(consumed, previous_consumed)
    checks = {
        "never_written_bit_zero": not np.any(current[:, :, high_water:].copy().view("u1")).item()
    }
    if previous_consumed > consumed:
        checks["stale_suffix_unchanged"] = (
            current[:, :, consumed:previous_consumed].tobytes()
            == previous[:, :, consumed:previous_consumed].tobytes()
        )
    return checks


def evaluate_generation(
    fixture, device, *, payload_limits=None, previous_kv=None, previous_consumed=0
):
    request = json.loads((fixture / "request.json").read_text())
    reference = json.loads((fixture / "reference.json").read_text())
    generations, checks = [], {}
    previous_kv = dict(previous_kv or {})
    high_water = previous_consumed
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
                visibility = check_kv_reset(
                    full, reference["consumed"], previous_kv.get(name), high_water
                )
                previous_kv[name] = full
                checks[f"request.{index}.{name}.unwritten_bit_zero"] = visibility[
                    "never_written_bit_zero"
                ]
                if "stale_suffix_unchanged" in visibility:
                    checks[f"request.{index}.{name}.stale_suffix_unchanged"] = visibility[
                        "stale_suffix_unchanged"
                    ]
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
        high_water = max(high_water, reference["consumed"])
    payload_limits = payload_limits or {"const": 2**30, "nonconst": 2**30}
    native = json.loads((device / "execution.json").read_text())
    checks["release"] = (
        native["released"]
        and native["const_live_bytes"] == native["nonconst_live_bytes"] == 0
        and all(
            type(native[part + "_peak_bytes"]) is int
            and 0 <= native[part + "_peak_bytes"] <= payload_limits[part]
            for part in ("const", "nonconst")
        )
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "performance": summarize(generations) if all(checks.values()) else None,
        "generations": generations,
    }
