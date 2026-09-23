"""Summarize public SDK call wall times; never infer kernel or execution-engine time."""

import csv
import math
from collections import defaultdict
from pathlib import Path


def summarize_sdk_timing(path: Path) -> dict:
    rows = []
    with path.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for name in ("calls", "exceptions"):
                row[name] = int(row[name])
            for name in ("total_seconds", "max_seconds"):
                row[name] = float(row[name])
            if (
                row["calls"] <= 0
                or not 0 <= row["exceptions"] <= row["calls"]
                or not all(math.isfinite(row[k]) for k in ("total_seconds", "max_seconds"))
                or not 0 <= row["max_seconds"] <= row["total_seconds"]
            ):
                raise ValueError("invalid SDK timing observation")
            rows.append(row)
    if not rows:
        raise ValueError("SDK timing file has no completed calls")
    phases, components, calls = defaultdict(float), defaultdict(float), defaultdict(int)
    api_seconds, api_maximum = defaultdict(float), defaultdict(float)
    for row in rows:
        if row["request"].startswith("measured."):
            phases[row["phase"]] += row["total_seconds"]
            components[row["component"]] += row["total_seconds"]
            calls[row["api"]] += row["calls"]
            api_seconds[row["api"]] += row["total_seconds"]
            api_maximum[row["api"]] = max(api_maximum[row["api"]], row["max_seconds"])
    return {
        "scope": "non-overlapping instrumented SDK calls on the submitting host thread",
        "includes_kernel_proof": False,
        "rows": rows,
        "measured_phase_seconds": dict(phases),
        "measured_component_seconds": dict(components),
        "measured_api_calls": dict(calls),
        "measured_api_seconds": dict(api_seconds),
        "measured_api_max_seconds": dict(api_maximum),
        "measured_sdk_seconds": sum(phases.values()),
        "note": "Warmups/setup/release excluded from measured totals; unsampled SDK calls and host work remain outside them.",
    }
