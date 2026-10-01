"""Attribute SDK summary timings without mixing preparation and graph execution."""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

SUMMARY_COLUMNS = (
    "request",
    "phase",
    "role",
    "api",
    "count",
    "total_seconds",
    "min_seconds",
    "max_seconds",
    "completed",
    "exceptions",
    "nested_count",
)
COUNT_COLUMNS = ("count", "completed", "exceptions", "nested_count")
SECONDS_COLUMNS = ("total_seconds", "min_seconds", "max_seconds")
GRAPH_APIS = frozenset(("vsi_nn_RunGraph", "vxProcessGraph"))
IO_APIS = frozenset(("vsi_nn_CopyDataToTensor", "vsi_nn_ConvertTensorToData"))
VERIFY_APIS = frozenset(("vsi_nn_VerifyGraph", "vxVerifyGraph"))
SETUP_APIS = frozenset(("vsi_nn_SetupGraph",))
COMPILATION_APIS = VERIFY_APIS | SETUP_APIS
GENERATION_PHASES = frozenset(("prefill", "decode"))


def _seconds(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _count(value, name):
    if isinstance(value, bool) or not str(value).isdigit():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _validate_row(row):
    if set(row) != set(SUMMARY_COLUMNS):
        raise ValueError("SDK summary row has unexpected or missing columns")
    result = dict(row)
    for key in SUMMARY_COLUMNS[:4]:
        if not isinstance(result[key], str) or any(char in result[key] for char in "\t\r\n"):
            raise ValueError(f"invalid SDK summary label: {key}")
    if not result["api"]:
        raise ValueError("SDK summary API must be nonempty")
    for key in COUNT_COLUMNS:
        result[key] = _count(result[key], key)
    for key in SECONDS_COLUMNS:
        result[key] = _seconds(result[key], key)
    count = result["count"]
    if not count or result["completed"] + result["exceptions"] != count:
        raise ValueError("SDK summary count must equal completed plus exceptions")
    if result["nested_count"] > count:
        raise ValueError("SDK summary nested_count exceeds count")
    lower, upper, total = (result[key] for key in ("min_seconds", "max_seconds", "total_seconds"))
    tolerance = max(1e-9, total * 1e-10)
    if lower > upper or total < count * lower - tolerance or total > count * upper + tolerance:
        raise ValueError("SDK summary duration bounds are inconsistent with count")
    return result


def read_summary(path):
    """Read the native aggregate TSV and reject malformed, repeated or nonfinite rows."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != list(SUMMARY_COLUMNS):
            raise ValueError("unexpected SDK summary header")
        rows = [_validate_row(row) for row in reader]
    keys = [tuple(row[key] for key in SUMMARY_COLUMNS[:4]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("SDK summary contains duplicate aggregate keys")
    return rows


def summarize_request(rows, request_name, generation, prompt_length, expected_tokens=None):
    """Return graph/IO timings and wall times for one resident-model request.

    Only nonnested rows enter totals: their inclusive spans cannot overlap through
    timer nesting. A mixed aggregate containing both nested and outer calls cannot
    be separated from this TSV, so the entire row is omitted and reported as a risk.
    Reset precedes the generation wall timer and is always accounted separately.
    """
    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("prompt_length must be a positive integer")
    if type(generation.get("consumed")) is not int or generation["consumed"] < prompt_length:
        raise ValueError("generation consumed count is smaller than the prompt")
    tokens = generation.get("tokens")
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("generation tokens must be nonnegative integer IDs")
    if expected_tokens is not None and (
        not isinstance(expected_tokens, list)
        or any(type(token) is not int or token < 0 for token in expected_tokens)
    ):
        raise ValueError("expected_tokens must be a list of nonnegative integer IDs")
    wall = {
        key: _seconds(generation.get(key), key)
        for key in ("prefill_seconds", "first_token_seconds", "total_seconds")
    }
    if not wall["prefill_seconds"] <= wall["first_token_seconds"] <= wall["total_seconds"]:
        raise ValueError("generation wall times must satisfy prefill <= first token <= total")
    selected = [_validate_row(row) for row in rows if row["request"] == request_name]
    if not selected:
        raise ValueError(f"no SDK summary rows for request {request_name}")

    totals = {kind: defaultdict(float) for kind in ("graph", "io", "verify", "setup", "other")}
    counts = defaultdict(lambda: defaultdict(int))
    graph_counts = defaultdict(int)
    role_totals = defaultdict(lambda: {"seconds": 0.0, "count": 0})
    risks, excluded = [], []
    for row in selected:
        phase, api = row["phase"], row["api"]
        counts[phase][api] += row["count"]
        if phase in GENERATION_PHASES and api in COMPILATION_APIS:
            risks.append(f"{phase}/{row['role']}/{api}: runtime compilation observed")
        if row["exceptions"]:
            risks.append(f"{phase}/{row['role']}/{api}: {row['exceptions']} exception spans")
        if row["nested_count"]:
            excluded.append(row)
            qualifier = "mixed outer/nested" if row["nested_count"] < row["count"] else "nested"
            risks.append(f"{phase}/{row['role']}/{api}: {qualifier} aggregate excluded")
            continue
        kind = (
            "graph"
            if api in GRAPH_APIS
            else "io"
            if api in IO_APIS
            else "verify"
            if api in VERIFY_APIS
            else "setup"
            if api in SETUP_APIS
            else "other"
        )
        totals[kind][phase] += row["total_seconds"]
        if kind == "graph":
            graph_counts[phase] += row["count"]
            hotspot = role_totals[(phase, row["role"], api)]
            hotspot["seconds"] += row["total_seconds"]
            hotspot["count"] += row["count"]
        if phase not in GENERATION_PHASES and phase != "reset":
            risks.append(f"{phase}/{row['role']}/{api}: outside generation/reset phases")

    graph_prefill = totals["graph"].get("prefill", 0.0)
    graph_decode = totals["graph"].get("decode", 0.0)
    graph_total = graph_prefill + graph_decode
    accounted = sum(
        seconds
        for phases in totals.values()
        for phase, seconds in phases.items()
        if phase in GENERATION_PHASES
    )
    residual = wall["total_seconds"] - accounted
    if residual < -max(1e-6, wall["total_seconds"] * 0.005):
        risks.append(
            "accounted SDK spans exceed generation wall time; inspect overlap or phase labels"
        )
    decode_steps = generation["consumed"] - prompt_length
    missing_graphs = not graph_counts.get("prefill") or (
        decode_steps and not graph_counts.get("decode")
    )
    if missing_graphs:
        risks.append("expected prefill/decode graph spans are missing")
    outer_graph_roles = set(role_totals)
    graph_complete = not missing_graphs and not any(
        row["phase"] in GENERATION_PHASES
        and (
            row["api"] in COMPILATION_APIS
            or (
                row["api"] in GRAPH_APIS
                and (
                    row["nested_count"] < row["count"]
                    or not any(
                        (row["phase"], row["role"], api) in outer_graph_roles for api in GRAPH_APIS
                    )
                )
            )
        )
        for row in excluded
    )
    return {
        "request": request_name,
        "prompt_length": prompt_length,
        "consumed": generation["consumed"],
        "generated_count": len(tokens),
        "tokens": tokens,
        "tokens_match_expected": tokens == expected_tokens if expected_tokens is not None else None,
        "decode_steps": decode_steps,
        "graph_seconds_by_phase": dict(totals["graph"]),
        "io_seconds_by_phase": dict(totals["io"]),
        "runtime_verify_seconds_by_phase": dict(totals["verify"]),
        "runtime_setup_seconds_by_phase": dict(totals["setup"]),
        "runtime_compilation_observed": any(
            row["phase"] in GENERATION_PHASES and row["api"] in COMPILATION_APIS for row in selected
        ),
        "other_sdk_seconds_by_phase": dict(totals["other"]),
        "counts_by_phase": {phase: dict(values) for phase, values in counts.items()},
        "graph_counts_by_phase": dict(graph_counts),
        "reset_seconds": sum(phases.get("reset", 0.0) for phases in totals.values()),
        "reset_timing_scope": "observed outer SDK spans before generation wall timer",
        "graph_total_seconds": graph_total,
        "graph_timing_scope": "outer graph calls excluding separately timed outer Verify/Setup; "
        "nested compilation prevents pure graph attribution",
        "graph_decode_ms_per_token": 1000 * graph_decode / decode_steps
        if decode_steps and graph_complete
        else None,
        "graph_decode_tokens_per_second": decode_steps / graph_decode
        if decode_steps and graph_decode and graph_complete
        else None,
        "graph_prefill_ms_per_input": 1000 * graph_prefill / prompt_length
        if graph_complete
        else None,
        "wall_prefill_seconds": wall["prefill_seconds"],
        "wall_first_token_seconds": wall["first_token_seconds"],
        "wall_total_seconds": wall["total_seconds"],
        "wall_decode_seconds": wall["total_seconds"] - wall["first_token_seconds"],
        "accounted_generation_sdk_seconds": accounted,
        "unattributed_generation_seconds": residual,
        "unattributed_scope": "unmeasured CPU work and SDK boundaries; not device stall time",
        "graph_attribution_complete": graph_complete,
        "role_hotspots": sorted(
            [
                {"phase": phase, "role": role, "api": api, **values}
                for (phase, role, api), values in role_totals.items()
                if phase in GENERATION_PHASES
            ],
            key=lambda item: item["seconds"],
            reverse=True,
        ),
        "excluded_nested_rows": excluded,
        "attribution_risks": risks,
    }


def summarize_runs(samples):
    """Preserve individual runs and describe their spread without tail estimates."""
    if not samples:
        raise ValueError("at least one performance sample is required")
    metric_names = (
        "graph_total_seconds",
        "graph_decode_ms_per_token",
        "graph_decode_tokens_per_second",
        "graph_prefill_ms_per_input",
        "reset_seconds",
        "wall_prefill_seconds",
        "wall_first_token_seconds",
        "wall_total_seconds",
        "wall_decode_seconds",
        "accounted_generation_sdk_seconds",
        "unattributed_generation_seconds",
    )
    metrics = {}
    for name in metric_names:
        values = [sample[name] for sample in samples if sample.get(name) is not None]
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError(f"performance metric {name} must be finite")
        metrics[name] = (
            {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
            if values
            else None
        )
    return {"sample_count": len(samples), "metrics": metrics, "samples": list(samples)}
