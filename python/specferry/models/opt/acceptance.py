"""Validate repeated resident generation and summarize host-observed timings."""

import json
import math
import statistics
from pathlib import Path

from specferry.validation.device import clean_execution


def distribution(values):
    """Linear-interpolated percentiles; always retain the small sample count."""
    ordered = sorted(values)
    if not ordered:
        return None

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
        "mean": statistics.mean(ordered),
    }


def elapsed(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def valid_generation(result, request, expected, maximum):
    tokens = result.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int for token in tokens)
        or tokens != expected
    ):
        return False
    prompt_length = len(request["tokens"])
    consumed = prompt_length + len(tokens) - 1
    eos = request["selection"]["resolved_config"]["eos_token_id"]
    stop = (
        "eos" if tokens[-1] == eos else "max_new_tokens" if len(tokens) == maximum else "capacity"
    )
    if (
        len(tokens) > maximum
        or eos in tokens[:-1]
        or consumed > request["capacity"]
        or (stop == "capacity" and consumed != request["capacity"])
        or type(result.get("consumed")) is not int
        or result["consumed"] != consumed
        or result.get("stop_reason") != stop
    ):
        return False
    prefill = result.get("prefill_seconds")
    first = result.get("first_token_seconds")
    total = result.get("total_seconds")
    intervals = result.get("token_seconds")
    return (
        all(elapsed(value) for value in (prefill, first, total))
        and isinstance(intervals, list)
        and len(intervals) == len(tokens) - 1
        and all(elapsed(value) for value in intervals)
        and prefill <= first <= total
        and first + sum(intervals) <= total + 1e-9
    )


def evaluate_benchmark(directory: Path, request, reference, evidence, warmups, repeats, maximum):
    execution_path = directory / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.is_file() else {}
    runs = []
    checks = {}
    for kind, count in (("warmup", warmups), ("measured", repeats)):
        for index in range(count):
            name = f"{kind}.{index}"
            path = directory / f"{name}.json"
            result = json.loads(path.read_text()) if path.is_file() else {}
            checks[name] = valid_generation(result, request, reference["tokens"], maximum)
            runs.append({"name": name, "warmup": kind == "warmup", "generation": result})

    steps = (len(request["tokens"]) + len(reference["tokens"]) - 1) * (warmups + repeats)
    predictions = len(reference["tokens"]) * (warmups + repeats)
    checks["lifecycle_and_transfers"] = (
        clean_execution(evidence)
        and execution.get("status") == "executed"
        and execution.get("phase") == "complete"
        and execution.get("released") is True
        and execution.get("layers") == 24
        and execution.get("steps") == steps
        and execution.get("cache_writes") == steps * 24
        and execution.get("uploads") == steps * 26
        and execution.get("upload_bytes") == steps * 104
        and execution.get("reads") == predictions
        and execution.get("read_bytes") == predictions * 4
        and elapsed(execution.get("initialize_seconds"))
    )
    options = evidence.get("runtime_options", {})
    checks["untraced_measurement"] = (
        evidence.get("driver_traced") is False
        and options.get("VIV_VX_ENABLE_PRINT_TARGET") == "0"
        and options.get("VIV_VX_PROFILE") in (None, "", "0")
        and options.get("VIV_MEMORY_PROFILE") in (None, "", "0")
    )
    resource_path = directory / "host-resources.jsonl"
    resources = (
        [json.loads(line) for line in resource_path.read_text().splitlines()]
        if resource_path.is_file()
        else []
    )
    phases = ["before_initialize", "after_initialize"]
    phases.extend(run["name"] for run in runs)
    phases.append("after_release")
    checks["host_observations"] = [entry.get("phase") for entry in resources] == phases and all(
        type(entry.get("rss_bytes")) is int and entry["rss_bytes"] > 0 for entry in resources
    )
    passed = all(checks.values())
    measured = [run["generation"] for run in runs if not run["warmup"]]
    metrics = None
    rss_growth = None
    if checks["host_observations"]:
        baseline = resources[1 + warmups]["rss_bytes"]
        rss_growth = resources[-2]["rss_bytes"] - baseline
    if passed:
        intervals = [value for run in measured for value in run["token_seconds"]]
        metrics = {
            "prefill_seconds": distribution([run["prefill_seconds"] for run in measured]),
            "first_token_seconds": distribution([run["first_token_seconds"] for run in measured]),
            "decode_token_seconds": distribution(intervals),
            "request_seconds": distribution([run["total_seconds"] for run in measured]),
            "prefill_tokens_per_second": len(request["tokens"])
            * repeats
            / sum(run["prefill_seconds"] for run in measured),
            "decode_tokens_per_second": len(intervals) / sum(intervals) if intervals else None,
        }
    return {
        "status": "numerical_pass" if passed else "failed",
        "checks": checks,
        "runs": runs,
        "metrics": metrics,
        "initialize_seconds": execution.get("initialize_seconds"),
        "host_peak_rss_kib": execution.get("host_peak_rss_kib"),
        "host_resources": resources,
        "measured_host_rss_growth_bytes": rss_growth,
        "timing_scope": "host wall time, reset and initialization excluded; warmups excluded from metrics",
        "weight_bytes": request["weight_bytes"],
        "kv_bytes": request["kv_bytes"],
        "known_weight_kv_bytes": request["weight_bytes"] + request["kv_bytes"],
        "physical_device_peak_bytes": None,
        "sdk_layout_workspace_bytes": None,
        "hardware_execution_proven": False,
        "device_residency_proven": False,
    }


def summarize(stages, benchmarks, expected_benchmarks):
    numerical = (
        len(stages) == 2
        and {stage["name"] for stage in stages} == {"selection", "teacher"}
        and all(stage["passed"] is True for stage in stages)
        and expected_benchmarks >= 2
        and len(benchmarks) == expected_benchmarks
        and all(run["status"] == "numerical_pass" for run in benchmarks)
    )
    remaining = [
        "per-kernel execution backend and completion",
        "physical weight/KV residency and SDK-internal transfers",
        "device memory peak, layouts/workspace and release accounting",
        "representative long-context acceptance and formal performance measurements",
    ]
    if any((run.get("measured_host_rss_growth_bytes") or 0) > 0 for run in benchmarks):
        remaining.append("explain host allocation growth across repeated requests")
    return {
        "status": "hardware_pending" if numerical else "failed",
        "numerical_pass": numerical,
        "stages": stages,
        "completed_benchmarks": len(benchmarks),
        "expected_benchmarks": expected_benchmarks,
        "benchmarks": benchmarks,
        "hardware_execution_proven": False,
        "device_residency_proven": False,
        "formal_performance_accepted": False,
        "remaining_gates": remaining,
    }
