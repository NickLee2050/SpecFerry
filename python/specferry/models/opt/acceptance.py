"""Validate repeated resident generation and summarize host-observed timings."""

import json
import math
import statistics
from pathlib import Path

from specferry.validation.device import clean_execution
from specferry.validation.timing import summarize_sdk_timing


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
    cache = result.get("cache_write_seconds", 0)
    verify = result.get("cache_revalidation_seconds", 0)
    return (
        all(elapsed(value) for value in (prefill, first, total))
        and all(type(value) in (int, float) and math.isfinite(value) for value in (cache, verify))
        and 0 <= verify <= cache <= total
        and isinstance(intervals, list)
        and len(intervals) == len(tokens) - 1
        and all(elapsed(value) for value in intervals)
        and prefill <= first <= total
        and first + sum(intervals) <= total + 1e-9
    )


def evaluate_benchmark(
    directory: Path, request, reference, evidence, warmups, repeats, maximum, sequence=None
):
    if sequence is not None and len(sequence) != warmups + repeats:
        raise ValueError("request sequence length differs from the native run count")
    execution_path = directory / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.is_file() else {}
    runs = []
    checks = {}
    steps = predictions = 0
    measured_prompt_tokens = 0
    for kind, count in (("warmup", warmups), ("measured", repeats)):
        for index in range(count):
            name = f"{kind}.{index}"
            path = directory / f"{name}.json"
            result = json.loads(path.read_text()) if path.is_file() else {}
            current_request, current_reference = (
                sequence[len(runs)] if sequence is not None else (request, reference)
            )
            checks[name] = valid_generation(
                result, current_request, current_reference["tokens"], maximum
            )
            steps += len(current_request["tokens"]) + len(current_reference["tokens"]) - 1
            predictions += len(current_reference["tokens"])
            if kind == "measured":
                measured_prompt_tokens += len(current_request["tokens"])
            runs.append(
                {
                    "name": name,
                    "warmup": kind == "warmup",
                    "generation": result,
                    "prompt_tokens": len(current_request["tokens"]),
                }
            )

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
        and options.get("SPECFERRY_SDK_TIMING") in (None, "off", "summary")
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
    sdk_timings = None
    if options.get("SPECFERRY_SDK_TIMING") == "summary":
        try:
            sdk_timings = summarize_sdk_timing(directory / "sdk-timing.tsv")
            measured_steps = sum(
                len((sequence[i][0] if sequence is not None else request)["tokens"])
                + len((sequence[i][1] if sequence is not None else reference)["tokens"])
                - 1
                for i in range(warmups, warmups + repeats)
            )
            measured_predictions = sum(
                len((sequence[i][1] if sequence is not None else reference)["tokens"])
                for i in range(warmups, warmups + repeats)
            )
            counts = sdk_timings["measured_api_calls"]
            checks["sdk_timing_counts"] = (
                counts.get("vsi_nn_RunGraph") == measured_steps * 50 + measured_predictions * 2
                and counts.get("vxProcessGraph") == measured_steps * 24
                and counts.get("vsi_nn_CopyDataToTensor") == measured_steps * 26
                and counts.get("vsi_nn_ConvertTensorToData") == measured_predictions
                and not any(row["exceptions"] for row in sdk_timings["rows"])
            )
        except (OSError, ValueError, KeyError):
            checks["sdk_timing_counts"] = False
    measured = [run["generation"] for run in runs if not run["warmup"]]
    if sdk_timings and all(checks.values()):
        checks["sdk_timing_scope"] = sdk_timings["measured_sdk_seconds"] <= (
            sum(run["total_seconds"] for run in measured) + 1e-6
        )
    passed = all(checks.values())
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
            "prefill_tokens_per_second": measured_prompt_tokens
            / sum(run["prefill_seconds"] for run in measured),
            "decode_tokens_per_second": len(intervals) / sum(intervals) if intervals else None,
            "output_tokens_per_request_second": sum(len(run["tokens"]) for run in measured)
            / sum(run["total_seconds"] for run in measured),
            "cache_write_seconds": sum(run.get("cache_write_seconds", 0) for run in measured),
            "cache_revalidation_seconds": sum(
                run.get("cache_revalidation_seconds", 0) for run in measured
            ),
        }
    return {
        "status": "numerical_pass" if passed else "failed",
        "checks": checks,
        "runs": runs,
        "metrics": metrics,
        "sdk_timings": sdk_timings,
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
        {stage["name"] for stage in stages} >= {"selection", "teacher"}
        and all(stage["passed"] is True for stage in stages)
        and expected_benchmarks >= 1
        and len(benchmarks) == expected_benchmarks
        and all(run["status"] == "numerical_pass" for run in benchmarks)
    )
    remaining = [
        "per-kernel execution backend and completion",
        "physical weight/KV residency and SDK-internal transfers",
        "device memory peak, layouts/workspace and release accounting",
        "long-running resource stability and context compression (deferred)",
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
        "application_timing_measured": numerical and all(run.get("metrics") for run in benchmarks),
        "kv_and_reset_pass": numerical and {s["name"] for s in stages} >= {"kv", "reset"},
        "remaining_gates": remaining,
    }


def write_results(path, report):
    """Keep throughput, measurement scope and readable continuations next to raw JSON."""
    lines = [
        "# Generation observations",
        "",
        f"Status: {report['status']}",
        "",
        "Host wall time; sequential prefill; checkpoint-default greedy generation.",
        "Initialization, tokenization and warmups are excluded. Hardware backend and physical residency remain unproven.",
        "",
        "| Process | Prompt tokens | Decode tokens/s | Prefill tokens/s | TTFT median (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in report["benchmarks"]:
        metrics = run.get("metrics")
        if not metrics:
            continue
        rate = metrics["decode_tokens_per_second"]
        rate_text = f"{rate:.4f}" if rate is not None else "N/A (one output token)"
        count = run["runs"][0]["prompt_tokens"]
        lines.append(
            f"| {run.get('cycle', 0)}/{run.get('request', 0)} | {count} | {rate_text} | "
            f"{metrics['prefill_tokens_per_second']:.4f} | "
            f"{metrics['first_token_seconds']['median']:.3f} |"
        )
    for run in report["benchmarks"]:
        lines.extend(
            [
                "",
                f"## Process {run.get('cycle', 0)}, prompt {run.get('request', 0)}",
                "",
                "Prompt ending (full text and IDs are in request.json):",
                "",
                "```text",
                run.get("prompt", "")[-400:],
                "```",
            ]
        )
        for sample in run["runs"]:
            if not sample["warmup"]:
                generated = sample["generation"]
                lines.extend(
                    [
                        "",
                        f"{sample['name']}: {generated.get('stop_reason', 'incomplete')}",
                        "",
                        "```text",
                        sample.get("text", ""),
                        "```",
                    ]
                )
        observations = run.get("sdk_timings")
        if observations:
            lines.extend(
                [
                    "",
                    "Instrumented SDK call wall time (measured requests only):",
                    "",
                    "| Phase | SDK seconds |",
                    "|---|---:|",
                ]
            )
            for phase, seconds in observations["measured_phase_seconds"].items():
                lines.append(f"| {phase} | {seconds:.6f} |")
            lines.extend(
                [
                    "",
                    "| Public API | Calls | Total seconds | Maximum seconds |",
                    "|---|---:|---:|---:|",
                ]
            )
            for api, seconds in sorted(
                observations["measured_api_seconds"].items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                calls = observations["measured_api_calls"][api]
                maximum = observations["measured_api_max_seconds"][api]
                lines.append(f"| {api} | {calls} | {seconds:.6f} | {maximum:.6f} |")
            lines.extend(
                [
                    "",
                    "Largest component totals; each includes its separately recorded SDK calls:",
                    "",
                    "| Component | SDK seconds |",
                    "|---|---:|",
                ]
            )
            largest = sorted(
                observations["measured_component_seconds"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:8]
            for component, seconds in largest:
                lines.append(f"| {component} | {seconds:.6f} |")
            lines.extend(
                [
                    "",
                    "Per-layer/API counts and maxima: sdk-timing.tsv in this process directory. "
                    "These are host API durations, not kernel times; remaining host work and "
                    "unobserved calls are outside these totals.",
                ]
            )
        if run.get("status") != "numerical_pass":
            lines.extend(
                [
                    "",
                    "FAILED correctness/lifecycle validation; excluded from accepted timing metrics.",
                ]
            )
    lines.extend(
        [
            "",
            "Known limitations: host allocation growth; SDK accounting semantics; large-allocation corruption.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
