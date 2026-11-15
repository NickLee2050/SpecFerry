"""Pinned compiled-model acceptance with preparation outside generation timings."""

import json
import math
from pathlib import Path

import numpy as np

from tools.deploy.execute.np101.device import (
    RECOVERY_ROOT,
    write_json,
)
from tools.deploy.execute.np101.recovery_controls import healthy
from tools.deploy.execute.np101.runner import run_guarded
from tools.deploy.export.checkpoint import sha256
from tools.deploy.export.opt.config import TOLERANCES
from tools.validate.arrays import compare_arrays

ROOT = RECOVERY_ROOT.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_case(plan_path, name):
    """Validate artifact/reference identities without kernel or device access."""
    plan = read_json(plan_path)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported compiled validation plan")
    case = dict(plan["cases"][name])
    if case["kind"] not in {"fullgraph", "partitioned"}:
        raise ValueError("unsupported compiled validation output contract")
    if case["stop_reason"] not in {"max_new_tokens", "capacity", "eos"}:
        raise ValueError("compiled validation requires an explicit expected stop reason")
    for segment in ("const", "nonconst"):
        payload = case["payload_bound_bytes"][segment]
        if type(payload) is not int or payload < 0 or payload > 2**30:
            raise ValueError("compiled validation exceeds the application payload limit")
    if not 1 <= case["repeats"] <= 20:
        raise ValueError("compiled validation requires bounded measured repeats")
    if case["kind"] == "fullgraph" and case["repeats"] != 1:
        raise ValueError("the accepted fullgraph runner has one measured request")
    files = {}
    for group in case["file_sets"]:
        files.update(plan["file_sets"][group])
    for value, digest in files.items():
        if sha256(resolve_path(value)) != digest:
            raise ValueError(f"compiled artifact/reference identity changed: {value}")
    binary = resolve_path(case["binary"]).resolve(strict=True)
    if case.get("binary_sha256") and sha256(binary) != case["binary_sha256"]:
        raise ValueError("compiled validation binary identity changed")
    request = resolve_path(case["request"])
    required = [request / "reference.json", request / "expected.0.logits.bin"]
    if "cache" in case:
        cache = case["cache"]
        if cache.get("dtype") not in {"F16", "F32"}:
            raise ValueError("compiled KV reference dtype must be F16 or F32")
        shape = cache.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise ValueError("compiled KV reference requires a positive rank3 shape")
        expected_bytes = math.prod(shape) * (2 if cache["dtype"] == "F16" else 4)
        if any(
            resolve_path(cache[part]).stat().st_size != expected_bytes
            for part in ("keys", "values")
        ):
            raise ValueError("compiled KV reference size differs from its dtype and shape")
        required.extend(resolve_path(case["cache"][part]) for part in ("keys", "values"))
    if any(path not in {resolve_path(value) for value in files} for path in required):
        raise ValueError("independent numerical references require pinned file identities")
    reference = read_json(request / "reference.json")["requests"]
    if len(reference) != 1 or not reference[0]["tokens"]:
        raise ValueError("compiled validation requires one independent nonempty token oracle")
    case.update(binary=binary, request=request, reference=reference[0], files=files)
    return case, plan["sdk"]


def summarize(generations):
    measured = generations[1:]
    wall = sum(run["total_seconds"] for run in measured)
    decode = sum(sum(run["token_seconds"]) for run in measured)
    count = sum(len(run["tokens"]) for run in measured)
    steps = sum(len(run["token_seconds"]) for run in measured)
    if (
        not math.isfinite(wall)
        or not math.isfinite(decode)
        or wall <= 0
        or decode <= 0
        or not steps
    ):
        raise ValueError("compiled benchmark requires measured generation and decode intervals")
    return {
        "measured_requests": len(measured),
        "generated_tokens": count,
        "generation_seconds": wall / len(measured),
        "output_tokens_per_second": count / wall,
        "first_token_seconds": sum(run["first_token_seconds"] for run in measured) / len(measured),
        "decode_ms": 1000 * decode / steps,
        "decode_tokens_per_second": steps / decode,
    }


def evaluate(case, device):
    fullgraph = case["kind"] == "fullgraph"
    names = (
        ["warmup", "measured"]
        if fullgraph
        else [
            "warmup.0.0",
            *[f"measured.0.{index}" for index in range(case["repeats"])],
        ]
    )
    generations = [read_json(device / (name + ".json")) for name in names]
    checks = {
        "tokens": all(
            run["tokens"] == case["reference"]["tokens"]
            and run["consumed"] == case["reference"]["consumed"]
            and run["stop_reason"] == case["stop_reason"]
            for run in generations
        )
    }
    expected = np.fromfile(case["request"] / "expected.0.logits.bin", dtype="<f2")
    logits = []
    for name in ["logits"] if fullgraph else [name + ".logits" for name in names[1:]]:
        logits.append(
            compare_arrays(
                np.fromfile(device / (name + ".bin"), dtype="<f2"), expected, **TOLERANCES["device"]
            )
        )
    checks["logits"] = all(result["passed"] for result in logits)
    cache = []
    if "cache" in case:
        contract = case["cache"]
        dtype = {"F16": "<f2", "F32": "<f4"}[contract["dtype"]]
        shape = tuple(contract["shape"])
        consumed = case["reference"]["consumed"]
        for part, field in (("key", "keys"), ("value", "values")):
            filename = (
                f"layer23.present_{part}.bin" if fullgraph else f"measured.0.0.layer.23.{field}.bin"
            )
            actual = np.fromfile(device / filename, dtype=dtype).reshape(shape)
            expected_cache = np.fromfile(resolve_path(contract[field]), dtype=dtype).reshape(shape)
            result = compare_arrays(
                actual[:, :consumed], expected_cache[:, :consumed], **TOLERANCES["device"]
            )
            result["unwritten_slots_bit_zero"] = not np.any(actual[:, consumed:].view("u1"))
            cache.append(result)
        checks["cache"] = all(
            result["passed"] and result["unwritten_slots_bit_zero"] for result in cache
        )
    native = read_json(device / ("execution.json" if fullgraph else "run.json"))
    checks["release_budget"] = (
        native["released"]
        and native["const_live_bytes"] == native["nonconst_live_bytes"] == 0
        and max(native["const_peak_bytes"], native["nonconst_peak_bytes"]) <= 2**30
    )
    if not fullgraph:
        checks["benchmark_complete"] = read_json(device / "benchmark.json")["completed"]
    if case.get("device_reset"):
        checks["device_reset"] = (
            read_json(device / "reset-check.json")["passed"]
            and native["kv_reset_uploads_per_request"] == 0
        )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        logits=logits,
        cache=cache,
        native=native,
        generations=generations,
        performance=summarize(generations) if all(checks.values()) else None,
    )


def validate_numerical_failure(previous):
    """Require a completed, released numeric failure before an explicit repair diagnostic."""
    execution = previous.get("execution", {})
    directory = resolve_path(previous["evidence"])
    if execution.get("returncode") == 0:
        # Qwen writes execution.json; its independent Python oracle, after
        # process exit, may reject numbers despite a clean native completion.
        native = read_json(directory / "device/execution.json")
        result = read_json(directory / "result.json")
        failed = [name for name, passed in result.get("checks", {}).items() if passed is False]
        # Legacy records may carry the retired private audit; a recorded failure
        # still blocks recovery, while current runs use numerical/release contracts.
        numerical_failure = (
            native.get("status") == "executed"
            and native.get("phase") == "complete"
            and result.get("passed") is False
            and not result.get("error")
            and result.get("compilation", {"passed": True}).get("passed") is True
            and bool(failed)
            and all(name.startswith("request.") for name in failed)
        )
    else:
        native = read_json(directory / "device/run.json")
        numerical_failure = (
            execution.get("returncode") == 1 and native.get("status") == "numerical_mismatch"
        )
    if (
        not numerical_failure
        or execution.get("timeout") is not False
        or execution.get("device_recovery_required") is not False
        or execution.get("process_group_exited") is not True
        or execution.get("remaining_processes") != []
        or native.get("released") is not True
        or any(native.get(f"{segment}_live_bytes") != 0 for segment in ("const", "nonconst"))
        or any(
            native.get(f"{segment}_peak_bytes", 2**30 + 1) > 2**30
            for segment in ("const", "nonconst")
        )
        or not healthy(previous.get("kernel_gate", {"faults": ["unknown"], "owners": ""}))
    ):
        raise ValueError("numerical diagnostic requires a completed healthy numerical failure")


def run_case(
    case,
    identity,
    output,
    *,
    sudo_read_only=False,
    timeout=300,
    evaluate_result=evaluate,
    numerical_diagnostic=False,
    board_power_cycled=False,
    sdk_timing=False,
):
    if not 0 < timeout <= 300:
        raise ValueError("NP101 device runs require a timeout in (0, 300] seconds")
    if numerical_diagnostic and board_power_cycled:
        raise ValueError("board power-cycle and numerical diagnostics are separate recovery paths")
    sdk = resolve_path(identity["directory"])
    binary = resolve_path(case["binary"]).resolve(strict=True)
    for name, digest in identity["sha256"].items():
        if sha256(sdk / name) != digest:
            raise ValueError(f"compiled validation requires the pinned SDK: {name}")
    if Path("/sys/module/galcore/srcversion").read_text().strip() != identity["module_srcversion"]:
        raise ValueError("compiled validation requires the pinned loaded driver")
    output.mkdir(parents=True, exist_ok=False)
    device = output / "device"
    arguments = [
        str(device)
        if value == "{output}"
        else value
        if index in case.get("literal_arguments", [])
        else str(resolve_path(value))
        if value != "-"
        else value
        for index, value in enumerate(case["arguments"])
    ]
    write_json(
        output / "plan.json",
        dict(case={k: str(v) if isinstance(v, Path) else v for k, v in case.items()}, sdk=identity),
    )
    marker = RECOVERY_ROOT / "np101-recovery-required.json"
    # Explicitly authorized diagnostics may follow a completed, released numerical
    # failure. This exception never admits a kernel fault or an incomplete SDK run.
    if numerical_diagnostic:
        previous = read_json(marker)
        validate_numerical_failure(previous)
        write_json(output / "previous-numerical-marker.json", previous)
    report = run_guarded(
        binary,
        arguments,
        output,
        sdk,
        timeout,
        evaluate_result=lambda evidence: evaluate_result(case, device),
        sudo_read_only=sudo_read_only,
        sdk_timing=sdk_timing,
        numerical_diagnostic=numerical_diagnostic,
        board_power_cycled=board_power_cycled,
    )
    write_json(output / "result.json", report)
    return report
