"""Pinned compiled-model acceptance with preparation outside generation timings."""

import csv
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from specferry.data.checkpoint import sha256
from specferry.models.opt.config import TOLERANCES
from specferry.validation.arrays import compare_arrays
from specferry.validation.device import (
    RECOVERY_ROOT,
    VERIFY_TRACE_OPENVX_SHA256,
    clean_execution,
    device_lock,
    run_device,
    write_json,
)
from specferry.validation.recovery_controls import healthy, snapshot

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


def audit_compilation(device):
    with (device / "verify-internals.tsv-nodes.tsv").open() as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    graphs = [row for row in records if row["event"] == "summary"]
    operations = [row for row in records if row["event"] == "node"]
    extensions = sum("extension" in row["kernel_name"] for row in operations)
    passed = (
        bool(graphs)
        and extensions < 1024
        and all(
            row["scan_complete"] == "1"
            and row["read_errors"] == row["null_operations"] == row["zero_operations"] == "0"
            for row in graphs
        )
    )
    return dict(
        passed=passed, graphs=len(graphs), outer_operations=len(operations), extensions=extensions
    )


def run_case(case, identity, output, *, sudo_read_only=False, timeout=600, audit_verify=False):
    sdk = resolve_path(identity["directory"])
    for name, digest in identity["sha256"].items():
        if sha256(sdk / name) != digest:
            raise ValueError(f"compiled validation requires the pinned SDK: {name}")
    if Path("/sys/module/galcore/srcversion").read_text().strip() != identity["module_srcversion"]:
        raise ValueError("compiled validation requires the pinned loaded driver")
    if audit_verify:
        (ROOT / "build/tests/libnp101_verify_trace.so").resolve(strict=True)
        if sha256(sdk / "libOpenVX.so") != VERIFY_TRACE_OPENVX_SHA256:
            raise ValueError("Verify-only compilation audit requires pinned v105")
    (ROOT / "docs/np101-recovery-incidents.md").read_text()
    output.mkdir(parents=True, exist_ok=False)
    device = output / "device"
    arguments = [
        str(device) if value == "{output}" else str(resolve_path(value)) if value != "-" else value
        for value in case["arguments"]
    ]
    write_json(
        output / "plan.json",
        dict(case={k: str(v) if isinstance(v, Path) else v for k, v in case.items()}, sdk=identity),
    )
    marker = RECOVERY_ROOT / "np101-recovery-required.json"
    with device_lock(RECOVERY_ROOT):
        preflight = snapshot(output / "preflight", sudo_read_only)
        if not healthy(preflight):
            raise RuntimeError("compiled validation preflight failed; board not started")
        write_json(
            marker,
            dict(
                boot_id=preflight["boot_id"],
                members=[],
                evidence=str(output),
                pid_namespace=os.readlink("/proc/self/ns/pid"),
                reason="Compiled validation and delayed checks are incomplete",
            ),
        )
        os.environ["SPECFERRY_VERIFY_EXECUTION_TRACE"] = "0"
        report = dict(passed=False)
        try:
            evidence = run_device(
                case["binary"],
                arguments,
                device,
                sdk,
                timeout,
                recovery_control=True,
                verify_trace=ROOT / "build/tests/libnp101_verify_trace.so"
                if audit_verify
                else None,
                require_compiled_operations=audit_verify,
                abort_check=lambda: bool(snapshot(output / "monitor", sudo_read_only)["faults"]),
            )
            report["execution"] = evidence
            immediate = snapshot(output / "immediate", sudo_read_only)
            print("Process exited; waiting 45 seconds for delayed kernel checks", flush=True)
            time.sleep(45)
            postflight = snapshot(output / "postflight", sudo_read_only)
            report.update(immediate=immediate, postflight=postflight)
            if clean_execution(evidence):
                report.update(evaluate(case, device))
                if audit_verify:
                    report["compilation"] = audit_compilation(device)
                    report["passed"] &= report["compilation"]["passed"]
            report["passed"] = report["passed"] and healthy(immediate) and healthy(postflight)
        except (OSError, ValueError, KeyError, RuntimeError, KeyboardInterrupt) as error:
            report["error"] = str(error)
        write_json(output / "result.json", report)
        if report["passed"]:
            shutil.copy2(marker, output / "completed-run-marker.json")
            marker.unlink()
        else:
            phase = next(
                (
                    path
                    for path in [device / "phase.json", device / "execution.json"]
                    if path.exists()
                ),
                None,
            )
            incident = dict(
                boot_id=preflight["boot_id"],
                evidence=str(output),
                reason="Compiled validation failed or did not complete health checks",
                execution=report.get("execution"),
                kernel_gate=report.get("postflight"),
                phase=read_json(phase) if phase else None,
                pid_namespace=os.readlink("/proc/self/ns/pid"),
                members=report.get("execution", {}).get("remaining_processes", []),
                established_cause="Not established; inspect phase/API and kernel timestamps",
                next_run_rule="Stop board work and preserve marker; review the incident ledger",
            )
            write_json(output / "incident.json", incident)
            write_json(marker, incident)
    return report
