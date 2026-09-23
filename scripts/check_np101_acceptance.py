#!/usr/bin/env python3
"""Validate complete OPT inference and measure repeated requests on resident models."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.acceptance import evaluate_benchmark, summarize, write_results
from specferry.models.opt.generation import prepare_request
from specferry.models.opt.validation_generation import (
    evaluate,
    prepare,
    prepare_selection,
    reference_generation,
)
from specferry.validation.device import (
    clean_execution,
    device_lock,
    record_sources,
    run_device,
    snapshot_binary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def preflight_anomaly(evidence, limit):
    """A latency outlier is separate from numerical failure and driver recovery."""
    return (
        evidence.get("wall_seconds", 0) > limit
        or evidence.get("driver_waits", {}).get("calls_at_least_one_second", 0) > 0
    )


def selection_only(args, output):
    fixture, device = output / "selection-fixture", output / "selection"
    metadata = prepare_selection(fixture)
    if args.prepare_only:
        return {"status": "prepared", "numerical_pass": False}
    with device_lock(ROOT / ".cache/runs"):
        binary = snapshot_binary(args.io_binary, output / "np101_opt_io_check")
        record_sources(ROOT, binary, output / "sources.json")
        print("Running selection only (synthetic weights; no checkpoint needed) ...", flush=True)
        evidence = run_device(
            binary,
            [str(fixture / "deployment"), str(fixture), str(device)],
            device,
            args.sdk_lib,
            args.selection_timeout,
            shader_header=args.shader_header,
            sdk_timing=args.sdk_timing,
            progress_interval=args.progress_interval,
        )
        report = evaluate(fixture, device, metadata, evidence)
        report["numerical_pass"] = report["status"] == "numerical_pass"
        report["latency_anomaly"] = preflight_anomaly(evidence, args.selection_warning)
        write_json(output / "selection.json", report)
        return report


def prepare_suite(args, output):
    selection = prepare_selection(output / "selection-fixture")
    requests = []
    prompts = [(prompt, None) for prompt in args.prompt]
    if args.lengths:
        passage = (ROOT / "tests/fixtures/opt_continuation.txt").read_text()
        # Exact tokenizer prefixes of continuous prose; no random IDs or padding.
        # A prefix may end in the middle of a sentence or subword.
        text = "Chapter 1\n" + passage
        prompts = [(text, length) for length in args.lengths]
    for index, (prompt, length) in enumerate(prompts):
        directory = output / f"request-{index}"
        capacity = (
            args.capacity
            if length is None
            else min(2048, max(args.capacity, length + args.max_new_tokens - 1))
        )
        request, _ = prepare_request(
            args.model, args.checkpoint, directory, prompt, capacity, prompt_tokens=length
        )
        if request["selection"]["mode"] != "greedy":
            raise ValueError("repeatable token acceptance requires greedy checkpoint defaults")
        reference = reference_generation(args.checkpoint, request, args.max_new_tokens)
        write_json(directory / "reference.json", reference)
        requests.append({"directory": str(directory), "request": request, "reference": reference})
    # A is longer than B: B must mask the old suffix; the second A must reset position.
    reuse = []
    for index, prompt in enumerate(
        (
            "The library stood beside a quiet river. Anna opened the door and saw",
            "The capital of France is",
        )
    ):
        directory = output / f"reset-request-{index}"
        request, _ = prepare_request(args.model, args.checkpoint, directory, prompt, 64)
        reference = reference_generation(args.checkpoint, request, args.max_new_tokens)
        write_json(directory / "reference.json", reference)
        reuse.append({"directory": str(directory), "request": request, "reference": reference})
    # A real prompt followed by CPU-generated tokens crosses the prefill/decode
    # boundary while keeping all-layer diagnostic readbacks small.
    trajectory = (reuse[1]["request"]["tokens"] + reuse[1]["reference"]["tokens"])[: args.steps]
    teacher = prepare(
        args.model,
        args.checkpoint,
        output / "teacher-fixture",
        "teacher",
        24,
        64,
        len(trajectory),
        input_ids=trajectory,
    )
    boundary = prepare(
        args.model, args.checkpoint, output / "boundary-fixture", "teacher", 24, 8, 8
    )
    return {
        "selection": selection,
        "teacher": teacher,
        "boundary": boundary,
        "requests": requests,
        "reuse": reuse,
    }


def run_suite(args, output, fixtures):
    stages, benchmarks = [], []
    expected = args.cycles * len(fixtures["requests"])

    def save(*, running=False):
        report = summarize(stages, benchmarks, expected)
        if running:
            report["status"] = "running"
        write_json(output / "acceptance.json", report)
        write_results(output / "results.md", report)
        return report

    with device_lock(ROOT / ".cache/runs"):
        if not os.access("/dev/galcore", os.R_OK | os.W_OK):
            raise RuntimeError("/dev/galcore is unavailable or not readable/writable")
        binary = snapshot_binary(args.binary, output / "specferry_opt_generate")
        io_binary = snapshot_binary(args.io_binary, output / "np101_opt_io_check")
        record_sources(ROOT, binary, output / "sources.json")
        first_evidence = None

        def execute(executable, arguments, directory, *, measurement=False):
            nonlocal first_evidence
            evidence = run_device(
                executable,
                arguments,
                directory,
                args.sdk_lib,
                getattr(args, "selection_timeout", args.timeout)
                if directory.name == "selection"
                else args.timeout,
                shader_header=args.shader_header,
                trace_driver=not measurement,
                print_targets=not measurement,
                sdk_timing=getattr(args, "sdk_timing", "off"),
                progress_interval=getattr(args, "progress_interval", 10),
            )
            if first_evidence is None:
                first_evidence = evidence
            if any(evidence[key] != first_evidence[key] for key in ("boot_id", "sdk_sha256")):
                raise RuntimeError("boot or SDK identity changed during acceptance")
            return evidence

        for name in ("selection", "teacher", "boundary"):
            fixture = output / f"{name}-fixture"
            device = output / name
            if name == "selection":
                executable = io_binary
                arguments = [str(fixture / "deployment"), str(fixture), str(device)]
            else:
                executable = binary
                arguments = [
                    str(args.model),
                    str(fixture),
                    str(device),
                    "teacher",
                    "24",
                    "0",
                    str(fixture / "tokens.txt"),
                ]
            print(f"Running {name} correctness gate ...", flush=True)
            evidence = execute(executable, arguments, device)
            report = evaluate(fixture, device, fixtures[name], evidence)
            write_json(device / "validation.json", report)
            passed = clean_execution(evidence) and report["status"] == "numerical_pass"
            stages.append(
                {"name": name, "passed": passed, "report": str(device / "validation.json")}
            )
            if not passed:
                return save()
            if name == "selection" and preflight_anomaly(
                evidence, getattr(args, "selection_warning", 30)
            ):
                report = save()
                report.update(
                    status="latency_anomaly",
                    latency_anomaly=True,
                    error="Selection passed numerically but exceeded the latency guard; inspect it before full-model runs.",
                )
                write_json(output / "acceptance.json", report)
                write_results(output / "results.md", report)
                return report
            save(running=True)

        kv_binary = snapshot_binary(args.kv_binary, output / "np101_kv_cache_check")
        for capacity in sorted({8, 64, *(r["request"]["capacity"] for r in fixtures["requests"])}):
            device = output / f"kv-{capacity}"
            ev = execute(kv_binary, [str(device), "16", "64", str(capacity)], device)
            cache = (
                json.loads((device / "cache.json").read_text())
                if (device / "cache.json").is_file()
                else {}
            )
            passed = (
                clean_execution(ev)
                and cache.get("status") == "numerical_pass"
                and cache.get("released") is True
            )
            stages.append({"name": "kv", "capacity": capacity, "passed": passed})
            if not passed:
                return save()

        sequence = [fixtures["reuse"][i] for i in (0, 1, 0)]
        playlist = output / "requests.txt"
        playlist.write_text(
            "".join(str(Path(r["directory"]) / "tokens.txt") + "\n" for r in sequence)
        )
        device = output / "reset"
        ev = execute(
            binary,
            [
                str(args.model),
                sequence[0]["directory"],
                str(device),
                "requests",
                "24",
                str(args.max_new_tokens),
                str(playlist),
            ],
            device,
            measurement=True,
        )
        report = evaluate_benchmark(
            device,
            sequence[0]["request"],
            sequence[0]["reference"],
            ev,
            0,
            3,
            args.max_new_tokens,
            sequence=[(r["request"], r["reference"]) for r in sequence],
        )
        write_json(device / "validation.json", report)
        stages.append({"name": "reset", "passed": report["status"] == "numerical_pass"})
        if report["status"] != "numerical_pass":
            return save()
        save(running=True)

        for cycle in range(args.cycles):
            for index, prepared in enumerate(fixtures["requests"]):
                device = output / f"cycle-{cycle}-request-{index}"
                directory = Path(prepared["directory"])
                consumed_budget = (
                    len(prepared["request"]["tokens"]) + len(prepared["reference"]["tokens"]) - 1
                ) * (args.warmups + args.repeats)
                available = (
                    int(
                        next(
                            line.split()[1]
                            for line in Path("/proc/meminfo").read_text().splitlines()
                            if line.startswith("MemAvailable:")
                        )
                    )
                    * 1024
                )
                if available < consumed_budget * 4 * 1024**2 + 4 * 1024**3:
                    raise RuntimeError(
                        "insufficient host headroom for observed SDK growth plus model"
                    )
                print(
                    f"Measuring fresh model {cycle}, prompt {index} ({len(prepared['request']['tokens'])} tokens) ...",
                    flush=True,
                )
                evidence = execute(
                    binary,
                    [
                        str(args.model),
                        str(directory),
                        str(device),
                        "benchmark",
                        "24",
                        str(args.max_new_tokens),
                        str(directory / "tokens.txt"),
                        str(args.warmups),
                        str(args.repeats),
                    ],
                    device,
                    measurement=True,
                )
                report = evaluate_benchmark(
                    device,
                    prepared["request"],
                    prepared["reference"],
                    evidence,
                    args.warmups,
                    args.repeats,
                    args.max_new_tokens,
                )
                report.update(cycle=cycle, request=index, directory=str(device))
                tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
                report["prompt"] = prepared["request"]["prompt"]
                for run in report["runs"]:
                    run["text"] = tokenizer.decode(
                        run["generation"].get("tokens", []), skip_special_tokens=True
                    )
                write_json(device / "benchmark.json", report)
                if report["metrics"]:
                    print(
                        f"  decode: {report['metrics']['decode_tokens_per_second']} tokens/s; "
                        f"TTFT: {report['metrics']['first_token_seconds']['median']:.3f} s",
                        flush=True,
                    )
                if report["sdk_timings"]:
                    print(
                        "  measured SDK call seconds by phase: "
                        + json.dumps(report["sdk_timings"]["measured_phase_seconds"]),
                        flush=True,
                    )
                benchmarks.append(report)
                if report["status"] != "numerical_pass":
                    return save()
                save(running=len(benchmarks) < expected)
    return save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/specferry_opt_generate")
    parser.add_argument("--io-binary", type=Path, default=ROOT / "build/tests/np101_opt_io_check")
    parser.add_argument("--kv-binary", type=Path, default=ROOT / "build/tests/np101_kv_cache_check")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        help="exact natural-text token prefix lengths, e.g. 32 128 512 2048",
    )
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--shader-header", type=Path, default=Path("/usr/inc/CL/cl_viv_vx_ext.h"))
    parser.add_argument(
        "--prompt", action="append", help="repeatable; defaults to two fixed prompts"
    )
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument(
        "--steps", type=int, default=8, help="maximum natural teacher trajectory snapshots (1..64)"
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=2, help="fresh model processes per prompt")
    parser.add_argument("--timeout", type=int, default=1800, help="maximum seconds per SDK process")
    parser.add_argument("--selection-only", action="store_true", help="synthetic IO gate only")
    parser.add_argument("--selection-timeout", type=int, default=120)
    parser.add_argument("--selection-warning", type=float, default=30)
    parser.add_argument(
        "--progress-interval", type=int, default=10, help="heartbeat seconds; 0 disables"
    )
    parser.add_argument("--sdk-timing", choices=("off", "summary", "calls"), default="off")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="numerical success may exit zero")
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.capacity <= 2048
        or not 1 <= args.steps <= 64
        or not 1 <= args.max_new_tokens <= 512
        or not 0 <= args.warmups <= 5
        or not 1 <= args.repeats <= 20
        or not 1 <= args.cycles <= 5
        or (args.lengths and any(not 1 <= value <= 2048 for value in args.lengths))
        or args.timeout < 1
        or args.selection_timeout < 1
        or not 0 < args.selection_warning <= args.selection_timeout
        or args.progress_interval < 0
    ):
        parser.error(
            "use a fresh output, capacity 1..2048, steps 1..64, new tokens 1..512, "
            "warmups 0..5, repeats 1..20, cycles 1..5, and a positive timeout"
        )
    if args.sdk_timing == "calls" and not args.selection_only:
        parser.error("per-call file logging is diagnostic only; use --selection-only")
    if not args.prepare_only and any(
        os.environ.get(name) not in (None, "", "0")
        for name in ("VIV_VX_PROFILE", "VIV_MEMORY_PROFILE")
    ):
        parser.error("disable VIV_VX_PROFILE and VIV_MEMORY_PROFILE before collecting timings")
    args.prompt = args.prompt or ["The capital of France is", "For dinner tonight, I will cook"]
    args.model, args.checkpoint = args.model.resolve(), args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    try:
        if args.selection_only:
            report = selection_only(args, output)
            print(f"{report['status']}: {output}", flush=True)
            if args.prepare_only:
                return 0
            if report["latency_anomaly"]:
                print("Latency anomaly: inspect selection.json and the SDK/driver logs.")
                return 1
            return 0 if report["numerical_pass"] else 1
        print("Preparing independent references and checkpoint policies ...", flush=True)
        fixtures = prepare_suite(args, output)
        write_json(
            output / "suite.json",
            {
                "capacity": args.capacity,
                "teacher_steps": fixtures["teacher"]["count"],
                "max_new_tokens": args.max_new_tokens,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "cycles": args.cycles,
                "fixtures": fixtures,
            },
        )
        if args.prepare_only:
            print(f"Prepared without device access: {output}")
            return 0
        report = run_suite(args, output, fixtures)
        print(f"{report['status']}: {output / 'acceptance.json'}")
        return (0 if args.diagnostic else 2) if report["numerical_pass"] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        # Preserve incremental results, but never leave a previous pass after a late failure.
        path = output / "acceptance.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        report.update(status="failed", numerical_pass=False, error=str(error))
        write_json(path, report)
        print(f"Acceptance failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
