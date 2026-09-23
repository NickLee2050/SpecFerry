#!/usr/bin/env python3
"""Validate complete OPT inference and measure repeated requests on resident models."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.acceptance import evaluate_benchmark, summarize
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


def prepare_suite(args, output):
    selection = prepare_selection(output / "selection-fixture")
    teacher = prepare(
        args.model,
        args.checkpoint,
        output / "teacher-fixture",
        "teacher",
        24,
        args.capacity,
        min(args.steps, args.capacity),
    )
    requests = []
    for index, prompt in enumerate(args.prompt):
        directory = output / f"request-{index}"
        request, _ = prepare_request(args.model, args.checkpoint, directory, prompt, args.capacity)
        if request["selection"]["mode"] != "greedy":
            raise ValueError("repeatable token acceptance requires greedy checkpoint defaults")
        reference = reference_generation(args.checkpoint, request, args.max_new_tokens)
        write_json(directory / "reference.json", reference)
        requests.append({"directory": str(directory), "request": request, "reference": reference})
    return {"selection": selection, "teacher": teacher, "requests": requests}


def run_suite(args, output, fixtures):
    stages, benchmarks = [], []
    expected = args.cycles * len(fixtures["requests"])

    def save(*, running=False):
        report = summarize(stages, benchmarks, expected)
        if running:
            report["status"] = "running"
        write_json(output / "acceptance.json", report)
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
                args.timeout,
                shader_header=args.shader_header,
                trace_driver=not measurement,
                print_targets=not measurement,
            )
            if first_evidence is None:
                first_evidence = evidence
            if any(evidence[key] != first_evidence[key] for key in ("boot_id", "sdk_sha256")):
                raise RuntimeError("boot or SDK identity changed during acceptance")
            return evidence

        for name in ("selection", "teacher"):
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
            save(running=True)

        for cycle in range(args.cycles):
            for index, prepared in enumerate(fixtures["requests"]):
                device = output / f"cycle-{cycle}-request-{index}"
                directory = Path(prepared["directory"])
                print(f"Measuring fresh model {cycle}, prompt {index} ...", flush=True)
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
                write_json(device / "benchmark.json", report)
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
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--shader-header", type=Path, default=Path("/usr/inc/CL/cl_viv_vx_ext.h"))
    parser.add_argument(
        "--prompt", action="append", help="repeatable; defaults to two fixed prompts"
    )
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8, help="teacher-forced correctness steps")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=2, help="fresh model processes per prompt")
    parser.add_argument("--timeout", type=int, default=1800, help="maximum seconds per SDK process")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="numerical success may exit zero")
    args = parser.parse_args()
    if (
        args.output.exists()
        or not 1 <= args.capacity <= 512
        or not 1 <= args.steps <= 512
        or not 1 <= args.max_new_tokens <= 512
        or not 0 <= args.warmups <= 5
        or not 2 <= args.repeats <= 20
        or not 2 <= args.cycles <= 5
        or args.timeout < 1
    ):
        parser.error(
            "use a fresh output, capacity/steps/tokens 1..512, warmups 0..5, repeats 2..20, cycles 2..5, positive timeout"
        )
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
        print("Preparing independent references and checkpoint policies ...", flush=True)
        fixtures = prepare_suite(args, output)
        write_json(
            output / "suite.json",
            {
                "capacity": args.capacity,
                "teacher_steps": min(args.steps, args.capacity),
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
