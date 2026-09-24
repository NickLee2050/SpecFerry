#!/usr/bin/env python3
"""Prepare or run one bounded experimental graph-inference gate at a time."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.optimization import (
    CASES,
    capacity_comparison,
    evaluate,
    prepare,
    verify_prepared,
    write_results,
)
from specferry.validation.device import (
    clean_execution,
    device_lock,
    fingerprint,
    host_boot_id,
    record_sources,
    run_device,
    snapshot_binary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def prerequisites(args):
    gate = args.cache_gate
    if gate is None:
        raise ValueError("run the current indexed-cache gate first; pass --cache-gate DIRECTORY")
    summary = json.loads((gate / "summary.json").read_text())
    evidence = json.loads((gate / "execution-evidence.json").read_text())
    mode = "stack" if args.case == "decode" else "stack-block"
    if (
        summary["status"] != "numerical_pass"
        or summary["mode"] != mode
        or not clean_execution(evidence)
        or evidence["boot_id"] != host_boot_id()
        or evidence["binary_sha256"] != fingerprint(ROOT / "build/tests/np101_graph_cache_check")
    ):
        raise ValueError("cache prerequisite did not pass on this boot/current build")
    for name, expected in evidence["sdk_sha256"].items():
        if Path(name).name != name or fingerprint(Path("/usr/lib/ljmicro") / name) != expected:
            raise ValueError("SDK changed since cache validation")
    previous_name = CASES[args.case]["previous"]
    previous = None
    if previous_name:
        if args.previous is None:
            raise ValueError(f"pass --previous DIRECTORY for the completed {previous_name} gate")
        previous = json.loads((args.previous / "summary.json").read_text())
        if (
            previous["status"] != "numerical_pass"
            or previous["case"] != previous_name
            or previous["evidence"]["boot_id"] != host_boot_id()
            or not clean_execution(previous["evidence"])
            or previous["evidence"]["sdk_sha256"] != evidence["sdk_sha256"]
            or previous["prepared_sha256"] != fingerprint(args.prepared / "prepared.json")
            or previous["evidence"]["binary_sha256"]
            != fingerprint(ROOT / "build/bin/specferry_opt_graph_generate")
        ):
            raise ValueError(
                "preceding inference gate is incomplete or uses a different build/boot"
            )
    return evidence, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cache-gate", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    args = parser.parse_args()
    if args.output.exists() or (args.timeout is not None and args.timeout <= 0):
        parser.error("use a fresh output and positive timeout")
    if not args.prepare_only and (args.prepared is None or args.case is None):
        parser.error("device execution requires --prepared and --case")
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        if args.prepare_only:
            prepare(args.model.resolve(), args.checkpoint.resolve(), output)
            print(f"Prepared CPU token/KV references without device access: {output}")
            return 0
        metadata = verify_prepared(args.prepared.resolve(), args.model.resolve())
        prerequisite, previous = prerequisites(args)
        case = metadata["cases"][args.case]
        expected = output / "expected-tokens.txt"
        expected.write_text(
            "".join(" ".join(map(str, item["expected_tokens"])) + "\n" for item in case["requests"])
        )
        cache_readback = args.case in ("decode", "prefill")
        with device_lock(ROOT / ".cache/runs"):
            binary = snapshot_binary(
                ROOT / "build/bin/specferry_opt_graph_generate",
                output / "specferry_opt_graph_generate",
            )
            record_sources(ROOT, binary, output / "sources.json")
            os.environ["VIV_VX_PROFILE"] = "0"
            os.environ["VIV_MEMORY_PROFILE"] = "0"
            device = output / "device"
            fixture = args.prepared.resolve() / args.case
            evidence = run_device(
                binary,
                [
                    str(args.model.resolve()),
                    str(fixture),
                    str(device),
                    str(case["block"]),
                    str(case["maximum"]),
                    "prefix" if cache_readback else "none",
                    str(expected),
                ],
                device,
                Path("/usr/lib/ljmicro"),
                args.timeout or CASES[args.case]["timeout"],
                shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
                trace_driver=False,
                print_targets=False,
                sdk_timing="summary",
                progress_interval=10,
            )
        if evidence["sdk_sha256"] != prerequisite["sdk_sha256"]:
            raise RuntimeError("SDK changed since cache validation")
        report = evaluate(case, fixture, device, evidence, cache_readback)
        report["case"] = args.case
        report["prepared"] = str(args.prepared.resolve())
        report["prepared_sha256"] = fingerprint(args.prepared / "prepared.json")
        if args.case == "capacity128" and report["status"] == "numerical_pass":
            report["capacity_comparison"] = capacity_comparison(previous, report)
        write_json(output / "summary.json", report)
        write_results(output / "results.md", report)
        print(
            json.dumps(
                {"status": report["status"], "case": args.case, "metrics": report["metrics"]}
            )
        )
        return 0 if report["status"] == "numerical_pass" else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
