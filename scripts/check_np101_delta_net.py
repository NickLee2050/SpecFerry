#!/usr/bin/env python3
"""Validate the real layer-0 DeltaNet mixer, sequential state and reset."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.qwen3_5.validation_delta_net import evaluate, prepare
from specferry.validation.device import (
    device_lock,
    record_sources,
    run_device,
    snapshot_binary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument("--trace", type=Path, required=True, help="deployment-fp16 layer-0/3 trace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/tests/np101_delta_net_check")
    parser.add_argument("--steps", type=int, choices=(1, 2, 4, 8, 32), default=32)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--shader-header", type=Path, default=Path("/usr/inc/CL/cl_viv_vx_ext.h"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="numerical pass may exit zero")
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error("use a new output directory and a positive timeout")
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        print("Preparing the official single-layer CPU reference ...", flush=True)
        metadata = prepare(
            args.model.resolve(), args.trace.resolve(), output / "fixture", args.steps
        )
        if args.prepare_only:
            print(f"Prepared {args.steps} steps: {output / 'fixture'}")
            return 0
        with device_lock(ROOT / ".cache/runs"):
            binary = output / "np101_delta_net_check"
            snapshot_binary(args.binary.resolve(), binary)
            record_sources(ROOT, binary, output / "sources.json")
            print("Running fixed-graph DeltaNet trajectories and reset checks ...", flush=True)
            evidence = run_device(
                binary,
                [
                    str(args.model.resolve()),
                    str(output / "fixture"),
                    str(output / "device"),
                    str(args.steps),
                ],
                output / "device",
                args.sdk_lib,
                args.timeout,
                shader_header=args.shader_header,
            )
        report = evaluate(output / "fixture", output / "device", metadata, evidence)
        write_json(output / "delta-net.json", report)
        failures = [key for key, value in report["checks"].items() if not value["passed"]]
        print(
            f"{report['status']}: {len(failures)} failed comparisons; report: {output / 'delta-net.json'}"
        )
        if report["status"] == "failed":
            return 1
        return 0 if args.diagnostic else 2
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(f"DeltaNet validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
