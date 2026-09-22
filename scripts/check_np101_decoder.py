#!/usr/bin/env python3
"""Validate complete decoder layers with shared activations and persistent state."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.qwen3_5.validation_decoder import evaluate, prepare
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
    parser.add_argument("--trace", type=Path, help="deployment-fp16 layer-0/3 trace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, help="override the selected native test executable")
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", help="explicit ordered layer slice")
    parser.add_argument("--steps", type=int, choices=(2, 8, 32, 512), default=32)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--shader-header", type=Path, default=Path("/usr/inc/CL/cl_viv_vx_ext.h"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--no-trace", action="store_true", help="collect host timings without strace"
    )
    parser.add_argument("--diagnostic", action="store_true", help="numerical pass may exit zero")
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error("use a new output directory and a positive timeout")
    if args.trace is None:
        parser.error("Decoder validation requires --trace")
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        print("Preparing the official decoder CPU reference ...", flush=True)
        metadata = prepare(
            args.model.resolve(),
            args.trace.resolve(),
            output / "fixture",
            args.steps,
            args.layers[0] if args.layers else args.first_layer,
            args.layers,
        )
        if args.prepare_only:
            print(f"Prepared {args.steps} steps: {output / 'fixture'}")
            return 0
        with device_lock(ROOT / ".cache/runs"):
            target = "np101_decoder_check"
            binary = output / target
            snapshot_binary((args.binary or ROOT / "build/tests" / target).resolve(), binary)
            record_sources(ROOT, binary, output / "sources.json")
            print(f"Running {target} ...", flush=True)
            arguments = [
                str(args.model.resolve()),
                str(output / "fixture"),
                str(output / "device"),
                str(args.steps),
                str(args.layers[0] if args.layers else args.first_layer),
            ]
            evidence = run_device(
                binary,
                arguments,
                output / "device",
                args.sdk_lib,
                args.timeout,
                shader_header=args.shader_header,
                trace_driver=not args.no_trace,
            )
        report = evaluate(output / "fixture", output / "device", metadata, evidence)
        write_json(output / "decoder.json", report)
        failures = [key for key, value in report["checks"].items() if not value["passed"]]
        print(
            f"{report['status']}: {len(failures)} failed comparisons; report: {output / 'decoder.json'}"
        )
        if report["status"] == "failed":
            return 1
        return 0 if args.diagnostic else 2
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(f"Decoder validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
