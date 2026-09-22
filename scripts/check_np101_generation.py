#!/usr/bin/env python3
"""Validate OPT input/output or resident decoder prefixes against the official CPU model."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.validation_generation import evaluate, prepare, prepare_selection
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
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("io", "selection", "teacher"), default="teacher")
    parser.add_argument("--layers", type=int, choices=(4, 8, 24), default=24)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error("use a new output directory and positive timeout")
    output = args.output.resolve()
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    try:
        print("Preparing independent expectations ...", flush=True)
        fixture = output / "fixture"
        deployment = args.model.resolve()
        metadata = (
            prepare_selection(fixture)
            if args.mode == "selection"
            else prepare(
                args.model.resolve(),
                args.checkpoint.resolve(),
                fixture,
                args.mode,
                args.layers,
                args.capacity,
                args.steps,
            )
        )
        if args.prepare_only:
            print(f"Prepared: {fixture}")
            return 0
        with device_lock(ROOT / ".cache/runs"):
            if args.mode != "teacher":
                if args.mode == "selection":
                    deployment = fixture / "deployment"
                source = ROOT / "build/tests/np101_opt_io_check"
                arguments = [str(deployment), str(fixture), str(output / "device")]
            else:
                source = ROOT / "build/bin/specferry_opt_generate"
                arguments = [
                    str(args.model.resolve()),
                    str(fixture),
                    str(output / "device"),
                    "teacher",
                    str(args.layers),
                    "0",
                    str(fixture / "tokens.txt"),
                ]
            binary = output / source.name
            snapshot_binary(source, binary)
            record_sources(ROOT, binary, output / "sources.json")
            print(
                f"Running {args.mode}"
                + (f", {args.layers} resident layers" if args.mode == "teacher" else "")
                + " ...",
                flush=True,
            )
            evidence = run_device(
                binary,
                arguments,
                output / "device",
                Path("/usr/lib/ljmicro"),
                args.timeout,
                shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
            )
        report = evaluate(fixture, output / "device", metadata, evidence)
        write_json(output / "validation.json", report)
        print(f"{report['status']}: {report['first_failures']}; {output / 'validation.json'}")
        return (0 if args.diagnostic else 2) if report["status"] == "numerical_pass" else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
