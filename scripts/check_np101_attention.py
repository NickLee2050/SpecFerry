#!/usr/bin/env python3
"""Validate layer-3 Attention, append-only KV storage, truncation and reset."""

import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.attention import evaluate, evaluate_storage, prepare
from specferry.validation.device import device_lock, fingerprint, run_device, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument("--trace", type=Path, help="deployment-fp16 layer-0/3 trace")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, help="override the selected native test executable")
    parser.add_argument(
        "--storage-only", action="store_true", help="check slot views without weights"
    )
    parser.add_argument("--steps", type=int, choices=(2, 8, 32, 512), default=32)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--shader-header", type=Path, default=Path("/usr/inc/CL/cl_viv_vx_ext.h"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="numerical pass may exit zero")
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error("use a new output directory and a positive timeout")
    if not args.storage_only and args.trace is None:
        parser.error("Attention validation requires --trace")
    if args.storage_only and args.prepare_only:
        parser.error("the storage-only test has no CPU fixture preparation phase")
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        metadata = None
        if not args.storage_only:
            print("Preparing the official single-layer CPU reference ...", flush=True)
            metadata = prepare(
                args.model.resolve(), args.trace.resolve(), output / "fixture", args.steps
            )
        if args.prepare_only:
            print(f"Prepared {args.steps} steps: {output / 'fixture'}")
            return 0
        with device_lock(ROOT / ".cache/runs"):
            target = "np101_kv_cache_check" if args.storage_only else "np101_attention_check"
            binary = output / target
            shutil.copy2((args.binary or ROOT / "build/tests" / target).resolve(), binary)
            write_json(
                output / "sources.json",
                {
                    "binary_sha256": fingerprint(binary),
                    "sources": {
                        str(path.relative_to(ROOT)): fingerprint(path)
                        for directory in ("native", "python", "scripts", "tests")
                        for path in sorted((ROOT / directory).rglob("*"))
                        if path.is_file() and path.suffix in (".cpp", ".hpp", ".py")
                    },
                },
            )
            print(f"Running {target} ...", flush=True)
            arguments = (
                [str(output / "device")]
                if args.storage_only
                else [
                    str(args.model.resolve()),
                    str(output / "fixture"),
                    str(output / "device"),
                    str(args.steps),
                ]
            )
            evidence = run_device(
                binary,
                arguments,
                output / "device",
                args.sdk_lib,
                args.timeout,
                shader_header=args.shader_header,
            )
        report = (
            evaluate_storage(output / "device", evidence)
            if args.storage_only
            else evaluate(output / "fixture", output / "device", metadata, evidence)
        )
        write_json(output / "attention.json", report)
        failures = [key for key, value in report["checks"].items() if not value["passed"]]
        print(
            f"{report['status']}: {len(failures)} failed comparisons; report: {output / 'attention.json'}"
        )
        if report["status"] == "failed":
            return 1
        return 0 if args.diagnostic else 2
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(f"Attention validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
