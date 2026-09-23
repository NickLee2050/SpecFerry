#!/usr/bin/env python3
"""Run model-independent host-growth or SDK memory-accounting diagnostics."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.device import (
    clean_execution,
    device_lock,
    record_sources,
    run_device,
    snapshot_binary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--timeout", type=int, default=120)
    subparsers = parser.add_subparsers(dest="case", required=True)
    growth = subparsers.add_parser("growth", help="one fixed copy graph, no model weights")
    growth.add_argument("--mode", choices=("fixed", "same", "advance"), default="advance")
    growth.add_argument("--iterations", type=int, default=128)
    accounting = subparsers.add_parser("accounting", help="SDK counter delta for one tensor")
    accounting.add_argument("--mib", type=int, default=8)
    accounting.add_argument("--storage", choices=("constant", "mutable"), default="constant")
    accounting.add_argument("--dtype", choices=("F16", "F32"), default="F16")
    args = parser.parse_args()
    if args.output.exists() or args.timeout < 1:
        parser.error("use a fresh output directory and a positive timeout")
    if args.case == "growth":
        if not 1 <= args.iterations <= 4096:
            parser.error("iterations must be in [1,4096]")
        name, arguments = "np101_memory_growth_check", [args.mode, str(args.iterations)]
    else:
        if not 1 <= args.mib <= 64:
            parser.error("tensor size must be in [1,64] MiB")
        name, arguments = "np101_memory_accounting_check", [str(args.mib), args.storage, args.dtype]
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        with device_lock(ROOT / ".cache/runs"):
            binary = snapshot_binary(ROOT / "build/tests" / name, output / name)
            record_sources(ROOT, binary, output / "sources.json")
            # Process-local settings; the calling shell and inference defaults are unchanged.
            os.environ["VIV_MEMORY_PROFILE"] = "1" if args.case == "accounting" else "0"
            os.environ["VIV_VX_PROFILE"] = "0"
            evidence = run_device(
                binary,
                arguments,
                output,
                args.sdk_lib,
                args.timeout,
                trace_driver=False,
                print_targets=False,
            )
            passed = clean_execution(evidence)
            report = {
                "experiment_completed": passed,
                "case": args.case,
                "arguments": arguments,
                "physical_memory_measured": False,
                "evidence": evidence,
            }
            lines = (output / "sdk.log").read_text().splitlines()
            if args.case == "growth" and passed:
                rows = [
                    parts
                    for line in lines
                    if len(parts := line.split()) == 4
                    and parts[0].isdigit()
                    and parts[1] in ("bind", "verify", "process")
                ]
                if len(rows) != args.iterations * 3:
                    raise RuntimeError("native phase observations are incomplete")
                baseline, final = int(rows[2][2]), int(rows[-1][2])
                observations = {
                    "iterations": args.iterations,
                    "first_copy_rss_bytes": baseline,
                    "last_copy_rss_bytes": final,
                    "subsequent_growth_bytes": final - baseline,
                    "phases": {},
                }
                report["observations"] = observations
                print(f"Mode: {args.mode}; completed iterations: {args.iterations}")
                print(f"Host RSS after first / last copy: {baseline:,} / {final:,} bytes")
                print(
                    f"Growth over {args.iterations - 1} subsequent copies: {final - baseline:,} bytes"
                )
                for phase in ("bind", "verify", "process"):
                    delta = sum(
                        int(rows[i][2]) - int(rows[i - 1][2])
                        for i in range(3, len(rows))
                        if rows[i][1] == phase
                    )
                    seconds = sum(float(row[3]) for row in rows if row[1] == phase)
                    observations["phases"][phase] = {
                        "rss_delta_bytes_after_first_copy": delta,
                        "total_seconds": seconds,
                    }
                    print(f"  {phase:7}: RSS delta {delta:>12,} bytes; total {seconds:.6f} s")
                print(
                    "Readback and release: PASS. RSS does not establish a leak or physical occupancy."
                )
            else:
                print("\n".join(lines))
            # A clean native exit is insufficient if its observations are incomplete.
            write_json(output / "summary.json", report)
            print(f"Evidence: {output / 'summary.json'}")
            return 0 if passed else 1
    except (OSError, ValueError, RuntimeError) as error:
        write_json(output / "failure.json", {"error": str(error)})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
