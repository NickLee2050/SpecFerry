#!/usr/bin/env python3
"""Probe bounded, simultaneously retained FP16/FP32 tensor payloads with full readback."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.allocation import (
    FLOAT_DTYPES,
    MAXIMUM_CAPACITY_MIB,
    MIB,
    STORAGE_MODES,
    add_run_arguments,
    capacity_complete,
    run_allocation,
)
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, ROOT / "build/tests/np101_capacity_check", 600)
    parser.add_argument("--storage", choices=STORAGE_MODES, default="constant")
    parser.add_argument("--dtype", choices=FLOAT_DTYPES, default="F16")
    parser.add_argument("--target-mib", type=int, default=1152)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.target_mib <= MAXIMUM_CAPACITY_MIB or args.timeout < 1:
        parser.error("use a fresh output directory, 1..4096 MiB and a positive timeout")
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        evidence = run_allocation(
            args.binary,
            [str(output / "capacity.json"), args.storage, args.dtype, str(args.target_mib)],
            output,
            args.sdk_lib,
            args.timeout,
        )
        path = output / "capacity.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        uploaded = report.get("uploaded_bytes", 0)
        complete = capacity_complete(report, evidence, args.storage, args.dtype, args.target_mib)
        summary = {
            "status": report.get("status") if complete else "failed",
            "retained_payload_mib": uploaded / MIB,
            "retained_payload_gib": uploaded / 1024**3,
            "readback_and_release_passed": complete,
            "exceeds_one_gib": complete and uploaded > 1024**3,
            "allocation": report,
            "evidence": evidence,
            "physical_memory_limit_proven": False,
        }
        write_json(output / "summary.json", summary)
        print(
            f"{summary['status']}: {uploaded / MIB:g} MiB {args.dtype} {args.storage}; "
            f"full readback/release={complete}"
        )
        print(f"Evidence: {output / 'summary.json'}")
        # A clean SDK rejection is an observed bound, not physical exhaustion.
        return 0 if complete else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
