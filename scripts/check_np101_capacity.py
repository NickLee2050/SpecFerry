#!/usr/bin/env python3
"""Probe retained FP16/FP32 allocation capacity and optional byte integrity separately."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.allocation import (
    FLOAT_DTYPES,
    MIB,
    SEGMENT_LIMIT_MIB,
    STORAGE_MODES,
    add_run_arguments,
    capacity_allocation_complete,
    capacity_complete,
    capacity_scan_complete,
    run_allocation,
)
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, ROOT / "build/tests/np101_capacity_check", 600)
    parser.add_argument("--storage", choices=STORAGE_MODES, default="constant")
    parser.add_argument("--dtype", choices=FLOAT_DTYPES, default="F16")
    parser.add_argument("--target-mib", type=int, default=64)
    parser.add_argument(
        "--readback",
        choices=("first", "all", "none"),
        default="first",
        help="stop at first mismatch (default), scan all blocks, or skip readback",
    )
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.target_mib <= SEGMENT_LIMIT_MIB or args.timeout < 1:
        parser.error("use a fresh output directory, 1..1024 MiB and a positive timeout")
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        arguments = [str(output / "capacity.json"), args.storage, args.dtype, str(args.target_mib)]
        if args.readback != "first":
            arguments.append(args.readback)
        evidence = run_allocation(
            args.binary,
            arguments,
            output,
            args.sdk_lib,
            args.timeout,
        )
        path = output / "capacity.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        uploaded = report.get("uploaded_bytes", 0)
        complete = capacity_complete(report, evidence, args.storage, args.dtype, args.target_mib)
        allocated = capacity_allocation_complete(
            report, evidence, args.storage, args.dtype, args.target_mib, args.readback
        )
        scanned = args.readback == "all" and capacity_scan_complete(
            report, evidence, args.storage, args.dtype, args.target_mib, output
        )
        summary = {
            "status": report.get("status") if complete or allocated else "failed",
            "readback_mode": args.readback,
            "retained_payload_mib": uploaded / MIB,
            "retained_payload_gib": uploaded / 1024**3,
            "readback_and_release_passed": complete,
            "allocation_and_release_passed": allocated,
            "full_scan_completed": scanned,
            "allocation_exceeds_one_gib": allocated and uploaded > 1024**3,
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
        if scanned:
            print(
                f"Scanned {report['scanned_blocks']} blocks; "
                f"{report['mismatched_blocks']} damaged blocks, "
                f"{report['mismatched_bytes']} mismatched bytes."
            )
            scan_path = output / "readback-blocks.jsonl"
            if scan_path.is_file():
                for line in scan_path.read_text().splitlines():
                    block = json.loads(line)
                    if block["mismatched_bytes"]:
                        print(
                            f"  Block {block['block_index']}: "
                            f"{block['mismatched_bytes']} bytes differ; "
                            f"first ranges {block['byte_ranges'][:16]} (zero-based, end-exclusive)"
                        )
        print(f"Evidence: {output / 'summary.json'}")
        # A clean SDK rejection is an observed bound, not physical exhaustion.
        # A completed scan with corrupt bytes remains a nonzero integrity result.
        return 0 if complete or (args.readback == "none" and allocated) else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
