#!/usr/bin/env python3
"""Probe DLM operators and state feedback using isolated, bounded SDK processes."""

import argparse
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.export.weights import verify_export
from specferry.validation.capabilities import check_case
from specferry.validation.device import device_lock, write_json
from specferry.validation.operator_cases import catalog
from specferry.validation.reference_cases import reference_catalog

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/tests/np101_op_check")
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--scale", choices=("small", "model", "all"), default="small")
    parser.add_argument("--case", action="append", default=[], help="select a case; repeatable")
    parser.add_argument("--list", action="store_true", help="list available cases without running")
    parser.add_argument(
        "--reference-trace",
        type=Path,
        help="deployment-fp16 layer-0-3-sequential.npz; enables real projection cases",
    )
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument(
        "--prepare-only", action="store_true", help="write fixtures without device IO"
    )
    parser.add_argument("--timeout", type=int, default=300, help="maximum seconds per case")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="return success for numeric passes even when hardware proof is absent",
    )
    args = parser.parse_args()
    cases = catalog()
    if args.reference_trace:
        cases.update(reference_catalog(args.model.resolve(), args.reference_trace.resolve()))
    if args.list:
        for case in cases.values():
            print(f"{case.name:32} {case.scale:8} {case.family}")
        return 0
    if args.output is None or args.timeout < 1:
        parser.error("--output and a positive timeout are required")
    if unknown := set(args.case) - cases.keys():
        parser.error(f"unknown cases: {sorted(unknown)}")
    if args.reference_trace:
        verify_export(args.model)
    selected = (
        [cases[name] for name in args.case]
        if args.case
        else [case for case in cases.values() if args.scale in ("all", case.scale)]
    )
    output = args.output.resolve()
    if output.exists():
        parser.error("use a new output directory to preserve earlier evidence")
    output.mkdir(parents=True)
    # Keep one immutable executable for the suite, even if another terminal rebuilds.
    binary = args.binary.resolve()
    if not args.prepare_only:
        snapshot = output / "np101_op_check"
        shutil.copy2(binary, snapshot)
        binary = snapshot
    results = []
    summary = {
        "status": "running",
        "cases": results,
        "hardware_acceptance": "unverified",
        "full_catalog": len(selected) == len(cases),
        "pending_cases": sorted(set(cases) - {case.name for case in selected}),
    }
    try:
        with nullcontext() if args.prepare_only else device_lock(ROOT / ".cache/runs"):
            for case in selected:
                print(f"Checking {case.name} ...", flush=True)
                try:
                    result = check_case(
                        case,
                        output / case.name,
                        binary,
                        args.sdk_lib,
                        args.timeout,
                        args.prepare_only,
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    result = {"name": case.name, "status": "failed", "error": str(error)}
                results.append(result)
                write_json(output / "op-capabilities.json", summary)
                print(f"  {result['status']}", flush=True)
                if "device_recovery_required" in result.get("blockers", []):
                    summary["error"] = (
                        "SDK run requires device recovery; remaining device cases were not run"
                    )
                    summary["pending_cases"] = sorted(
                        set(cases) - {item["name"] for item in results}
                    )
                    break
    except RuntimeError as error:
        summary["error"] = str(error)
        summary["pending_cases"] = sorted(set(cases) - {item["name"] for item in results})
        print(str(error), file=sys.stderr)
    numeric_pass = len(results) == len(selected) and all(
        item["status"] == "numerical_pass" for item in results
    )
    prepared = len(results) == len(selected) and all(
        item["status"] == "prepared" for item in results
    )
    summary["status"] = "prepared" if args.prepare_only and prepared else "blocked"
    summary["numeric_pass"] = numeric_pass
    write_json(output / "op-capabilities.json", summary)
    print(f"Capability status: {summary['status']}; report: {output / 'op-capabilities.json'}")
    if args.prepare_only:
        return int(not prepared)
    return 0 if args.diagnostic and numeric_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
