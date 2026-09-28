#!/usr/bin/env python3
"""Run one memory experiment. Allocation success and byte integrity are separate."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from specferry.validation.allocation import (
    MIB,
    capacity_result,
    check_segment_budget,
    prepare_weight_check,
    print_capacity_summary,
    weights_complete,
)
from specferry.validation.device import clean_execution, device_lock, run_device, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = parser.add_subparsers(dest="case", required=True)
    for name in ("capacity", "integrity", "weights", "growth", "accounting"):
        case = cases.add_parser(name)
        case.add_argument("--output", type=Path, required=True)
        case.add_argument("--timeout", type=int, default=120)
        case.add_argument("--prepare-only", action="store_true")
        if name in ("capacity", "integrity"):
            case.add_argument(
                "--probe-limit",
                action="store_true",
                help="explicit capacity probe: allow up to 4096 MiB; stop at SDK rejection",
            )
        if name == "growth":
            case.add_argument("--mode", choices=("fixed", "same", "advance"), default="advance")
            case.add_argument("--iterations", type=int, default=128)
        else:
            case.add_argument("--storage", choices=("constant", "mutable"), default="constant")
            if name == "weights":
                case.add_argument("--deployment", type=Path, required=True)
                case.add_argument("--state-spec", type=Path)
            else:
                case.add_argument("--dtype", choices=("F16", "F32"), default="F16")
                case.add_argument("--mib", type=int, default=8 if name == "accounting" else 64)
    args = parser.parse_args()
    limit_mib = 4096 if getattr(args, "probe_limit", False) else 1024
    if args.timeout < 1 or (
        hasattr(args, "mib")
        and not 1 <= args.mib <= (64 if args.case == "accounting" else limit_mib)
    ):
        parser.error(
            "payload exceeds the case limit (large probes need --probe-limit); timeout must be positive"
        )
    if args.case == "growth" and not 1 <= args.iterations <= 4096:
        parser.error("iterations must be in [1,4096]")
    readback = "none" if args.case == "capacity" else "all"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = None
    if args.case == "weights":
        inputs = prepare_weight_check(args.deployment, args.state_spec, output)
        expected = inputs["expected"]
        check_segment_budget(
            expected["expected_weight_bytes"], expected["expected_state_bytes"], args.storage
        )
        write_json(output / "inputs.json", inputs)
        target = "weight_allocation_check"
        arguments = [
            args.deployment.resolve(),
            output / "allocation.json",
            "--weight-storage",
            args.storage,
        ]
        if args.state_spec:
            arguments += ["--state-spec", output / "allocation-states.txt"]
    elif args.case in ("capacity", "integrity"):
        target = "capacity_check"
        arguments = [output / "capacity.json", args.storage, args.dtype, args.mib, readback]
        if args.probe_limit:
            arguments.append("--probe-limit")
    elif args.case == "growth":
        target = "memory_growth_check"
        arguments = [args.mode, args.iterations]
    else:
        target = "memory_accounting_check"
        arguments = [args.mib, args.storage, args.dtype]
    if args.prepare_only:
        print(f"Prepared {args.case}: {arguments}")
        return 0
    with device_lock(ROOT / ".cache/runs"):
        evidence = run_device(
            ROOT / "build/tests" / ("np101_" + target),
            arguments,
            output,
            Path("/usr/lib/ljmicro"),
            args.timeout,
        )
    passed = clean_execution(evidence)
    result = {"experiment_completed": passed, "evidence": evidence}
    if args.case in ("capacity", "integrity", "weights"):
        path = output / ("allocation.json" if args.case == "weights" else "capacity.json")
        report = json.loads(path.read_text()) if path.is_file() else {}
        if args.case in ("capacity", "integrity"):
            result.update(
                capacity_result(
                    report, evidence, args.mib * MIB, args.storage, args.dtype, readback
                )
            )
            result["experiment_completed"] = result["allocation_and_release_passed"]
            passed = result[
                "allocation_and_release_passed"
                if readback == "none"
                else "readback_and_release_passed"
            ]
            print_capacity_summary(result, report)
        else:
            passed = weights_complete(report, evidence, args.storage, inputs["expected"])
            result["weight_readback_and_release_passed"] = passed
        result["observations"] = report
    else:
        print((output / "sdk.log").read_text())
    result["passed"] = passed
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("evidence", "observations")}, indent=2
        )
    )
    print(f"Details: {output}")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
