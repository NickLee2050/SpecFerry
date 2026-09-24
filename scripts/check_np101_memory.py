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
    weights_complete,
)
from specferry.validation.device import clean_execution, device_lock, run_device, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("capacity", "weights", "growth", "accounting"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--storage", choices=("constant", "mutable"), default="constant")
    parser.add_argument("--dtype", choices=("F16", "F32"), default="F16")
    parser.add_argument("--mib", type=int, default=64)
    parser.add_argument("--readback", choices=("first", "all", "none"), default="all")
    parser.add_argument("--mode", choices=("fixed", "same", "advance"), default="advance")
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--state-spec", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.mib <= (64 if args.case == "accounting" else 1024) or args.timeout < 1:
        parser.error("payload must fit the segment limit; timeout must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = None
    if args.case == "weights":
        if args.deployment is None:
            parser.error("weights requires --deployment")
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
    elif args.case == "capacity":
        target = "capacity_check"
        arguments = [output / "capacity.json", args.storage, args.dtype, args.mib, args.readback]
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
    if args.case in ("capacity", "weights"):
        path = output / ("capacity.json" if args.case == "capacity" else "allocation.json")
        report = json.loads(path.read_text()) if path.is_file() else {}
        if args.case == "capacity":
            result.update(
                capacity_result(
                    report, evidence, args.mib * MIB, args.storage, args.dtype, args.readback
                )
            )
            passed = result[
                "allocation_and_release_passed"
                if args.readback == "none"
                else "readback_and_release_passed"
            ]
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
