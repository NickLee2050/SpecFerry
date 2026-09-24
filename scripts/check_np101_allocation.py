#!/usr/bin/env python3
"""Check retained model weight/state bytes without claiming full model memory fit."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.allocation import (
    STORAGE_MODES,
    add_run_arguments,
    check_segment_budget,
    prepare_weight_check,
    run_allocation,
    weights_complete,
)
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, ROOT / "build/tests/np101_weight_allocation_check", 300)
    parser.add_argument(
        "--deployment",
        "--model",
        dest="deployment",
        type=Path,
        required=True,
        help="verified common weight-pack directory; --model is an alias",
    )
    parser.add_argument(
        "--state-spec", type=Path, help="optional explicit state allocation fixture"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="verify inputs without device IO"
    )
    parser.add_argument(
        "--weight-storage",
        choices=STORAGE_MODES,
        default="constant",
        help="mutable creates every weight with is_const=false and explicitly uploads its bytes",
    )
    args = parser.parse_args()
    if args.timeout < 1 or args.output.exists():
        parser.error("positive timeout and a new output directory are required")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True)
        inputs = prepare_weight_check(args.deployment, args.state_spec, output)
        inputs["memory_budget"] = check_segment_budget(
            inputs["expected"]["expected_weight_bytes"],
            inputs["expected"]["expected_state_bytes"],
            args.weight_storage,
        )
        write_json(output / "inputs.json", inputs)
        if args.prepare_only:
            print(f"Verified weight-check inputs: {output / 'inputs.json'}")
            return 0
        arguments = [
            str(args.deployment.resolve()),
            str(output / "allocation.json"),
            "--weight-storage",
            args.weight_storage,
        ]
        if args.state_spec is not None:
            arguments.extend(["--state-spec", str(output / "allocation-states.txt")])
        evidence = run_allocation(
            args.binary,
            arguments,
            output,
            args.sdk_lib,
            args.timeout,
        )
        path = output / "allocation.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        complete = weights_complete(report, evidence, args.weight_storage, inputs["expected"])
        write_json(
            output / "summary.json",
            {
                "status": "allocation_pass" if complete else "failed",
                "readback_and_release_passed": complete,
                "allocation": report,
                "evidence": evidence,
                "model_memory_fit_verified": False,
                "inputs": inputs,
            },
        )
        print(f"Weight/state readback and release: {'passed' if complete else 'failed'}")
        print(f"Evidence: {output / 'summary.json'}")
        return int(not complete)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if output.is_dir():
            write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(f"allocation check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
