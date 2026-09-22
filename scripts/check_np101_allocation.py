#!/usr/bin/env python3
"""Check retained model weight/state bytes without claiming full model memory fit."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.qwen3_5.export import verify_export
from specferry.models.qwen3_5.memory import write_allocation_states
from specferry.validation.allocation import (
    STORAGE_MODES,
    add_run_arguments,
    run_allocation,
    weights_complete,
)
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser, ROOT / "build/tests/np101_weight_allocation_check", 300)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
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
        verify_export(args.model)
        output.mkdir(parents=True)
        states = output / "allocation-states.txt"
        write_allocation_states(states)
        evidence = run_allocation(
            args.binary,
            [
                str(args.model.resolve()),
                str(output / "allocation.json"),
                "--state-spec",
                str(states),
                "--weight-storage",
                args.weight_storage,
            ],
            output,
            args.sdk_lib,
            args.timeout,
        )
        path = output / "allocation.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        complete = weights_complete(report, evidence, args.weight_storage)
        write_json(
            output / "summary.json",
            {
                "status": "allocation_pass" if complete else "failed",
                "readback_and_release_passed": complete,
                "allocation": report,
                "evidence": evidence,
                "model_memory_fit_verified": False,
            },
        )
        print(
            json.dumps(
                {
                    "returncode": evidence["returncode"],
                    "device_recovery_required": evidence["device_recovery_required"],
                    "output": str(output),
                    "weight_storage": args.weight_storage,
                    "allocation_and_readback_complete": complete,
                    "model_memory_fit_verified": False,
                }
            )
        )
        return int(not complete)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if output.is_dir():
            write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(f"allocation check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
