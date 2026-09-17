#!/usr/bin/env python3
"""Check weight/state tensor allocation without claiming full model memory fit."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.export.weights import verify_export
from specferry.validation.device import device_lock, run_device

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build/tests/np101_weight_allocation_check"
    )
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--weight-storage",
        choices=("constant", "mutable"),
        default="constant",
        help="mutable creates every weight with is_const=false and explicitly uploads its bytes",
    )
    args = parser.parse_args()
    if args.timeout < 1 or args.output.exists():
        parser.error("positive timeout and a new output directory are required")
    try:
        verify_export(args.model)
        output = args.output.resolve()
        with device_lock(ROOT / ".cache/runs"):
            output.mkdir(parents=True)
            binary = output / "np101_weight_allocation_check"
            shutil.copy2(args.binary.resolve(), binary)
            result = run_device(
                binary,
                [
                    str(args.model.resolve()),
                    str(output / "allocation.json"),
                    "--weight-storage",
                    args.weight_storage,
                ],
                output,
                args.sdk_lib,
                args.timeout,
            )
        report_path = output / "allocation.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        complete = allocation_complete(report, args.weight_storage)
        print(
            json.dumps(
                {
                    "returncode": result["returncode"],
                    "device_recovery_required": result["device_recovery_required"],
                    "output": str(output),
                    "weight_storage": args.weight_storage,
                    "allocation_and_readback_complete": complete,
                    "model_memory_fit_verified": False,
                }
            )
        )
        return int(result["returncode"] != 0 or result["device_recovery_required"] or not complete)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"allocation check failed: {error}", file=sys.stderr)
        return 1


def allocation_complete(report: dict, storage: str) -> bool:
    """Require complete uploads, retained-byte checks, and normal SDK release."""
    expected_bytes = report.get("expected_weight_bytes", 0)
    expected_weights = report.get("expected_weights", 0)
    return (
        report.get("status") == "allocation_pass"
        and report.get("phase") == "complete"
        and report.get("weight_storage") == storage
        and report.get("is_const") is (storage == "constant")
        and expected_weights > 0
        and report.get("completed_weights") == expected_weights
        and expected_bytes > 0
        and report.get("uploaded_weight_bytes") == expected_bytes
        and report.get("verified_weight_bytes") == expected_bytes
        and report.get("state_allocation_complete") is True
    )


if __name__ == "__main__":
    raise SystemExit(main())
