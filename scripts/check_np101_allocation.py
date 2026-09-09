#!/usr/bin/env python3
"""Check weight/state tensor allocation without claiming full model memory fit."""

import argparse
import json
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
    args = parser.parse_args()
    if args.timeout < 1 or args.output.exists():
        parser.error("positive timeout and a new output directory are required")
    try:
        verify_export(args.model)
        output = args.output.resolve()
        with device_lock(ROOT / ".cache/runs"):
            result = run_device(
                args.binary,
                [str(args.model.resolve()), str(output / "allocation.json")],
                output,
                args.sdk_lib,
                args.timeout,
            )
        print(
            json.dumps(
                {
                    "returncode": result["returncode"],
                    "device_recovery_required": result["device_recovery_required"],
                    "output": str(output),
                    "model_memory_fit_verified": False,
                }
            )
        )
        return int(result["returncode"] != 0 or result["device_recovery_required"])
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"allocation check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
