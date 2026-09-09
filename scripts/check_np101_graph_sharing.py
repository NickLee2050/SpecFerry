#!/usr/bin/env python3
"""Check cross-graph tensor attachment availability and repeated graph execution."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.device import device_lock, run_device

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build/tests/np101_graph_sharing_check"
    )
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.timeout < 1 or args.output.exists():
        parser.error("positive timeout and a new output directory are required")
    output = args.output.resolve()
    try:
        with device_lock(ROOT / ".cache/runs"):
            result = run_device(
                args.binary,
                [str(output / "graph-sharing.json")],
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
                }
            )
        )
        return int(result["returncode"] != 0 or result["device_recovery_required"])
    except (OSError, ValueError, RuntimeError) as error:
        print(f"graph sharing check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
