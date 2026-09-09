#!/usr/bin/env python3
"""Validate FP16 convolution, ReLU, and max pooling against a CPU reference."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.device import device_lock, run_device

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build/tests/np101_conv_relu_pool_test"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdk-lib", type=Path, default=Path("/usr/lib/ljmicro"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 1000 or args.timeout < 1:
        parser.error("invalid repeats or timeout")
    if args.output.exists():
        parser.error("use a new output directory to preserve earlier evidence")
    output = args.output.resolve()
    arguments = ["--output", str(output / "conv-relu-pool.json"), "--repeats", str(args.repeats)]
    try:
        with device_lock(ROOT / ".cache/runs"):
            evidence = run_device(
                args.binary,
                arguments,
                output,
                args.sdk_lib,
                args.timeout,
                artifact_prefix="conv-relu-pool",
            )
        print(
            json.dumps(
                {
                    "returncode": evidence["returncode"],
                    "device_recovery_required": evidence["device_recovery_required"],
                    "output": str(output),
                    "hardware_execution_proven": False,
                }
            )
        )
        return int(evidence["returncode"] != 0 or evidence["device_recovery_required"])
    except (OSError, ValueError, RuntimeError) as error:
        print(f"convolution check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
