#!/usr/bin/env python3
"""Check producer/cache/attention dependencies in a small, checkpoint-free SDK graph."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.device import (
    clean_execution,
    device_lock,
    record_sources,
    run_device,
    snapshot_binary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("separate", "view", "shared", "stack", "stack-block"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.output.exists() or args.timeout <= 0:
        parser.error("use a fresh output directory and positive timeout")
    output = args.output.resolve()
    with device_lock(ROOT / ".cache/runs"):
        output.mkdir(parents=True)
        binary = snapshot_binary(
            ROOT / "build/tests/np101_graph_cache_check", output / "np101_graph_cache_check"
        )
        record_sources(ROOT, binary, output / "sources.json")
        evidence = run_device(
            binary,
            [str(output), args.mode],
            output,
            Path("/usr/lib/ljmicro"),
            args.timeout,
            shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
            sdk_timing="calls",
            progress_interval=10,
        )
    passed = clean_execution(evidence)
    write_json(
        output / "summary.json",
        {
            "status": "numerical_pass" if passed else "failed",
            "mode": args.mode,
            "wall_seconds": evidence["wall_seconds"],
            "device_recovery_required": evidence["device_recovery_required"],
            "hardware_execution_proven": False,
            "production_path_accepted": False,
        },
    )
    if (output / "cache.txt").is_file():
        print((output / "cache.txt").read_text(), end="")
    print(f"Evidence: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
