#!/usr/bin/env python3
"""Validate shared operators with a small synthetic decoder and a second KV layout."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.qwen3_5.component_checks import prepare
from specferry.models.qwen3_5.validation_decoder import evaluate
from specferry.validation.device import device_lock, run_device, snapshot_binary, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    output = args.output.resolve()
    torch.set_num_threads(4)
    metadata = prepare(output)
    if args.prepare_only:
        return 0
    with device_lock(ROOT / ".cache/runs"):
        binary = output / "np101_decoder_check"
        snapshot_binary(ROOT / "build/tests/np101_decoder_check", binary)
        evidence = run_device(
            binary,
            [
                str(output / "weights"),
                str(output / "fixture"),
                str(output / "device"),
                str(metadata["steps"]),
                "0",
            ],
            output / "device",
            Path("/usr/lib/ljmicro"),
            120,
            shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
        )
        result = evaluate(output / "fixture", output / "device", metadata, evidence)
        write_json(output / "components.json", result)
        if result["status"] != "numerical_pass" or evidence["device_recovery_required"]:
            return 1
        cache_binary = output / "np101_kv_cache_check"
        snapshot_binary(ROOT / "build/tests/np101_kv_cache_check", cache_binary)
        cache = run_device(
            cache_binary,
            [str(output / "kv"), "3", "16", "8"],
            output / "kv",
            Path("/usr/lib/ljmicro"),
            120,
        )
        if cache["returncode"] != 0 or cache["device_recovery_required"]:
            return 1
    print(f"Synthetic decoder and alternate KV layout passed: {output}")
    return 0 if args.diagnostic else 2


if __name__ == "__main__":
    raise SystemExit(main())
