#!/usr/bin/env python3
"""Isolate OPT graph inputs, shared weights, and one-layer KV computation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.allocation import SEGMENT_LIMIT_BYTES
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
    parser.add_argument("--mode", choices=("lookup", "layer", "prefix"), required=True)
    parser.add_argument("--layers", type=int, default=1, help="diagnostic prefix layer count")
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, help="defaults: lookup/layer 120s, prefix 360s")
    args = parser.parse_args()
    if args.output.exists() or (args.timeout is not None and args.timeout <= 0) or args.layers < 1:
        parser.error("use a fresh output directory, positive timeout, and positive layer count")
    if args.mode != "prefix" and args.layers != 1:
        parser.error("--layers applies only to a diagnostic prefix")
    deployment = args.deployment.resolve(strict=True)
    fixture = args.fixture.resolve(strict=True)
    output = args.output.resolve()
    with device_lock(ROOT / ".cache/runs"):
        output.mkdir(parents=True)
        binary = snapshot_binary(
            ROOT / "build/tests/np101_graph_pipeline_check", output / "np101_graph_pipeline_check"
        )
        record_sources(ROOT, binary, output / "sources.json")
        evidence = run_device(
            binary,
            [str(deployment), str(fixture), str(output), args.mode, str(args.layers)],
            output,
            Path("/usr/lib/ljmicro"),
            args.timeout or {"lookup": 120, "layer": 120, "prefix": 360}[args.mode],
            shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
            sdk_timing="calls",
            progress_interval=5,
            weight_readback=True,
        )
    execution = {}
    if (output / "execution.json").exists():
        execution = json.loads((output / "execution.json").read_text())
    memory = {}
    if (output / "memory-budget.json").exists():
        memory = json.loads((output / "memory-budget.json").read_text())
    memory_passed = memory.get("segment_limit_bytes") == SEGMENT_LIMIT_BYTES and all(
        memory.get(f"{segment}_live_bytes") == 0
        and 0 <= memory.get(f"{segment}_peak_bytes", -1) <= SEGMENT_LIMIT_BYTES
        for segment in ("const", "nonconst")
    )
    passed = (
        clean_execution(evidence)
        and memory_passed
        and execution.get("phase") == "complete"
        and execution.get("released") is True
        and execution.get("checks", 0) > 0
        and execution.get("failures") == 0
    )
    report = {
        "status": "numerical_pass" if passed else "failed",
        "mode": args.mode,
        "execution": execution,
        "memory_budget": memory,
        "payload_budget_and_release_passed": memory_passed,
        "wall_seconds": evidence["wall_seconds"],
        "device_recovery_required": evidence["device_recovery_required"],
        "hardware_execution_proven": False,
    }
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))
    print(f"Per-stage comparisons: {output / 'checks.tsv'}")
    print(f"Segment payload peaks: {output / 'memory-budget.json'}")
    print(f"Complete shared weight readback: {output / 'weight-readback.tsv'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
