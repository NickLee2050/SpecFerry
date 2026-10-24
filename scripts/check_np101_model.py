#!/usr/bin/env python3
"""Validate model composition against CPU fixtures; preparation does not open NP101."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.data.checkpoint import sha256
from specferry.validation.runner import add_run_arguments, run_check

ROOT = Path(__file__).resolve().parents[1]


def prepare(args, fixture):
    """Model adapters own their reference shapes and expected output files."""
    if args.case == "selection":
        from specferry.models.opt.validation_generation import prepare_selection

        return prepare_selection(fixture)
    if args.case in ("io", "teacher"):
        from specferry.models.opt.validation_generation import prepare

        return prepare(
            args.model,
            args.checkpoint,
            fixture,
            args.case,
            args.layer_count if args.case != "io" else 24,
            args.capacity,
            args.steps,
        )
    if args.case == "opt":
        from specferry.models.opt.validation import prepare

        if args.trace is None:
            raise ValueError("--trace is required for decoder references")
        return prepare(args.model, args.trace, fixture, args.layers, args.steps, args.capacity)
    return {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=("selection", "io", "teacher", "opt"),
    )
    add_run_arguments(parser)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--fixture", type=Path, help="reuse a prepared reference directory")
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--layer-count", type=int, default=1, help="teacher layer count")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Torch intra-operation threads for CPU reference preparation (default: 4)",
    )
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    args.model = (args.model or ROOT / ".cache/np101/opt-350m").resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture, device = (
        args.fixture.resolve() if args.fixture else output / "fixture",
        output / "device",
    )
    import torch

    torch.set_num_threads(args.cpu_threads)
    reference_started = time.perf_counter()
    metadata = (
        json.loads((fixture / "reference.json").read_text())
        if args.fixture
        else prepare(args, fixture)
    )
    # Keep execution settings outside the fixture's checksummed reference contract.
    cpu_settings = {
        "torch_version": torch.__version__,
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "reference_action": "reused" if args.fixture else "prepared",
        "reference_prepare_or_load_seconds": time.perf_counter() - reference_started,
    }
    (output / "cpu-settings.json").write_text(json.dumps(cpu_settings, indent=2) + "\n")
    if metadata.get("manifest_sha256") and metadata["manifest_sha256"] != sha256(
        args.model / "deployment-manifest.json"
    ):
        raise ValueError("fixture and deployment manifest differ")
    if args.prepare_only:
        print(f"Prepared: {fixture}")
        return 0
    deployment = fixture / "deployment" if (fixture / "deployment").is_dir() else args.model
    arguments = [deployment, fixture, device]
    evaluate = None
    if args.case in ("selection", "io", "teacher"):
        from specferry.models.opt.validation_generation import evaluate

        binary = ROOT / "build/ep/tests/np101_opt_io_check"
        if args.case == "teacher":
            binary = ROOT / "build/ep/tests/np101_generation_check"
            arguments += [metadata["layers"], metadata["count"]]
    elif args.case == "opt":
        from specferry.models.opt.validation import evaluate

        binary = ROOT / "build/ep/tests/np101_decoder_check"
        arguments += [metadata["steps"]]
    return run_check(
        binary,
        arguments,
        output,
        args,
        evaluate=(lambda evidence: evaluate(fixture, device, metadata, evidence))
        if evaluate is not None
        else None,
        timeout=args.timeout if args.timeout is not None else 120,
    )


if __name__ == "__main__":
    raise SystemExit(main())
