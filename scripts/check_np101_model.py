#!/usr/bin/env python3
"""Validate model composition against CPU fixtures; preparation does not open NP101."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.data.checkpoint import sha256
from specferry.validation.runner import add_run_arguments, run_check

ROOT = Path(__file__).resolve().parents[1]


def prepare(args, fixture):
    """Model adapters own their reference shapes and expected output files."""
    if args.case in ("selection", "lookup"):
        from specferry.models.opt.validation_generation import prepare_selection

        return prepare_selection(fixture)
    if args.case in ("io", "teacher", "prefix"):
        from specferry.models.opt.validation_generation import prepare

        return prepare(
            args.model,
            args.checkpoint,
            fixture,
            args.case,
            args.layer_count if args.case in ("teacher", "prefix") else 24,
            args.capacity,
            args.steps,
        )
    if args.case in ("opt", "layer"):
        from specferry.models.opt.validation import prepare

        if args.trace is None:
            raise ValueError("--trace is required for decoder references")
        return prepare(args.model, args.trace, fixture, args.layers, args.steps, args.capacity)
    if args.case == "qwen":
        from specferry.models.qwen3_5.validation_decoder import prepare

        if args.trace is None:
            raise ValueError("--trace is required for Qwen decoder references")
        return prepare(args.model, args.trace, fixture, args.steps, args.layers[0], args.layers)
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", choices=("selection", "io", "teacher", "opt", "qwen", "lookup", "layer", "prefix")
    )
    add_run_arguments(parser)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--fixture", type=Path, help="reuse a prepared reference directory")
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--layer-count", type=int, default=1, help="teacher/prefix layer count")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=16)
    args = parser.parse_args()
    args.model = (
        args.model
        or ROOT / ".cache/np101" / ("Qwen3.5-0.8B" if args.case == "qwen" else "opt-350m")
    ).resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture, device = (
        args.fixture.resolve() if args.fixture else output / "fixture",
        output / "device",
    )
    import torch

    torch.set_num_threads(4)
    metadata = (
        json.loads((fixture / "reference.json").read_text())
        if args.fixture
        else prepare(args, fixture)
    )
    if metadata.get("manifest_sha256") and metadata["manifest_sha256"] != sha256(
        args.model / "deployment-manifest.json"
    ):
        raise ValueError("fixture and deployment manifest differ")
    if args.case == "layer" and (0 not in metadata["layers"] or metadata["steps"] < 2):
        raise ValueError("integrated layer diagnosis requires layer 0 and two reference steps")
    if args.case == "prefix":
        from specferry.models.opt.graph_validation import validate_prefix_fixture

        metadata = validate_prefix_fixture(args.model, fixture, args.layer_count)
    if args.prepare_only:
        print(f"Prepared: {fixture}")
        return 0
    deployment = fixture / "deployment" if (fixture / "deployment").is_dir() else args.model
    arguments = [deployment, fixture, device]
    evaluate = None
    if args.case in ("selection", "io", "teacher"):
        from specferry.models.opt.validation_generation import evaluate

        binary = ROOT / "build/tests/np101_opt_io_check"
        if args.case == "teacher":
            binary = ROOT / "build/tests/np101_generation_check"
            arguments += [metadata["layers"], metadata["count"]]
    elif args.case in ("opt", "qwen"):
        if args.case == "opt":
            from specferry.models.opt.validation import evaluate
        else:
            from specferry.models.qwen3_5.validation_decoder import evaluate
        binary = ROOT / "build/tests/np101_decoder_check"
        arguments = [args.case, *arguments, metadata["steps"]]
    elif args.case in ("lookup", "layer", "prefix"):
        binary = ROOT / "build/tests/np101_graph_pipeline_check"
        arguments += [args.case]
        if args.case == "prefix":
            arguments += [args.layer_count]
    return run_check(
        binary,
        arguments,
        output,
        args,
        evaluate=(lambda evidence: evaluate(fixture, device, metadata, evidence))
        if evaluate is not None
        else None,
        timeout=args.timeout
        if args.timeout is not None
        else (360 if args.case == "prefix" else 120),
    )


if __name__ == "__main__":
    raise SystemExit(main())
