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
from specferry.validation.device import write_json
from specferry.validation.runner import add_run_arguments, run_check

ROOT = Path(__file__).resolve().parents[1]
PREFIX_CASES = ("prefix-split", "prefix-split-repeat")


def prepare(args, fixture):
    """Model adapters own their reference shapes and expected output files."""
    if args.case == "selection":
        from specferry.models.opt.validation_generation import prepare_selection

        return prepare_selection(fixture)
    if args.case in ("io", "teacher", *PREFIX_CASES):
        from specferry.models.opt.validation_generation import prepare

        return prepare(
            args.model,
            args.checkpoint,
            fixture,
            "prefix" if args.case in PREFIX_CASES else args.case,
            args.layer_count if args.case != "io" else 24,
            args.capacity,
            args.steps,
        )
    if args.case == "opt":
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=("selection", "io", "teacher", "opt", "qwen", *PREFIX_CASES),
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
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Torch intra-operation threads for CPU reference preparation (default: 4)",
    )
    parser.add_argument(
        "--readback",
        choices=("outputs", "outputs-kv", "boundaries"),
        help="split/repeat diagnostics: final outputs by default; finer checks are explicit",
    )
    parser.add_argument(
        "--boundary-layers",
        type=int,
        nargs="+",
        help="strictly increasing 1-based decoder boundaries for --readback boundaries",
    )
    parser.add_argument(
        "--decoder-group-size",
        type=int,
        choices=(1, 2),
        help="decoder layers per execution graph for split/repeat diagnostics (default: 1)",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "merged-kv",
            "merged-kv-local-const",
            "merged-kv-local-mutable",
        ),
        help="split/repeat implementation variant (default: merged-kv)",
    )
    args = parser.parse_args(argv)
    split_readback = args.case in PREFIX_CASES
    if args.decoder_group_size is not None and not split_readback:
        parser.error("--decoder-group-size requires prefix-split or prefix-split-repeat")
    args.decoder_group_size = args.decoder_group_size or 1
    if args.variant is not None and not split_readback:
        parser.error("--variant requires prefix-split or prefix-split-repeat")
    args.variant = args.variant or "merged-kv"
    if (args.readback is not None or args.boundary_layers is not None) and not split_readback:
        parser.error("readback controls require prefix-split or prefix-split-repeat")
    if args.readback == "boundaries":
        boundaries = args.boundary_layers
        if (
            not boundaries
            or boundaries != sorted(set(boundaries))
            or boundaries[0] < 1
            or boundaries[-1] > args.layer_count
        ):
            parser.error("boundaries require increasing unique layer numbers in 1..--layer-count")
    elif args.boundary_layers is not None:
        parser.error("--boundary-layers requires --readback boundaries")
    if split_readback:
        args.readback = args.readback or "outputs"
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
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
    if args.fixture and split_readback:
        source = fixture
        fixture = output / "fixture"
        fixture.mkdir()
        for path in source.iterdir():
            if path.name != "run-options.txt":
                (fixture / path.name).symlink_to(path, target_is_directory=path.is_dir())
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
    if args.case in PREFIX_CASES:
        from specferry.models.opt.graph_validation import validate_prefix_fixture

        metadata = validate_prefix_fixture(args.model, fixture, args.layer_count)
    if split_readback:
        from specferry.models.opt.graph_validation import prefix_readback_scope

        readback_request = {
            **prefix_readback_scope(
                metadata,
                args.readback,
                args.boundary_layers or [],
                repeat=args.case == "prefix-split-repeat",
                decoder_group_size=args.decoder_group_size,
                variant=args.variant,
            ),
            "teacher_steps": metadata["count"],
            "automatic_readback_escalation": False,
            "full_model_lifecycle_accepted": False,
        }
        write_json(output / "readback-request.json", readback_request)
        options = {
            "layers": args.layer_count,
            "variant": args.variant,
            "decoder_group_size": args.decoder_group_size,
            "readback": args.readback,
            "repeat": int(args.case == "prefix-split-repeat"),
        }
        if args.boundary_layers:
            options["boundary_layers"] = ",".join(map(str, args.boundary_layers))
        (fixture / "run-options.txt").write_text(
            "".join(f"{key} {value}\n" for key, value in options.items())
        )
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
    else:
        binary = ROOT / "build/tests/np101_graph_pipeline_check"
    if split_readback:
        from specferry.models.opt.graph_validation import evaluate_prefix_readback

        def evaluate(fixture, device, metadata, evidence):
            return evaluate_prefix_readback(
                fixture,
                device,
                metadata,
                evidence,
                args.readback,
                args.boundary_layers or [],
                repeat=args.case == "prefix-split-repeat",
                decoder_group_size=args.decoder_group_size,
                variant=args.variant,
            )

    returncode = run_check(
        binary,
        arguments,
        output,
        args,
        evaluate=(lambda evidence: evaluate(fixture, device, metadata, evidence))
        if evaluate is not None
        else None,
        timeout=args.timeout
        if args.timeout is not None
        else (360 if args.case in PREFIX_CASES else 120),
    )
    if split_readback:
        # Preserve the requested scope even when native execution fails before its
        # numerical evaluator can run. Partial observations are not acceptance.
        result_path = output / "result.json"
        result = json.loads(result_path.read_text())
        result["validation_request"] = readback_request
        observations = {}
        for path in sorted(device.glob("**/validation.json")):
            try:
                observations[str(path.relative_to(device))] = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                observations[str(path.relative_to(device))] = {"read_error": str(error)}
        result["native_readback_observations"] = observations
        write_json(result_path, result)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
