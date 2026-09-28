#!/usr/bin/env python3
"""Run one model-independent NP101 operator or cache check in a bounded process."""

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.runner import add_run_arguments, run_check

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    cases = parser.add_subparsers(dest="case", required=True)
    for name in (
        "conv",
        "lookup",
        "projection",
        "sampling",
        "kv",
        "cache-attention",
        "cache-append",
        "fp16-preservation",
    ):
        case = cases.add_parser(name)
        add_run_arguments(case)
        if name == "lookup":
            case.add_argument("--operator", choices=("gather", "embedding"), required=True)
        elif name == "projection":
            case.add_argument("--operator", choices=("matmul-add", "fcl"), required=True)
        elif name == "fp16-preservation":
            case.add_argument("--operator", choices=("add", "select"), required=True)
        elif name == "sampling":
            case.add_argument("--dtype", choices=("F16", "F32", "F16_TO_F32"), default="F16_TO_F32")
            case.add_argument("--classes", type=int, default=8)
            case.add_argument("--samples", type=int, default=4096)
        elif name == "cache-attention":
            case.add_argument("--block", type=int, choices=(1, 4), default=1)
        elif name == "cache-append":
            case.add_argument("--rank", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    if args.case == "sampling" and not (
        8 <= args.classes <= 50272 and args.classes % 8 == 0 and 1 <= args.samples <= 8192
    ):
        parser.error("sampling requires 8..50272 classes (multiple of 8), 1..8192 samples")
    return args


def main():
    args = parse_args()
    output = args.output.resolve()
    device = output / "device"
    targets = {
        "conv": "conv_relu_pool_test",
        "lookup": "lookup_check",
        "projection": "projection_check",
        "sampling": "sampling_check",
        "kv": "kv_cache_check",
        "cache-attention": "graph_cache_check",
        "cache-append": "cache_append_check",
        "fp16-preservation": "fp16_preservation_check",
    }
    binary = ROOT / "build/tests" / ("np101_" + targets[args.case])
    arguments = [device]
    evaluate = None
    if args.case == "conv":
        arguments = []
    elif args.case in ("lookup", "projection", "fp16-preservation"):
        arguments += [args.operator]
    elif args.case == "cache-attention":
        arguments += ["stack" if args.block == 1 else "stack-block"]
    elif args.case == "cache-append":
        arguments += [args.rank]
    elif args.case == "sampling":
        from specferry.validation.sampling import evaluate as evaluate_samples

        arguments += [args.dtype, args.classes, args.samples]
        evaluate = partial(evaluate_samples, device, args.dtype, args.classes, args.samples)
    if args.prepare_only:
        print("Command:", binary, *arguments)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    return run_check(binary, arguments, output, args, evaluate=evaluate)


if __name__ == "__main__":
    raise SystemExit(main())
