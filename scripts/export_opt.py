#!/usr/bin/env python3
"""Export or independently verify the native-FP16 OPT-350M weight pack."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.checkpoint import load_checkpoint
from specferry.models.opt.export import compare_source, export_checkpoint, verify_export

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        if args.verify_only:
            verify_export(args.output)
            weights, _ = load_checkpoint(args.model)
            result = compare_source(weights, args.output)
        else:
            result = export_checkpoint(args.model.resolve(), args.output.resolve(), args.capacity)
        print(result)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"OPT export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
