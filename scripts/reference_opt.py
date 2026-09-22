#!/usr/bin/env python3
"""Validate OPT-350M CPU prefill, cached decode and captured decoder boundaries."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.reference import run
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    torch.set_num_threads(4)
    try:
        result = run(args.model.resolve(), args.output.resolve())
        print(f"OPT CPU reference: {result['status']}; {len(result['checks'])} comparisons")
        return int(result["status"] != "passed")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(args.output / "failure.json", {"status": "failed", "error": str(error)})
        print(f"OPT reference failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
