#!/usr/bin/env python3
"""Verify every exported text parameter against independent PyTorch conversion."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.export.verification import compare_checkpoint
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/models/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--deployment", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output file to preserve earlier verification")
    try:
        result = compare_checkpoint(args.model, args.deployment)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(f"Verified {result['tensors']} tensors against PyTorch conversion.")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"export verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
