#!/usr/bin/env python3
"""Run CPU reference inference using only the verified local Qwen3.5-0.8B checkpoint."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
# These tools may never resolve or download a second model or optional kernel.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/models/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    from specferry.models.qwen3_5.runner import run

    return run(args.model.resolve(), args.output.resolve(), args.threads)


if __name__ == "__main__":
    raise SystemExit(main())
