#!/usr/bin/env python3
"""Run CPU reference inference using only the verified local Qwen3.5-0.8B checkpoint."""

import argparse
import json
import os
import sys
import traceback
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
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.max_new_tokens <= 64:
        parser.error("threads must be positive; max-new-tokens must be 1..64")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        from specferry.reference.runner import run

        return run(args.model.resolve(), args.output.resolve(), args.threads, args.max_new_tokens)
    except Exception as error:
        (args.output / "reference-summary.json").write_text(
            json.dumps(
                {"status": "failed", "error": str(error), "hardware_execution": False}, indent=2
            )
            + "\n"
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
