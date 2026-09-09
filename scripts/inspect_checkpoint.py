#!/usr/bin/env python3
"""Verify all checkpoint weight bytes and inventory the text/vision/MTP tensor groups."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from specferry.reference.checkpoint import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / '.cache/models/Qwen/Qwen3.5-0.8B')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.unlink(missing_ok=True)
    try:
        result = inventory(args.model.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['groups'], indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f'checkpoint inspection failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
