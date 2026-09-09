#!/usr/bin/env python3
"""Apply safe Ruff lint fixes, sort imports, and format project Python code."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check without modifying files")
    args = parser.parse_args()

    # Run through this interpreter so the pinned Conda package is used.
    command = [sys.executable, "-m", "ruff"]
    lint_options = [] if args.check else ["--fix"]
    format_options = ["--check"] if args.check else []
    lint = subprocess.run([*command, "check", *lint_options, "."], cwd=ROOT)
    formatting = subprocess.run([*command, "format", *format_options, "."], cwd=ROOT)
    return int(lint.returncode != 0 or formatting.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
