#!/usr/bin/env python3
"""Validate one pinned compiled OPT configuration; no automatic test sweep."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.validation.compiled import load_case, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sudo-read-only", action="store_true")
    parser.add_argument("--audit-verify", action="store_true", help="pinned v105 Verify-only audit")
    parser.add_argument("--timeout", type=int, default=300)
    options = parser.parse_args()
    if options.timeout <= 0:
        parser.error("timeout must be positive")
    case, identity = load_case(options.plan, options.case)
    if options.prepare_only:
        print("Validated artifact/reference plan:", options.case)
        return 0
    report = run_case(
        case,
        identity,
        options.output.resolve(),
        sudo_read_only=options.sudo_read_only,
        timeout=options.timeout,
        audit_verify=options.audit_verify,
    )
    print(json.dumps({k: report.get(k) for k in ("passed", "performance", "error")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
