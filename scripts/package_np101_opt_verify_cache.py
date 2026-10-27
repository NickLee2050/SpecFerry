#!/usr/bin/env python3
"""Attach validated SoC FFN initialization caches to an existing OPT package."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.verify_cache import package_verify_cache
from specferry.platforms.soc.board_runner import save, sha256, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "deployment", "package", "consumer"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(12, 19), default=12)
    args = parser.parse_args()
    result = package_verify_cache(
        args.source, args.deployment, args.package, args.consumer, layers=args.layers
    )
    bundle = args.package.resolve().parent
    if (bundle / "bundle.json").is_file():
        metadata = json.loads((bundle / "bundle.json").read_text())
        for path in args.package.resolve().rglob("*"):
            if path.is_file():
                metadata["files"][str(path.relative_to(bundle))] = dict(
                    bytes=path.stat().st_size, sha256=sha256(path), executable=False
                )
        metadata["payload_bytes"] = sum(entry["bytes"] for entry in metadata["files"].values())
        metadata["verified_prefill_cache"] = result
        save(bundle / "bundle.json", metadata)
        verify_bundle(bundle, check_sdk=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
