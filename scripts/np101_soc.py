#!/usr/bin/env python3
"""Prepare, deploy and supervise the explicit OPT fullgraph SoC configuration."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    from specferry.platforms.soc.deployment import prepare_bundle
    from specferry.platforms.soc.remote import collect, deploy, run

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="offline: assemble a relocatable bundle")
    prepare.add_argument(
        "--binary", type=Path, default=ROOT / "build/soc/bin/specferry_opt_graph_generate"
    )
    prepare.add_argument("--deployment", type=Path, default=ROOT / ".cache/np101/opt-350m")
    for name in ("package", "request", "sdk", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument(
        "--prefill-package", type=Path, help="pinned EP M16 FFN package for block16 prefill"
    )
    for name in ("deploy", "run", "collect"):
        command = commands.add_parser(name)
        command.add_argument("bundle", type=Path)
        if name != "collect":
            command.add_argument("--host", default="np101")
            command.add_argument("--remote", help="new directory below /root/specferry-soc")
        if name != "deploy":
            command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument(
                "--diagnostic-baseline",
                type=Path,
                help="explicit authorization and exact old fault/boot/gate/bundle identities",
            )
            command.add_argument(
                "--timeout", type=int, default=180, help="native child seconds, maximum 300"
            )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_bundle(
            args.binary,
            args.deployment,
            args.package,
            args.request,
            args.sdk,
            args.output,
            prefill_package=args.prefill_package,
        )
        print(
            json.dumps(
                {
                    "bundle": str(args.output),
                    "payload_bytes": result["payload_bytes"],
                    "files": len(result["files"]),
                }
            )
        )
        return 0
    if args.command == "collect":
        result = collect(args.bundle, args.output.resolve())
    else:
        destination = args.remote or "/root/specferry-soc/bundles/" + args.bundle.resolve().name
        if args.command == "deploy":
            result = deploy(args.bundle, args.host, destination)
        else:
            diagnostic = (
                json.loads(args.diagnostic_baseline.read_text())
                if args.diagnostic_baseline
                else None
            )
            result = run(args.bundle, args.host, destination, args.output, args.timeout, diagnostic)
    print(json.dumps(result, indent=2))
    return int(result.get("passed") is False)


if __name__ == "__main__":
    raise SystemExit(main())
