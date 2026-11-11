#!/usr/bin/env python3
"""Prepare, deploy and supervise explicit OPT or Qwen SoC deployments."""

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
    prepare.add_argument("--model", choices=("opt", "qwen"), default="opt")
    prepare.add_argument(
        "--qwen-projections",
        choices=("shared-fcl", "matmul"),
        help="Qwen projection backend, defaults to shared-fcl; matmul is the M1 fallback",
    )
    prepare.add_argument(
        "--qwen-kv-write-mode",
        choices=("copy-graph", "mapped"),
        help="Qwen KV append backend; mapped avoids runtime copy-graph revalidation",
    )
    prepare.add_argument(
        "--qwen-ffn-batch",
        type=int,
        choices=(1, 8, 16),
        help="Qwen compiled FFN batch width; defaults to the validated M1 chain",
    )
    prepare.add_argument(
        "--qwen-decoder-layout",
        choices=("separate", "fused-attention-norm"),
        help="Qwen batched FFN layout; optionally fuse attention input normalization",
    )
    prepare.add_argument("--binary", type=Path)
    prepare.add_argument(
        "--qwen-state-commit",
        choices=("combined", "ffn"),
        help="Qwen state copies: combined decode graph (default) or batched FFN graph",
    )
    prepare.add_argument("--deployment", type=Path)
    for name in ("package", "request", "sdk", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument(
        "--prefill-package",
        type=Path,
        help="source-bound compiled attention package for M16 prefill",
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
                "--timeout",
                type=int,
                default=300,
                help="post-ready execution seconds: maximum 600 for Qwen, 300 for OPT",
            )
            command.add_argument(
                "--prepare-timeout",
                type=int,
                default=240,
                help="model preparation seconds, maximum 240",
            )
    args = parser.parse_args()
    if args.command == "prepare":
        if args.model == "qwen":
            from specferry.platforms.soc.qwen_deployment import prepare_bundle as prepare_qwen

            if args.prefill_package is not None:
                parser.error("Qwen currently uses the existing single-token chain")
            result = prepare_qwen(
                args.binary or ROOT / "build/soc/bin/specferry_qwen_generate",
                args.deployment or ROOT / ".cache/np101/Qwen3.5-0.8B",
                args.package,
                args.request,
                args.sdk,
                args.output,
                projections=args.qwen_projections or "shared-fcl",
                kv_write_mode=args.qwen_kv_write_mode or "copy-graph",
                ffn_batch_tokens=args.qwen_ffn_batch or 1,
                decoder_layout=args.qwen_decoder_layout or "separate",
                state_commit=args.qwen_state_commit or "combined",
            )
        else:
            if any(
                value is not None
                for value in (
                    args.qwen_projections,
                    args.qwen_kv_write_mode,
                    args.qwen_ffn_batch,
                    args.qwen_decoder_layout,
                    args.qwen_state_commit,
                )
            ):
                parser.error("Qwen execution options require --model qwen")
            result = prepare_bundle(
                args.binary or ROOT / "build/soc/bin/specferry_opt_graph_generate",
                args.deployment or ROOT / ".cache/np101/opt-350m",
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
            result = run(
                args.bundle,
                args.host,
                destination,
                args.output,
                args.timeout,
                diagnostic,
                preparation_timeout=args.prepare_timeout,
            )
    print(json.dumps(result, indent=2))
    return int(result.get("passed") is False)


if __name__ == "__main__":
    raise SystemExit(main())
