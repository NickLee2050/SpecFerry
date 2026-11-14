#!/usr/bin/env python3
"""Validate cached Qwen generation with the accepted conservative defaults."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from tools.deploy.execute.np101.device import write_json
from tools.deploy.execute.np101.qwen3_5.request import prepare_reference, prepare_request
from tools.deploy.export.checkpoint import sha256
from tools.validate.compiled import run_case
from tools.validate.qwen3_5.generation import evaluate_generation

ROOT = Path(__file__).resolve().parents[5]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with only the number.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument("--compiled-mixers", type=Path, required=True)
    parser.add_argument("--head-backend", choices=("matmul", "shared-fcl"), default="matmul")
    parser.add_argument("--state-commit", choices=("per-layer", "combined"), default="per-layer")
    parser.add_argument("--capacity", type=int, choices=(64, 128, 512), default=64)
    parser.add_argument("--embedding-lookup", choices=("gather", "gather-nd"), default="gather-nd")
    parser.add_argument("--matmul-feed-forward-layer", type=int, choices=range(24), default=9)
    parser.add_argument("--token-readback", choices=("sdk", "mapped"), default="mapped")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sudo-read-only", action="store_true")
    parser.add_argument(
        "--board-power-cycled",
        action="store_true",
        help="confirm board power cycle and the agreed Qwen recovery diagnostic",
    )
    parser.add_argument("--sdk-timing", action="store_true", help="record diagnostic SDK calls")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capacity == 512 and not args.prepare_only:
        parser.error("C512 is supported through the SoC deployment entry; use --prepare-only")
    return args


def main():
    args = parse_arguments()
    package = json.loads((args.compiled_mixers / "manifest.json").read_text())
    if (
        package["kind"] != "qwen-delta"
        or package["schema_version"] != 1
        or package["deployment_sha256"] != sha256(args.model / "deployment-manifest.json")
    ):
        raise ValueError("complete Qwen requires its source-bound 18-mixer package")
    sections = [("graphs", "native_manifest_sha256", "compiled-mixers.txt", 18)]
    if not args.prepare_only:
        sections.append(
            ("feed_forward", "feed_forward_manifest_sha256", "compiled-feed-forward.txt", 24)
        )
    for key, digest, native, count in sections:
        graphs = package.get(key, [])
        if len(graphs) != count or sha256(args.compiled_mixers / native) != package.get(digest):
            raise ValueError(f"complete Qwen requires its source-bound {count}-graph {key} package")
        for graph in graphs:
            binary = (args.compiled_mixers / graph["path"]).resolve(strict=True)
            if (
                binary.parent != args.compiled_mixers.resolve()
                or sha256(binary) != graph["nbg_sha256"]
            ):
                raise ValueError("compiled Qwen NBG changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "request"
    request, tokenizer = prepare_request(
        args.model,
        args.checkpoint,
        fixture,
        args.prompt,
        args.capacity,
        args.max_new_tokens,
        args.repeats,
        args.embedding_lookup,
        args.matmul_feed_forward_layer,
        args.token_readback,
    )
    (fixture / "head-backend.txt").write_text(args.head_backend + "\n")
    request["head_backend"] = args.head_backend
    (fixture / "state-commit.txt").write_text(args.state_commit + "\n")
    request["state_commit"] = args.state_commit
    write_json(fixture / "request.json", request)
    reference = prepare_reference(args.checkpoint, request, fixture)
    if args.prepare_only:
        print(
            f"Prepared {fixture}; CPU output: {tokenizer.decode(reference['tokens'], skip_special_tokens=True)!r}"
        )
        return 0

    def accept(case, device):
        result = evaluate_generation(fixture, device)
        result["generated_text"] = [
            tokenizer.decode(run["tokens"], skip_special_tokens=True)
            for run in result.pop("generations")
        ]
        return result

    result = run_case(
        dict(
            binary=ROOT / "build/ep/bin/specferry_qwen_generate",
            arguments=[
                str(args.model.resolve()),
                str(args.compiled_mixers.resolve()),
                str(fixture),
                "{output}",
            ],
        ),
        package["sdk"],
        output / "guarded",
        sudo_read_only=args.sudo_read_only,
        timeout=args.timeout,
        evaluate_result=accept,
        board_power_cycled=args.board_power_cycled,
        sdk_timing=args.sdk_timing,
    )
    write_json(output / "result.json", result)
    print(
        json.dumps(
            {key: result.get(key) for key in ("passed", "generated_text", "performance")},
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
