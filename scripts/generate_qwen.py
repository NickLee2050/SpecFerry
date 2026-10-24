#!/usr/bin/env python3
"""Validate cached Qwen generation with the accepted conservative defaults."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from specferry.data.checkpoint import sha256
from specferry.models.qwen3_5.generation import prepare_reference, prepare_request
from specferry.models.qwen3_5.precision import TOLERANCES
from specferry.validation.arrays import compare_arrays
from specferry.validation.compiled import run_case
from specferry.validation.device import write_json
from specferry.validation.generation_metrics import summarize

ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with only the number.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument("--compiled-mixers", type=Path, required=True)
    parser.add_argument("--head-backend", choices=("matmul", "shared-fcl"), default="matmul")
    parser.add_argument("--state-commit", choices=("per-layer", "combined"), default="per-layer")
    parser.add_argument("--capacity", type=int, choices=(64, 128), default=64)
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
    return args


def main():
    args = parse_arguments()
    package = json.loads((args.compiled_mixers / "manifest.json").read_text())
    if (
        package["kind"] != "qwen-delta"
        or package["schema_version"] != 1
        or len(package["graphs"]) != 18
        or sha256(args.compiled_mixers / "compiled-mixers.txt") != package["native_manifest_sha256"]
        or package["deployment_sha256"] != sha256(args.model / "deployment-manifest.json")
    ):
        raise ValueError("complete Qwen requires its source-bound 18-mixer package")
    for graph in package["graphs"]:
        binary = (args.compiled_mixers / graph["path"]).resolve(strict=True)
        if binary.parent != args.compiled_mixers.resolve() or sha256(binary) != graph["nbg_sha256"]:
            raise ValueError("compiled Qwen mixer NBG changed")
    if not args.prepare_only:
        if len(package.get("feed_forward", [])) != 24 or sha256(
            args.compiled_mixers / "compiled-feed-forward.txt"
        ) != package.get("feed_forward_manifest_sha256"):
            raise ValueError("complete Qwen requires its source-bound 24-FFN package")
        for graph in package["feed_forward"]:
            binary = (args.compiled_mixers / graph["path"]).resolve(strict=True)
            if (
                binary.parent != args.compiled_mixers.resolve()
                or sha256(binary) != graph["nbg_sha256"]
            ):
                raise ValueError("compiled Qwen feed-forward NBG changed")
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
        generations, checks = [], {}
        for index in range(args.repeats):
            run = json.loads((device / f"measured.{index}.json").read_text())
            generations.append(run)
            checks[f"request.{index}"] = (
                run["tokens"] == reference["tokens"]
                and run["consumed"] == reference["consumed"]
                and run["stop_reason"] == reference["stop_reason"]
            )
            for name, contract in reference["snapshots"].items():
                expected = np.fromfile(
                    fixture / f"expected.{name}.bin", dtype=contract["dtype"]
                ).reshape(contract["shape"])
                actual = np.fromfile(
                    device / f"measured.{index}.{name}.bin", dtype=contract["dtype"]
                )
                if name in ("keys", "values"):
                    full = actual.reshape(1, 2, args.capacity, 256)
                    checks[f"request.{index}.{name}.unwritten_bit_zero"] = not np.any(
                        full[:, :, reference["consumed"] :].copy().view("u1")
                    ).item()
                    actual = full[:, :, : reference["consumed"]]
                else:
                    actual = actual.reshape(expected.shape)
                tolerance = TOLERANCES[
                    "fp32_state_from_fp16_pipeline" if name == "recurrent" else "fp16_pipeline"
                ]
                checks[f"request.{index}.{name}"] = compare_arrays(actual, expected, **tolerance)[
                    "passed"
                ]
                if index:
                    checks[f"request.{index}.{name}.reset"] = (
                        device / f"measured.{index}.{name}.bin"
                    ).read_bytes() == (device / f"measured.0.{name}.bin").read_bytes()
        native = json.loads((device / "execution.json").read_text())
        checks["release"] = (
            native["released"]
            and native["const_live_bytes"] == native["nonconst_live_bytes"] == 0
            and max(native["const_peak_bytes"], native["nonconst_peak_bytes"]) <= 2**30
        )
        return dict(
            passed=all(checks.values()),
            checks=checks,
            performance=summarize(generations) if all(checks.values()) else None,
            generated_text=[
                tokenizer.decode(run["tokens"], skip_special_tokens=True) for run in generations
            ],
        )

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
        audit_verify=True,
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
