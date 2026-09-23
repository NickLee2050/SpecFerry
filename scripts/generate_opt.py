#!/usr/bin/env python3
"""Tokenize on the host and run the complete resident OPT-350M SDK generation path."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.generation import lifecycle, prepare_request
from specferry.validation.device import device_lock, run_device, snapshot_binary, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--sdk-timing", choices=("off", "summary"), default="off")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="override checkpoint defaults with full-vocabulary sampling at temperature 1",
    )
    parser.add_argument("--seed", type=int, default=0, help="unsigned 32-bit sampling seed")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare-cpu",
        action="store_true",
        help="after native execution, compare token IDs against official CPU generation",
    )
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.max_new_tokens < 2**32 or args.timeout < 1:
        parser.error("use a new output directory, nonnegative token limit and positive timeout")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    if args.sample and args.compare_cpu:
        parser.error(
            "--compare-cpu checks greedy token equality; SDK and CPU random streams differ"
        )
    output = args.output.resolve()
    output.mkdir(parents=True)
    try:
        metadata, tokenizer = prepare_request(
            args.model.resolve(),
            args.checkpoint.resolve(),
            output / "request",
            args.prompt,
            args.capacity,
            sample=args.sample,
            seed=args.seed,
        )
        sampling = metadata["selection"]["mode"] == "multinomial"
        if sampling and args.compare_cpu:
            raise ValueError("CPU token equality requires a greedy checkpoint generation policy")
        print(
            f"Token selection: {metadata['selection']['mode']} ({metadata['selection']['origin']})",
            flush=True,
        )
        arguments = [
            str(args.model.resolve()),
            str(output / "request"),
            str(output / "device"),
            "generate",
            "24",
            str(args.max_new_tokens),
            str(output / "request/tokens.txt"),
        ]
        if sampling:
            arguments.append(str(metadata["selection"]["seed"]))
        with device_lock(ROOT / ".cache/runs"):
            binary = output / "specferry_opt_generate"
            snapshot_binary(ROOT / "build/bin/specferry_opt_generate", binary)
            print("Initializing resident OPT-350M and generating ...", flush=True)
            evidence = run_device(
                binary,
                arguments,
                output / "device",
                Path("/usr/lib/ljmicro"),
                args.timeout,
                shader_header=Path("/usr/inc/CL/cl_viv_vx_ext.h"),
                sdk_timing=args.sdk_timing,
                progress_interval=10,
            )
        path = output / "device/execution.json"
        execution = json.loads(path.read_text()) if path.is_file() else {}
        if not lifecycle(evidence, execution):
            raise RuntimeError(
                f"native execution failed in {execution.get('phase', 'unknown')}; see {output / 'device'}"
            )
        generation = json.loads((output / "device/generation.json").read_text())
        tokens = generation["tokens"]
        seed_uploads = len(tokens) if sampling else 0
        if (
            execution["uploads"] != generation["consumed"] * 26 + seed_uploads
            or execution["upload_bytes"] != generation["consumed"] * 104 + seed_uploads * 16
            or execution["reads"] != len(tokens)
            or execution["read_bytes"] != len(tokens) * 4
            or execution["cache_writes"] != generation["consumed"] * 24
        ):
            raise RuntimeError(
                "generation violated the scalar-control/token-only transfer contract"
            )
        result = {
            "status": "executed",
            "request": metadata,
            "generation": generation,
            "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "execution": execution,
            "evidence": evidence,
            "hardware_execution_proven": False,
            "device_residency_proven": False,
        }
        if args.compare_cpu:
            import torch

            from specferry.models.opt.validation_generation import compare_generation

            torch.set_num_threads(4)
            result["cpu_comparison"] = compare_generation(
                args.checkpoint.resolve(), metadata, generation, args.max_new_tokens
            )
            if not result["cpu_comparison"]["passed"]:
                result["status"] = "reference_mismatch"
        write_json(output / "result.json", result)
        print(result["text"])
        if "cpu_comparison" in result:
            comparison = result["cpu_comparison"]
            print(
                "CPU token comparison: "
                + (
                    "passed"
                    if comparison["passed"]
                    else f"first mismatch at token {comparison['first_mismatch']}"
                )
            )
        print(
            f"Stop: {generation['stop_reason']}; consumed: {generation['consumed']}; result: {output / 'result.json'}"
        )
        print(
            "SDK numerical execution; exclusive NPU execution and physical residency remain unproven."
        )
        return 0 if result["status"] == "executed" else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
