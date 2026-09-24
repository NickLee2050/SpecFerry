#!/usr/bin/env python3
"""Generate text or measure resident OPT inference; CPU comparison is independent."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.generation import lifecycle, prepare_request
from specferry.models.opt.metrics import summarize
from specferry.validation.device import device_lock, run_device, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--prompt-tokens", type=int, help="take this many tokens of natural source text"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sdk-timing", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("component", "graph"), default="component")
    parser.add_argument("--block", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (
        not 0 <= args.max_new_tokens < 2**32
        or not 0 <= args.warmups <= 5
        or not 1 <= args.repeats <= 20
    ):
        parser.error("invalid generation count, warmups or repetitions")
    if args.backend == "graph" and (
        args.sample or args.warmups or args.repeats != 1 or not 1 <= args.max_new_tokens <= 32
    ):
        parser.error(
            "experimental graph runs one A/B/A check, 1..32 predictions, without sampling/warmups"
        )
    if args.sample and args.compare_cpu:
        parser.error("CPU token equality is not a sampling test")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture, device = output / "request", output / "device"
    request, tokenizer = prepare_request(
        args.model,
        args.checkpoint,
        fixture,
        args.prompt_file.read_text() if args.prompt_file else args.prompt,
        args.capacity,
        sample=args.sample,
        seed=args.seed,
        prompt_tokens=args.prompt_tokens,
    )
    sampling = request["selection"]["mode"] == "multinomial"
    if sampling and (args.compare_cpu or args.backend == "graph"):
        raise ValueError("this validation requires greedy checkpoint defaults")
    expected, metadata = None, None
    if args.backend == "graph":
        from specferry.models.opt.graph_validation import prepare

        if args.capacity % args.block:
            raise ValueError("capacity must be divisible by prefill block")
        metadata = prepare(args.checkpoint, request, fixture, args.max_new_tokens)
        binary = ROOT / "build/bin/specferry_opt_graph_generate"
        arguments = [
            args.model.resolve(),
            fixture,
            device,
            args.block,
            args.max_new_tokens,
            "prefix",
            fixture / "expected.txt",
        ]
    else:
        if args.compare_cpu:
            from specferry.models.opt.validation_generation import reference_generation

            expected = reference_generation(args.checkpoint, request, args.max_new_tokens)["tokens"]
        binary = ROOT / "build/bin/specferry_opt_generate"
        arguments = [
            args.model.resolve(),
            fixture,
            device,
            args.max_new_tokens,
            args.warmups,
            args.repeats,
        ]
        if sampling:
            arguments += [request["selection"]["seed"]]
    if args.prepare_only:
        print(f"Prepared {fixture}")
        return 0
    with device_lock(ROOT / ".cache/runs"):
        evidence = run_device(
            binary,
            arguments,
            device,
            Path("/usr/lib/ljmicro"),
            args.timeout,
            sdk_timing=args.sdk_timing,
        )
    execution = (
        json.loads((device / "execution.json").read_text())
        if (device / "execution.json").is_file()
        else {}
    )
    if not lifecycle(evidence, execution):
        raise RuntimeError(f"native inference failed; see {device / 'sdk.log'}")
    checks = {}
    if metadata:
        from specferry.models.opt.graph_validation import evaluate

        checks, runs = evaluate(fixture, device, metadata, args.block)
    else:
        runs = [
            json.loads((device / f"measured.{i}.json").read_text()) for i in range(args.repeats)
        ]
        if expected is not None:
            for kind, count in (("warmup", args.warmups), ("measured", args.repeats)):
                for i in range(count):
                    name = f"{kind}.{i}"
                    checks[name] = (
                        json.loads((device / f"{name}.json").read_text())["tokens"] == expected
                    )
    passed = all(checks.values())
    metrics = summarize(runs) if passed and not args.sdk_timing else None
    texts = [tokenizer.decode(run["tokens"], skip_special_tokens=True) for run in runs]
    result = {
        "status": "executed" if passed else "reference_mismatch",
        "request": request,
        "checks": checks,
        "cpu_checked": bool(checks),
        "runs": runs,
        "texts": texts,
        "metrics": metrics,
        "execution": execution,
        "evidence": evidence,
        "timing_scope": "host wall time; excludes initialization, tokenization, warmups and readback",
    }
    write_json(output / "result.json", result)
    print("Input:", request["prompt"])
    for text in texts:
        print("Output:", text)
    print("Metrics:", metrics, "(CPU checked:", bool(checks), ")")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
