#!/usr/bin/env python3
"""Generate text or measure resident OPT inference; CPU comparison is independent."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.models.opt.generation import lifecycle, prepare_request
from specferry.validation.device import device_lock, run_device, write_json
from specferry.validation.generation_metrics import summarize
from specferry.validation.performance import read_summary, summarize_request, summarize_runs

ROOT = Path(__file__).resolve().parents[1]


def summarize_graph_metrics(
    device, runs, request, metadata=None, *, benchmark=False, expected=None
):
    """Group measured graph-call spans by original input, independently of wall metrics."""
    rows = read_summary(device / "sdk-summary.tsv")
    grouped = {}
    for ordinal, run in enumerate(runs):
        if type(run.get("consumed")) is int and run["consumed"] == 0:
            continue
        index = run["request_index"] if benchmark else (ordinal if metadata else 0)
        name = (
            f"measured.{index}.{run['iteration']}"
            if benchmark
            else f"measured.{index if metadata else ordinal}"
        )
        reference = metadata["requests"][index] if metadata else None
        sample = summarize_request(
            rows,
            name,
            run,
            reference["prompt_length"] if reference else len(request["tokens"]),
            reference["tokens"] if reference else expected,
        )
        grouped.setdefault(index, []).append(sample)
    return {f"request.{index}": summarize_runs(samples) for index, samples in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--prompt-tokens", type=int, help="take this many tokens of natural source text"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--numerical-diagnostic",
        action="store_true",
        help="explicitly authorized released numerical-failure diagnosis",
    )
    parser.add_argument("--projection-backend", choices=("fcl", "nbg-qkv", "nbg-output", "nbg"))
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--sdk-timing", action="store_true", help="trace individual SDK calls")
    timing.add_argument(
        "--sdk-summary", action="store_true", help="aggregate SDK timings in memory"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="graph: warm and repeat each resident request"
    )
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("component", "graph"), default="component")
    parser.add_argument("--graph-layers", type=int, help="diagnostic truncated OPT layer count")
    parser.add_argument(
        "--cache-readback",
        choices=("none", "logits", "kv"),
        help="graph diagnostics: none (default), final logits, or selected KV",
    )
    parser.add_argument(
        "--readback-transport",
        choices=("sdk", "mapped"),
        help="explicit graph diagnostic transfer path; mapped requires board calibration",
    )
    parser.add_argument(
        "--cache-layer-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help="inclusive 1-based decoder layer range; required with --cache-readback kv",
    )
    parser.add_argument(
        "--request-tokens-file",
        type=Path,
        help="graph acceptance: JSON list of token-ID lists, sharing one resident model",
    )
    parser.add_argument(
        "--lifecycle-check",
        action="store_true",
        help="graph acceptance: build and release two fresh SDK Context instances",
    )
    parser.add_argument("--prefill-head", choices=("every-token", "last-token"))
    parser.add_argument(
        "--variant",
        choices=(
            "merged-kv",
            "merged-kv-fc1-split2",
        ),
        help="explicit partitioned graph diagnostic variant (default: merged-kv)",
    )
    parser.add_argument("--block", type=int, choices=(1, 4, 8, 16), default=1)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument(
        "--compiled-ffn",
        type=Path,
        help="explicit source-bound M1 FFN finish package; graph backend only",
    )
    parser.add_argument(
        "--compiled-model",
        type=Path,
        help="explicit source-bound fullgraph or decoder C64/C128 package",
    )
    parser.add_argument(
        "--sudo-read-only",
        action="store_true",
        help="authorized journalctl checks for the compiled candidate",
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    compiled_package = None
    if args.compiled_model is not None:
        if (
            args.compiled_ffn
            or args.backend != "graph"
            or args.block != 1
            or args.capacity not in (64, 128)
            or args.variant
            or args.graph_layers
            or args.lifecycle_check
            or args.projection_backend
            or args.sdk_timing
            or args.sdk_summary
        ):
            parser.error(
                "compiled model requires full C64/C128 M1 graph mode without another implementation override"
            )
        from specferry.models.opt.compiled_model import load_model_package

        compiled_package = load_model_package(
            args.compiled_model, args.model.resolve(), capacity=args.capacity
        )
        args.variant = (
            "compiled-decoder" if compiled_package["kind"] == "decoders" else "compiled-fullgraph"
        )
        if args.variant == "compiled-fullgraph" and args.prefill_head not in (None, "every-token"):
            parser.error("fullgraph contains its head and uses every-token policy")
    if args.compiled_ffn is not None:
        if (
            args.backend != "graph"
            or args.capacity not in (64, 128, 256)
            or (args.capacity == 256 and args.block != 16)
            or args.block not in (1, 16)
            or args.graph_layers not in (None, 24)
            or args.lifecycle_check
            or args.variant is not None
        ):
            parser.error(
                "compiled FFN requires full OPT C64/C128 graph mode or C256 block16, without variant or lifecycle overrides"
            )
        if args.sdk_timing or args.sdk_summary:
            parser.error("compiled candidate entry currently uses untraced acceptance")
        from specferry.models.opt.compiled import load_ffn_package

        compiled_package = load_ffn_package(
            args.compiled_ffn, args.model.resolve(), capacity=args.capacity
        )
        if args.projection_backend is None and "projections" in compiled_package:
            args.projection_backend = "nbg"
        if args.capacity == 256 and args.projection_backend != "nbg":
            parser.error("C256 prefill requires the packaged NBG projections")
        if args.block == 16 and (
            args.capacity not in (64, 256)
            or "prefill" not in compiled_package
            or args.capacity
            not in compiled_package.get("prefill", {}).get("supported_capacities", [64])
            or args.projection_backend not in (None, "fcl", "nbg")
        ):
            parser.error(
                "block16 requires its declared prefill capacity and M1 FCL or NBG projections"
            )
        args.variant = "compiled-ffn-finish"
        if args.projection_backend not in (None, "fcl") and "projections" not in compiled_package:
            parser.error("NBG projection selection requires packaged projection artifacts")
    elif not compiled_package and (
        args.sudo_read_only or args.projection_backend or args.numerical_diagnostic
    ):
        parser.error("compiled candidate options require --compiled-ffn or --compiled-model")
    if args.backend != "graph" and any(
        option is not None
        for option in (
            args.prefill_head,
            args.variant,
            args.cache_readback,
            args.readback_transport,
            args.cache_layer_range,
        )
    ):
        parser.error("graph variant, head and readback require --backend graph")
    if args.backend != "graph" and (args.request_tokens_file or args.lifecycle_check):
        parser.error("request matrix and lifecycle check require --backend graph")
    if args.benchmark and args.backend != "graph":
        parser.error("--benchmark requires --backend graph; component already supports repetitions")
    if args.benchmark and args.lifecycle_check:
        parser.error("benchmark and lifecycle diagnostics are mutually exclusive")
    args.prefill_head = args.prefill_head or "every-token"
    args.variant = args.variant or "merged-kv"
    args.cache_readback = args.cache_readback or "none"
    args.readback_transport = args.readback_transport or "sdk"
    if args.readback_transport == "mapped" and args.cache_readback == "none":
        parser.error("mapped transport requires logits or KV diagnostic readback")
    if args.benchmark and args.cache_readback != "none":
        parser.error("benchmark requires --cache-readback none")
    if args.lifecycle_check and args.cache_readback != "logits":
        parser.error("--lifecycle-check requires --cache-readback logits for exact rebuild checks")
    if (args.cache_readback == "kv") != (args.cache_layer_range is not None):
        parser.error("--cache-readback kv requires --cache-layer-range FIRST LAST")
    if args.cache_layer_range is not None and not (
        1 <= args.cache_layer_range[0] <= args.cache_layer_range[1]
        and (args.graph_layers is None or args.cache_layer_range[1] <= args.graph_layers)
    ):
        parser.error("invalid cache layer range")
    if (
        not 0 <= args.max_new_tokens < 2**32
        or not 0 <= args.warmups <= 5
        or not 1 <= args.repeats <= 20
    ):
        parser.error("invalid generation count, warmups or repetitions")
    if args.backend == "graph":
        if args.sample or not 1 <= args.max_new_tokens <= 32:
            parser.error("graph generation requires 1..32 predictions without sampling")
        if not args.benchmark and (args.warmups or args.repeats != 1):
            parser.error("graph warmups/repeats require --benchmark")
    if args.graph_layers is not None and (args.backend != "graph" or args.graph_layers <= 0):
        parser.error("--graph-layers requires a positive count and partitioned graph mode")
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
        layers=args.graph_layers,
    )
    sampling = request["selection"]["mode"] == "multinomial"
    if sampling and (args.compare_cpu or args.backend == "graph"):
        raise ValueError("this validation requires greedy checkpoint defaults")
    request_tokens = (
        json.loads(args.request_tokens_file.read_text()) if args.request_tokens_file else None
    )
    if request_tokens is not None:
        request["request_tokens"] = request_tokens
        request["request_tokens_source"] = str(args.request_tokens_file.resolve())
    elif compiled_package:
        request_tokens = [request["tokens"]]
    if compiled_package:
        (
            fixture
            / ("compiled-model-package.txt" if args.compiled_model else "compiled-ffn-package.txt")
        ).write_text(
            json.dumps(
                str((args.compiled_model or args.compiled_ffn).resolve()), ensure_ascii=False
            )
            + "\n"
        )
    expected, metadata = None, None
    if args.backend == "graph":
        from specferry.models.opt.graph_validation import prepare

        if args.capacity % args.block:
            raise ValueError("capacity must be divisible by prefill block")
        metadata = prepare(
            args.checkpoint,
            request,
            fixture,
            args.max_new_tokens,
            layers=args.graph_layers,
            prefill_block=args.block,
            cache_readback=args.cache_readback,
            cache_layer_range=args.cache_layer_range,
            request_tokens=request_tokens,
        )
        binary = ROOT / "build/ep/bin/specferry_opt_graph_generate"
    else:
        if args.compare_cpu:
            from specferry.models.opt.validation_generation import reference_generation

            expected = reference_generation(args.checkpoint, request, args.max_new_tokens)["tokens"]
        binary = ROOT / "build/ep/bin/specferry_opt_generate"
    settings = {
        "max_new_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    if args.projection_backend:
        settings["projection_backend"] = args.projection_backend
    if args.backend == "graph":
        settings.update(
            block=args.block,
            variant=args.variant,
            prefill_head=args.prefill_head,
            readback=args.cache_readback,
            readback_transport=args.readback_transport,
            benchmark=int(args.benchmark),
            lifecycle=int(args.lifecycle_check),
        )
        if args.cache_layer_range is not None:
            settings.update(
                cache_first_layer=args.cache_layer_range[0],
                cache_last_layer=args.cache_layer_range[1],
            )
    elif sampling:
        settings["seed"] = request["selection"]["seed"]
    (fixture / "run-options.txt").write_text(
        "".join(f"{key} {value}\n" for key, value in settings.items())
    )
    arguments = [args.model.resolve(), fixture, device]
    if args.prepare_only:
        print(f"Prepared {fixture}")
        return 0
    if compiled_package:
        from specferry.models.opt.graph_validation import evaluate, evaluate_benchmark
        from specferry.validation.compiled import run_case

        device = output / "guarded/device"

        def accept(case, observed):
            if args.benchmark:
                checks, _ = evaluate_benchmark(
                    observed,
                    metadata,
                    args.block,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    variant=args.variant,
                    decoder_group_size=1,
                    prefill_head_policy=args.prefill_head,
                )
            else:
                checks, _ = evaluate(
                    fixture,
                    observed,
                    metadata,
                    args.block,
                    variant=args.variant,
                    decoder_group_size=1,
                    prefill_head_policy=args.prefill_head,
                    cache_readback=args.cache_readback,
                    cache_layer_range=args.cache_layer_range,
                    readback_transport=args.readback_transport,
                )
            native = json.loads((observed / "run.json").read_text())
            checks["head_backend"] = native.get("head_backend") == (
                "fullgraph-nbg"
                if args.variant == "compiled-fullgraph"
                else "matmul"
                if args.variant == "compiled-decoder"
                else "nbg"
                if "head" in compiled_package
                else "fcl"
            )
            checks["release_budget"] = native.get("released") is True and all(
                native.get(f"{segment}_live_bytes") == 0
                and type(native.get(f"{segment}_peak_bytes")) is int
                and 0 <= native[f"{segment}_peak_bytes"] <= 2**30
                for segment in ("const", "nonconst")
            )
            return dict(passed=all(checks.values()), checks=checks)

        guarded = run_case(
            dict(binary=binary, arguments=[str(args.model.resolve()), str(fixture), "{output}"]),
            compiled_package["sdk"],
            output / "guarded",
            sudo_read_only=args.sudo_read_only,
            timeout=args.timeout,
            evaluate_result=accept,
            numerical_diagnostic=args.numerical_diagnostic,
        )
        if not guarded["passed"]:
            raise RuntimeError("compiled candidate acceptance failed; inspect guarded/result.json")
        evidence = guarded["execution"]
    else:
        with device_lock(ROOT / ".cache/runs"):
            evidence = run_device(
                binary,
                arguments,
                device,
                Path("/usr/lib/ljmicro"),
                args.timeout,
                sdk_timing=args.sdk_timing,
                sdk_summary=args.sdk_summary,
                verify_trace=ROOT / "build/ep/tests/libnp101_verify_trace.so"
                if args.lifecycle_check
                else None,
                require_compiled_operations=args.lifecycle_check,
            )
    execution = (
        json.loads((device / "execution.json").read_text())
        if (device / "execution.json").is_file()
        else {}
    )
    if not lifecycle(evidence, execution):
        raise RuntimeError(f"native inference failed; see {device / 'sdk.log'}")
    checks = {}
    if metadata and args.benchmark:
        from specferry.models.opt.graph_validation import evaluate_benchmark

        checks, runs = evaluate_benchmark(
            device,
            metadata,
            args.block,
            warmups=args.warmups,
            repeats=args.repeats,
            variant=args.variant,
            decoder_group_size=1,
            prefill_head_policy=args.prefill_head,
        )
    elif metadata:
        from specferry.models.opt.graph_validation import evaluate

        checks, runs = {}, []
        cycle_count = 2 if args.lifecycle_check else 1
        cycles = range(cycle_count)
        for cycle in cycles:
            observed = device / f"cycle.{cycle}" if args.lifecycle_check else device
            cycle_checks, cycle_runs = evaluate(
                fixture,
                observed,
                metadata,
                args.block,
                variant=args.variant,
                decoder_group_size=1,
                prefill_head_policy=args.prefill_head,
                cache_readback=args.cache_readback,
                cache_layer_range=args.cache_layer_range,
                readback_transport=args.readback_transport,
            )
            checks.update({f"cycle.{cycle}.{key}": value for key, value in cycle_checks.items()})
            runs.extend(cycle_runs)
        if args.lifecycle_check:
            from specferry.models.opt.graph_validation import evaluate_rebuild

            checks.update(evaluate_rebuild(device, metadata))

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
    metrics = None
    if passed and not args.sdk_timing:
        metrics = (
            {
                f"request.{index}": summarize(
                    [run for run in runs if run["request_index"] == index]
                )
                for index in range(len(metadata["requests"]))
            }
            if args.benchmark
            else summarize(runs)
        )
    graph_metrics = None
    if args.sdk_summary and passed and not args.lifecycle_check:
        graph_metrics = summarize_graph_metrics(
            device, runs, request, metadata, benchmark=args.benchmark, expected=expected
        )
    texts = [tokenizer.decode(run["tokens"], skip_special_tokens=True) for run in runs]
    result = {
        "status": "executed" if passed else "reference_mismatch",
        "request": request,
        "checks": checks,
        "cpu_checked": bool(checks),
        "runs": runs,
        "texts": texts,
        "metrics": metrics,
        "graph_metrics": graph_metrics,
        "graph_metrics_scope": (
            "Measured outer SDK graph calls with completion waits; separately timed Verify, "
            "Setup, reset and IO are excluded. These spans are not pure NPU compute time. "
            "Lifecycle diagnostics reuse timing labels and do not provide per-cycle metrics."
        ),
        "execution": execution,
        "graph_variant": args.variant if args.backend == "graph" else None,
        "benchmark": args.benchmark,
        "validation_scope": (
            {
                "cache_readback": args.cache_readback,
                "readback_transport": args.readback_transport,
                "tokens_checked": True,
                "kv_checks_requested": args.cache_readback == "kv",
                "cache_layer_range": args.cache_layer_range,
                "logits_checks_requested": args.cache_readback == "logits",
                "full_model_accepted": False,
            }
            if metadata
            else None
        ),
        "evidence": evidence,
        "timing_scope": (
            "host generation wall time includes token IO and dispatch; SDK summary separates "
            "graph calls, reset and IO; initialization and warmups are excluded from metrics"
        ),
    }
    write_json(output / "result.json", result)
    print(
        "Input:",
        request["prompt"]
        if args.request_tokens_file is None
        else f"{len(request_tokens)} requests from {args.request_tokens_file}",
    )
    for text in texts:
        print("Output:", text)
    print("Result:", output / "result.json")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
