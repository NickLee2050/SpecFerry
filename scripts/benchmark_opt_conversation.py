#!/usr/bin/env python3
"""Measure one OPT M16 conversation mode in a bounded resident process."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import numpy as np
import torch
from transformers import AutoTokenizer, GenerationConfig

from specferry.data.checkpoint import sha256
from specferry.models.opt.compiled import load_ffn_package
from specferry.models.opt.config import TOLERANCES
from specferry.models.opt.generation import resolve_selection, write_config
from specferry.models.opt.reference import load_model, source_record
from specferry.validation.arrays import compare_arrays
from specferry.validation.compiled import run_case
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]
TURNS = (
    "Facts: Alice lives in Paris. Bob lives in Rome. Alice owns a red bicycle.\n"
    "Question: Based on these facts, where does Alice live?\nAnswer:",
    "\nQuestion: Remember the facts above about Alice and Bob. "
    "What does Alice own, and what color is it?\nAnswer:",
    "\nQuestion: Using the earlier facts, write a short summary about Alice. "
    "Mention both her city and her bicycle.\nAnswer:",
)


def prepare(args):
    directory = args.output / "request"
    directory.mkdir(parents=True)
    manifest = load_ffn_package(args.package, args.model, capacity=256)
    if (
        not manifest.get("head")
        or len(manifest.get("projections", [])) != 24
        or 256 not in manifest.get("prefill", {}).get("supported_capacities", [])
    ):
        raise ValueError("conversation requires the declared C256 NBG/M16/shared-head package")
    torch.set_num_threads(4)
    model, metadata = load_model(args.checkpoint)
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    selection = resolve_selection(args.checkpoint)
    if selection["mode"] != "greedy":
        raise ValueError("conversation benchmark requires the checkpoint greedy policy")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    write_config(directory, metadata["config"], capacity=256)
    references, history, requests = [], [], []
    with torch.inference_mode():
        for index, text in enumerate(TURNS):
            added = tokenizer.encode(text, add_special_tokens=index == 0)
            prompt = history + added
            if len(prompt) + args.max_new_tokens - 1 > 256:
                raise ValueError("conversation exceeds capacity; no truncation is allowed")
            ids = torch.tensor([prompt])
            tokens = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
            )[0, len(prompt) :].tolist()
            trajectory = prompt + tokens[:-1]
            result = model(input_ids=torch.tensor([trajectory]), use_cache=True)
            for part in ("keys", "values"):
                getattr(result.past_key_values.layers[23], part)[0].contiguous().numpy().tofile(
                    directory / f"expected.{index}.{part}.bin"
                )
            references.append(
                dict(
                    turn=index + 1,
                    input_text=text,
                    history_tokens=len(history),
                    new_input_tokens=len(added),
                    prompt_tokens=len(prompt),
                    tokens=tokens,
                    consumed=len(trajectory),
                    output_text=tokenizer.decode(tokens, skip_special_tokens=True),
                )
            )
            requests.append(prompt)
            # Append token IDs rather than retokenizing decoded text across a BPE boundary.
            history = prompt + tokens
    for name, rows in (("requests", requests), ("expected", [r["tokens"] for r in references])):
        (directory / f"{name}.txt").write_text(
            "".join(" ".join(map(str, row)) + "\n" for row in rows)
        )
    (directory / "compiled-ffn-package.txt").write_text(json.dumps(str(args.package)) + "\n")
    options = dict(
        max_new_tokens=args.max_new_tokens,
        block=16,
        variant="compiled-ffn-finish",
        projection_backend="nbg",
        prefill_head="last-token",
        history_mode=args.history_mode,
        readback="kv",
        readback_transport="mapped",
        cache_first_layer=24,
        cache_last_layer=24,
    )
    (directory / "run-options.txt").write_text(
        "".join(f"{key} {value}\n" for key, value in options.items())
    )
    reference = dict(
        capacity=256,
        maximum_new_tokens=args.max_new_tokens,
        history_mode=args.history_mode,
        selection=selection,
        source=source_record(),
        requests=references,
        tokenization="One BOS on first turn; later fragments append exact token IDs",
        package_sha256=sha256(args.package / "manifest.json"),
    )
    write_json(directory / "reference.json", reference)
    return manifest, reference


def evaluate_conversation(directory, reference, device):
    native = json.loads((device / "run.json").read_text())
    checks = dict(
        completed=native["status"] == "executed" and native["released"],
        nbg_m16=native["attention_projection_backend"] == native["head_backend"] == "nbg"
        and native["prefill_block"] == 16
        and native["capacity"] == 256
        and native["history_mode"] == reference["history_mode"],
        unwritten_kv=native["cache_suffix_matches"] and native["cache_suffix_checked_bytes"] > 0,
        closed_rejection=native["closed_checks"] == native["closed_rejections"] == 1,
        budgets=all(
            native[f"{segment}_live_bytes"] == 0 and 0 < native[f"{segment}_peak_bytes"] <= 2**30
            for segment in ("const", "nonconst")
        ),
    )
    runs = []
    for index, expected in enumerate(reference["requests"]):
        name = f"measured.{index}"
        run = json.loads((device / f"{name}.json").read_text())
        checks[f"request.{name}.tokens"] = run["tokens"] == expected["tokens"]
        checks[f"request.{name}.consumed"] = run["consumed"] == expected["consumed"]
        for part in ("keys", "values"):
            actual = np.fromfile(device / f"{name}.layer.23.{part}.bin", dtype="<f2")
            actual = actual.reshape(16, 256, 64)[:, : expected["consumed"]]
            target = np.fromfile(directory / f"expected.{index}.{part}.bin", dtype="<f2")
            target = target.reshape(actual.shape)
            checks[f"request.{name}.{part}"] = compare_arrays(
                actual, target, **TOLERANCES["device"]
            )["passed"]
        reset = run["state_preparation_seconds"]
        runs.append(
            expected
            | run
            | dict(
                turn_ttft_seconds=reset + run["first_token_seconds"],
                turn_seconds=reset + run["total_seconds"],
            )
        )
    count = sum(len(run["tokens"]) for run in runs)
    wall = sum(run["turn_seconds"] for run in runs)
    decode_seconds = sum(sum(run["token_seconds"]) for run in runs)
    decode_count = sum(len(run["token_seconds"]) for run in runs)
    performance = dict(
        turns=runs,
        generated_tokens=count,
        active_seconds=wall,
        mean_ttft_seconds=sum(run["turn_ttft_seconds"] for run in runs) / len(runs),
        output_tokens_per_second=count / wall,
        decode_tokens_per_second=decode_count / decode_seconds if decode_seconds else None,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        native=native,
        performance={reference["history_mode"]: performance},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/np101/opt-350m")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".cache/models/facebook/opt-350m")
    parser.add_argument("--max-new-tokens", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--history-mode", choices=("reuse", "reset"), default="reuse")
    args = parser.parse_args()
    args.package, args.model, args.checkpoint = (
        path.resolve(strict=True) for path in (args.package, args.model, args.checkpoint)
    )
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("output must be a new directory to preserve prior evidence")
    manifest, reference = prepare(args)
    print(json.dumps(reference["requests"], ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    binary = args.output / "specferry_opt_graph_generate"
    shutil.copy2(ROOT / "build/ep/bin/specferry_opt_graph_generate", binary)
    write_json(args.output / "identities.json", dict(binary_sha256=sha256(binary)))
    report = run_case(
        dict(binary=binary, arguments=[str(args.model), str(args.output / "request"), "{output}"]),
        manifest["sdk"],
        args.output / "guarded",
        sudo_read_only=True,
        timeout=300,
        audit_verify=True,
        evaluate_result=lambda case, device: evaluate_conversation(
            args.output / "request", reference, device
        ),
    )
    print(json.dumps(dict(passed=report["passed"], error=report.get("error"))), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
