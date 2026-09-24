"""Bounded correctness gates for the experimental fixed-graph OPT path."""

import json
from pathlib import Path

import numpy as np

from specferry.validation.capabilities import compare_arrays
from specferry.validation.device import clean_execution, fingerprint, write_json
from specferry.validation.timing import summarize_sdk_timing

from .acceptance import elapsed, valid_generation
from .config import TOLERANCES
from .generation import prepare_request, write_config

CASES = {
    "decode": {"block": 1, "capacity": 16, "lengths": [5, 3, 5], "timeout": 360, "previous": None},
    "prefill": {
        "block": 4,
        "capacity": 16,
        "lengths": [9, 5, 9],
        "timeout": 600,
        "previous": "decode",
    },
    "capacity64": {
        "block": 4,
        "capacity": 64,
        "lengths": [32],
        "timeout": 600,
        "previous": "prefill",
    },
    "capacity128": {
        "block": 4,
        "capacity": 128,
        "lengths": [32],
        "timeout": 600,
        "previous": "capacity64",
    },
}


def launch_count(prompt_length, predictions, block):
    if prompt_length < 1 or predictions < 1 or not 1 <= block <= 8:
        raise ValueError("invalid graph request dimensions")
    return prompt_length // block + prompt_length % block + predictions - 1


def prepare(deployment, checkpoint, output):
    """Load the independent CPU model once; reuse identical token/cache references."""
    import torch
    from transformers import GenerationConfig

    from .reference import load_model, source_record

    root = Path(__file__).resolve().parents[4]
    passage = "Chapter 1\n" + (root / "tests/fixtures/opt_continuation.txt").read_text()
    request, tokenizer = prepare_request(
        deployment, checkpoint, output / "base", passage, 128, prompt_tokens=32
    )
    if request["selection"]["mode"] != "greedy":
        raise ValueError("experimental graph path requires greedy checkpoint defaults")
    manifest = json.loads((deployment / "deployment-manifest.json").read_text())
    model, source = load_model(checkpoint)
    if source["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("CPU checkpoint differs from deployment")
    model.generation_config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    torch.set_num_threads(4)
    references, cases = {}, {}
    with torch.inference_mode():
        for name, policy in CASES.items():
            directory = output / name
            directory.mkdir()
            components = write_config(directory, manifest["config"], policy["capacity"])
            requests = []
            for index, length in enumerate(policy["lengths"]):
                ids = request["tokens"][:length]
                if length not in references:
                    input_ids = torch.tensor([ids], dtype=torch.long)
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=4,
                    )[0, length:].tolist()
                    trajectory = ids + generated[:-1]
                    cache = model(
                        input_ids=torch.tensor([trajectory]), use_cache=True
                    ).past_key_values
                    tensors = {
                        f"layer.{layer}.{kind}": getattr(cache.layers[layer], kind)[0]
                        .numpy()
                        .copy()
                        for layer in range(components.layers)
                        for kind in ("keys", "values")
                    }
                    references[length] = (generated, tensors)
                predicted, tensors = references[length]
                np.savez(directory / f"cache.{index}.npz", **tensors)
                current = request | {
                    "tokens": ids,
                    "prompt": tokenizer.decode(ids, skip_special_tokens=True),
                    "capacity": policy["capacity"],
                    "kv_bytes": components.state_bytes(range(components.layers)),
                }
                requests.append(
                    {
                        "request": current,
                        "expected_tokens": predicted,
                        "expected_text": tokenizer.decode(predicted, skip_special_tokens=True),
                    }
                )
            (directory / "requests.txt").write_text(
                "".join(" ".join(map(str, item["request"]["tokens"])) + "\n" for item in requests)
            )
            cases[name] = policy | {
                "requests": requests,
                "maximum": 4,
                "heads": components.heads,
                "head_dim": components.head_dim,
                "layers": components.layers,
            }
    metadata = {
        "status": "prepared",
        "cases": cases,
        "reference": source_record(),
        "deployment_manifest_sha256": request["manifest_sha256"],
        "files": {
            str(path.relative_to(output)): fingerprint(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
        "hardware_validated": False,
    }
    write_json(output / "prepared.json", metadata)
    return metadata


def verify_prepared(directory, deployment):
    metadata = json.loads((directory / "prepared.json").read_text())
    if metadata.get("status") != "prepared" or not metadata.get("files"):
        raise ValueError("missing completed CPU preparation")
    if metadata["deployment_manifest_sha256"] != fingerprint(
        deployment / "deployment-manifest.json"
    ):
        raise ValueError("prepared references use a different deployment")
    for name, expected in metadata["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or fingerprint(path) != expected:
            raise ValueError(f"prepared reference changed: {name}")
    return metadata


def read_cache(path, shape):
    expected_bytes = int(np.prod(shape)) * 2
    if not path.is_file() or path.stat().st_size != expected_bytes:
        return None
    return np.fromfile(path, dtype="<f2").reshape(shape)


def cache_checks(case, fixture, output):
    """Compare valid CPU prefixes and require byte-exact repeated-request state."""
    checks = {}
    shape = (case["heads"], case["capacity"], case["head_dim"])
    for index, item in enumerate(case["requests"]):
        consumed = len(item["request"]["tokens"]) + len(item["expected_tokens"]) - 1
        with np.load(fixture / f"cache.{index}.npz") as reference:
            for name in reference.files:
                actual = read_cache(output / f"measured.{index}.{name}.bin", shape)
                checks[f"cache.{index}.{name}"] = (
                    actual is not None
                    and compare_arrays(
                        actual[:, :consumed], reference[name], **TOLERANCES["device"]
                    )["passed"]
                )
                # CPU tolerance alone could hide persistent-state corruption.
                for earlier in range(index):
                    if case["requests"][earlier] != item:
                        continue
                    previous = read_cache(output / f"measured.{earlier}.{name}.bin", shape)
                    checks[f"reset.{index}.{name}"] = (
                        actual is not None
                        and previous is not None
                        and actual[:, :consumed].tobytes() == previous[:, :consumed].tobytes()
                    )
    return checks


def evaluate(case, fixture, output, evidence, cache_readback):
    checks, samples = {}, []
    total_launches = total_predictions = 0
    for index, item in enumerate(case["requests"]):
        path = output / f"measured.{index}.json"
        result = json.loads(path.read_text()) if path.is_file() else {}
        request, expected = item["request"], item["expected_tokens"]
        count = launch_count(len(request["tokens"]), len(expected), case["block"])
        total_launches += count
        total_predictions += len(expected)
        consumed = len(request["tokens"]) + len(expected) - 1
        valid = valid_generation(result, request, expected, case["maximum"])
        checks[f"request.{index}"] = valid
        checks[f"transfers.{index}"] = (
            result.get("launches") == count
            and result.get("uploads") == count
            and result.get("upload_bytes") == 4 * (consumed + 2 * count)
            and result.get("reads") == len(expected)
            and result.get("read_bytes") == 4 * len(expected)
        )
        samples.append(result)
    lifecycle_path = output / "run.json"
    lifecycle = json.loads(lifecycle_path.read_text()) if lifecycle_path.is_file() else {}
    checks["lifecycle"] = (
        clean_execution(evidence)
        and lifecycle.get("status") == "executed"
        and lifecycle.get("released") is True
        and lifecycle.get("layers") == case["layers"]
        and lifecycle.get("prefill_block") == case["block"]
        and lifecycle.get("capacity") == case["capacity"]
        and lifecycle.get("launches") == total_launches
        and lifecycle.get("weight_payload_bytes") == case["requests"][0]["request"]["weight_bytes"]
        and elapsed(lifecycle.get("initialize_seconds"))
    )
    resource_path = output / "host-resources.jsonl"
    resources = (
        [json.loads(line) for line in resource_path.read_text().splitlines()]
        if resource_path.is_file()
        else []
    )
    phases = ["before_initialize", "after_initialize"]
    phases += [f"measured.{i}" for i in range(len(samples))]
    phases += ["after_release"]
    checks["host_observations"] = [row.get("phase") for row in resources] == phases and all(
        type(row.get("rss_bytes")) is int and row["rss_bytes"] > 0 for row in resources
    )
    observations = None
    try:
        observations = summarize_sdk_timing(output / "sdk-timing.tsv")
        calls = observations["measured_api_calls"]
        checks["fixed_graph_calls"] = (
            calls.get("vsi_nn_RunGraph") == total_launches
            and calls.get("vsi_nn_CopyDataToTensor") == total_launches
            and calls.get("vsi_nn_ConvertTensorToData") == total_predictions
            and calls.get("vxVerifyGraph", 0) == calls.get("vsi_nn_VerifyGraph", 0) == 0
            and not any(row["exceptions"] for row in observations["rows"])
        )
    except (OSError, ValueError, KeyError):
        checks["fixed_graph_calls"] = False
    options = evidence.get("runtime_options", {})
    checks["untraced"] = (
        evidence.get("driver_traced") is False
        and options.get("SPECFERRY_SDK_TIMING") == "summary"
        and options.get("VIV_VX_ENABLE_PRINT_TARGET") == "0"
        and all(
            options.get(name) in (None, "", "0")
            for name in ("VIV_VX_PROFILE", "VIV_MEMORY_PROFILE")
        )
    )
    if observations and all(checks.values()):
        checks["sdk_timing_scope"] = observations["measured_sdk_seconds"] <= (
            sum(sample["total_seconds"] for sample in samples) + 1e-6
        )
    if cache_readback:
        checks.update(cache_checks(case, fixture, output))
    passed = all(checks.values())
    metrics = None
    if passed:
        intervals = [value for sample in samples for value in sample["token_seconds"]]
        metrics = {
            "decode_tokens_per_second": len(intervals) / sum(intervals) if intervals else None,
            "ttft_seconds": [sample["first_token_seconds"] for sample in samples],
            "initialize_seconds": lifecycle["initialize_seconds"],
            "prefill_tokens_per_second": sum(len(r["request"]["tokens"]) for r in case["requests"])
            / sum(sample["prefill_seconds"] for sample in samples),
            "measured_requests": len(samples),
            "decode_intervals": len(intervals),
            "warmups": 0,
            "scope": "bounded exploratory host-wall observation",
        }
    return {
        "status": "numerical_pass" if passed else "failed",
        "checks": checks,
        "samples": samples,
        "metrics": metrics,
        "sdk_timings": observations,
        "lifecycle": lifecycle,
        "hardware_execution_proven": False,
        "cache_compared": cache_readback,
        "evidence": evidence,
        "host_resources": resources,
        "request_rss_growth_bytes": resources[-2]["rss_bytes"] - resources[1]["rss_bytes"]
        if checks["host_observations"]
        else None,
        "requests": case["requests"],
        "capacity": case["capacity"],
        "block": case["block"],
        "physical_device_peak_bytes": None,
        "formal_performance_accepted": False,
    }


def capacity_comparison(previous, current):
    """Compare only matching clean observations, never mix old builds or prompts."""
    if (
        previous["status"] != current["status"]
        or current["status"] != "numerical_pass"
        or previous["capacity"] != 64
        or current["capacity"] != 128
        or previous["block"] != current["block"]
    ):
        raise ValueError("capacity comparison requires passed 64/128 cases with the same block")
    for field in ("boot_id", "binary_sha256", "sdk_sha256", "runtime_options"):
        if previous["evidence"][field] != current["evidence"][field]:
            raise ValueError(f"capacity comparison has different {field}")
    for report in (previous, current):
        if len(report["requests"]) != 1:
            raise ValueError("capacity comparison requires one matched request")
    before, after = previous["requests"][0], current["requests"][0]
    for field in ("tokens", "selection", "manifest_sha256"):
        if before["request"][field] != after["request"][field]:
            raise ValueError(f"capacity comparison has different {field}")
    if before["expected_tokens"] != after["expected_tokens"]:
        raise ValueError("capacity comparison has different output tokens")
    return {
        "capacity64": previous["metrics"],
        "capacity128": current["metrics"],
        "host_peak_rss_kib": [r["lifecycle"]["host_peak_rss_kib"] for r in (previous, current)],
        "conclusion": "Single cold request per capacity; no automatic optimization decision.",
    }


def write_results(path, report):
    lines = [
        "# Experimental graph observation",
        "",
        f"Status: {report['status']}",
        "",
        "No warmup; small-sample host wall times. Hardware engine/residency unproven.",
        "",
    ]
    metrics = report["metrics"]
    if metrics:
        rate = metrics["decode_tokens_per_second"]
        lines += [
            f"Decode: {rate:.4f} tokens/s"
            if rate is not None
            else "Decode: N/A (EOS on first prediction)",
            f"Prefill: {metrics['prefill_tokens_per_second']:.4f} tokens/s",
            f"TTFT (seconds): {metrics['ttft_seconds']}",
            f"Initialization: {metrics['initialize_seconds']:.3f} seconds",
            "",
        ]
    else:
        lines += ["Timing excluded because correctness/lifecycle validation failed.", ""]
    lines += [
        f"Host request RSS growth: {report['request_rss_growth_bytes']} bytes",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in report["checks"].items():
        if not passed:
            lines.append(f"| {name} | FAIL |")
    lines.append(f"| Total | {sum(report['checks'].values())}/{len(report['checks'])} passed |")
    for index, (request, sample) in enumerate(
        zip(report["requests"], report["samples"], strict=True)
    ):
        lines += [
            "",
            f"## Request {index}",
            "",
            "```text",
            request["request"].get("prompt", ""),
            "```",
            f"Output token IDs: {sample.get('tokens', [])}",
        ]
        if report["checks"][f"request.{index}"]:
            lines += ["", "```text", request.get("expected_text", ""), "```"]
    if report.get("capacity_comparison"):
        lines += [
            "",
            "## Capacity comparison",
            "",
            "```json",
            json.dumps(report["capacity_comparison"], indent=2),
            "```",
        ]
    path.write_text("\n".join(lines) + "\n")
