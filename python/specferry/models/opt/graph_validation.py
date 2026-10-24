"""Independent CPU token, logit and KV oracle for fixed-graph request matrices."""

import json
import shutil

import numpy as np
import torch
from transformers import GenerationConfig

from specferry.validation.arrays import compare_arrays
from specferry.validation.device import write_json

from .config import TOLERANCES
from .reference import load_model


def generation_cache_layers(readback, layers, layer_range=None):
    """Resolve explicit diagnostic scope; ranges are inclusive and one-based at the CLI."""
    if readback not in ("none", "logits", "kv"):
        raise ValueError("cache readback must be none, logits or kv")
    if (readback == "kv") != (layer_range is not None):
        raise ValueError("kv readback requires an explicit cache layer range")
    if readback == "kv":
        if (
            len(layer_range) != 2
            or any(type(value) is not int for value in layer_range)
            or not 1 <= layer_range[0] <= layer_range[1] <= layers
        ):
            raise ValueError("cache layer range must be within the configured model")
        return range(layer_range[0] - 1, layer_range[1])
    return range(0)


@torch.inference_mode()
def prepare(
    checkpoint,
    request,
    directory,
    maximum,
    *,
    layers=None,
    prefill_block=1,
    cache_readback="none",
    cache_layer_range=None,
    request_tokens=None,
):
    if type(prefill_block) is not int or prefill_block not in (1, 4, 8, 16):
        raise ValueError("prefill block must be 1, 4, 8 or 16")
    if request["capacity"] % prefill_block:
        raise ValueError("capacity must be divisible by prefill block")
    model, _ = load_model(checkpoint)
    model.generation_config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    if layers is not None:
        if not 1 <= layers <= model.config.num_hidden_layers:
            raise ValueError("invalid truncated OPT layer count")
        model.model.decoder.layers = model.model.decoder.layers[:layers]
        model.config.num_hidden_layers = layers
    cache_layers = generation_cache_layers(
        cache_readback, model.config.num_hidden_layers, cache_layer_range
    )
    if request["selection"]["mode"] != "greedy":
        raise ValueError("experimental graph requires greedy checkpoint policy")
    requests = (
        request_tokens
        if request_tokens is not None
        else [
            request["tokens"],
            request["tokens"][: max(1, len(request["tokens"]) // 2)],
            request["tokens"],
        ]
    )
    if not 1 <= len(requests) <= 64 or any(
        not 1 <= len(ids) <= request["capacity"]
        or any(type(token) is not int or not 0 <= token < model.config.vocab_size for token in ids)
        for ids in requests
    ):
        raise ValueError("expected one to 64 nonempty requests within capacity and vocabulary")
    references = []
    for index, ids in enumerate(requests):
        repeated = next((prior for prior in range(index) if requests[prior] == ids), None)
        if repeated is not None:
            # Identical inputs reuse the CPU oracle; the device still executes both requests.
            for path in directory.glob(f"expected.{repeated}.*.bin"):
                name = path.name.replace(f"expected.{repeated}.", f"expected.{index}.", 1)
                shutil.copyfile(path, directory / name)
            references.append(references[repeated] | {"repeat_of": repeated})
            continue
        count = min(maximum, request["capacity"] - len(ids) + 1)
        generated = model.generate(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            max_new_tokens=count,
        )[0, len(ids) :].tolist()
        trajectory = ids + generated[:-1]
        # The reference processes the complete causal trajectory independently of
        # the SDK's block schedule, including the last consumed decode token.
        if cache_readback != "none":
            reference = model(input_ids=torch.tensor([trajectory]), use_cache=True)
            cache = reference.past_key_values
            if cache_readback == "logits":
                reference.logits[0, -1].to(torch.float16).contiguous().numpy().astype("<f2").tofile(
                    directory / f"expected.{index}.logits.bin"
                )
            for layer in cache_layers:
                state = cache.layers[layer]
                for name in ("keys", "values"):
                    getattr(state, name)[0].contiguous().numpy().tofile(
                        directory / f"expected.{index}.{layer}.{name}.bin"
                    )
        references.append(
            {
                "tokens": generated,
                "consumed": len(trajectory),
                "prompt_length": len(ids),
                "repeat_of": repeated,
            }
        )
    (directory / "requests.txt").write_text(
        "".join(" ".join(map(str, ids)) + "\n" for ids in requests)
    )
    (directory / "expected.txt").write_text(
        "".join(" ".join(map(str, r["tokens"])) + "\n" for r in references)
    )
    config = model.config
    metadata = {
        "requests": references,
        "layers": len(model.model.decoder.layers),
        "heads": config.num_attention_heads,
        "width": config.hidden_size // config.num_attention_heads,
        "capacity": request["capacity"],
        "prefill_block": prefill_block,
        "cache_readback": cache_readback,
        "cache_layer_range": cache_layer_range,
        "vocabulary": config.vocab_size,
        "logits_dtype": "<f2",
        "logits_scope": "last_consumed_token",
    }
    write_json(directory / "reference.json", metadata)
    return metadata


def _mapped_slot_enabled(metadata, variant, block, kv_transfer):
    if kv_transfer not in (None, "device-copy", "mapped-slot"):
        raise ValueError("unknown KV transfer strategy")
    mapped = kv_transfer == "mapped-slot"
    if mapped and (variant != "compiled-fullgraph" or metadata["capacity"] != 64 or block != 1):
        raise ValueError("mapped-slot KV requires compiled fullgraph C64/M1")
    return mapped


def _mapped_slot_transfers(metadata, launches, predictions, reset):
    caches = 2 * metadata["layers"]
    slot_bytes = caches * metadata["heads"] * metadata["width"] * 4
    control_bytes = 8 + metadata["heads"] * 8 + metadata["capacity"] * 4
    return {
        "uploads": launches * (4 + caches) + (caches if reset else 0),
        "upload_bytes": launches * (control_bytes + slot_bytes)
        + (slot_bytes * metadata["capacity"] if reset else 0),
        "reads": launches * caches + predictions,
        "read_bytes": launches * slot_bytes + 4 * predictions,
    }


def evaluate(
    fixture,
    device,
    metadata,
    block,
    reset_uploads_per_request=None,
    variant=None,
    decoder_group_size=None,
    prefill_head_policy=None,
    cache_readback="none",
    cache_layer_range=None,
    readback_transport=None,
    kv_transfer=None,
):
    mapped = _mapped_slot_enabled(metadata, variant, block, kv_transfer)
    cache_layers = generation_cache_layers(cache_readback, metadata["layers"], cache_layer_range)
    read_logits = cache_readback == "logits"
    if type(block) is not int or block not in (1, 4, 8, 16):
        raise ValueError("prefill block must be 1, 4, 8 or 16")
    if decoder_group_size is not None and (
        type(decoder_group_size) is not int or decoder_group_size != 1
    ):
        raise ValueError("expected decoder group size must be 1")
    if prefill_head_policy is not None and prefill_head_policy not in (
        "every-token",
        "last-token",
    ):
        raise ValueError("expected prefill head policy must be every-token or last-token")
    checks, runs = {}, []
    run_path = device / "run.json"
    native_run = json.loads(run_path.read_text()) if run_path.is_file() else {}
    if native_run.get("graph_layout") != "partitioned":
        raise ValueError("graph validation requires partitioned execution")
    checks["cache_readback"] = native_run.get("cache_readback") == cache_readback
    if cache_readback == "none":
        for field in ("diagnostic_reads", "diagnostic_read_bytes"):
            observed = native_run.get(field)
            checks[field] = type(observed) is int and observed == 0
    if cache_readback in ("logits", "kv"):
        per_request = 2 * len(cache_layers)
        bytes_per_request = (
            per_request
            * metadata["heads"]
            * metadata["capacity"]
            * metadata["width"]
            * (4 if variant in ("compiled-decoder", "compiled-fullgraph") else 2)
        )
        if read_logits:
            # The head exposes one tensor per vocabulary block, including a short last block.
            configuration = (fixture / "model.txt").read_text().split()
            if len(configuration) != 10 or configuration[:2] != ["specferry-opt-model", "1"]:
                raise ValueError("invalid OPT model configuration for logits readback")
            vocabulary, block_rows = int(configuration[3]), int(configuration[6])
            if vocabulary != metadata["vocabulary"] or block_rows <= 0:
                raise ValueError("logits layout disagrees with the CPU reference")
            per_request = (
                1
                if variant in ("compiled-ffn-finish", "compiled-fullgraph")
                else (vocabulary + block_rows - 1) // block_rows
            )
            bytes_per_request = vocabulary * 2
        expected_reads = len(metadata["requests"]) * per_request
        for field, expected in (
            ("diagnostic_reads", expected_reads),
            ("diagnostic_read_bytes", len(metadata["requests"]) * bytes_per_request),
        ):
            checks[field] = type(native_run.get(field)) is int and native_run[field] == expected
        if cache_readback == "kv":
            for field, expected in zip(
                ("cache_first_layer", "cache_last_layer"), cache_layer_range, strict=True
            ):
                checks[field] = type(native_run.get(field)) is int and native_run[field] == expected
    block_prefill = block != 1
    if block_prefill and variant == "compiled-ffn-finish" and block != 16:
        raise ValueError("compiled FFN prefill requires block16")
    if block_prefill and (
        variant not in ("merged-kv", "merged-kv-fc1-split2", "compiled-ffn-finish")
        or decoder_group_size not in (None, 1)
    ):
        raise ValueError(
            "partitioned block prefill requires a shared merged-kv variant and group 1"
        )
    if block_prefill and metadata["capacity"] % block:
        raise ValueError("capacity must be divisible by prefill block")
    if block_prefill or "prefill_block" in metadata:
        checks["reference.prefill_block"] = (
            type(metadata.get("prefill_block")) is int and metadata["prefill_block"] == block
        )
    if reset_uploads_per_request is None:
        reset_uploads_per_request = native_run.get("kv_reset_uploads_per_request", 0)
    if type(reset_uploads_per_request) is not int or reset_uploads_per_request < 0:
        raise ValueError("reset upload count must be a nonnegative integer")
    if variant is not None:
        if variant not in (
            "merged-kv",
            "merged-kv-fc1-split2",
            "compiled-ffn-finish",
            "compiled-decoder",
            "compiled-fullgraph",
        ):
            raise ValueError("unknown partitioned-model variant")
        expected_reset = (
            0
            if not mapped
            and variant in ("compiled-ffn-finish", "compiled-decoder", "compiled-fullgraph")
            else 2 * metadata["layers"]
        )
        checks["variant"] = (
            native_run.get("graph_layout") == "partitioned"
            and native_run.get("variant") == variant
            and native_run.get("kv_reset_uploads_per_request") == expected_reset
            and reset_uploads_per_request == expected_reset
        )
    expected_options = {}
    if kv_transfer is not None:
        expected_options["kv_transfer"] = kv_transfer
    if readback_transport is not None:
        if readback_transport not in ("sdk", "mapped"):
            raise ValueError("unknown diagnostic readback transport")
        if readback_transport == "mapped" and cache_readback == "none":
            raise ValueError("mapped transport requires diagnostic readback")
        expected_options["readback_transport"] = readback_transport
    if decoder_group_size is not None:
        expected_options["decoder_group_size"] = decoder_group_size
    if prefill_head_policy is not None:
        expected_options["prefill_head_policy"] = prefill_head_policy
    if block_prefill:
        expected_options.update(
            prefill_block=block,
            decoder_group_size=1,
            prefill_head_policy=prefill_head_policy or "every-token",
        )
    if expected_options:
        # Numerical equality alone cannot establish that an execution optimization ran.
        # Only explicit expectations require the newer initialization/counter schema.
        checks["execution_options.partitioned"] = native_run.get("graph_layout") == "partitioned"
        if variant is not None:
            expected_options["variant"] = variant
        initialized_path = device / "initialization.json"
        initialized = json.loads(initialized_path.read_text()) if initialized_path.is_file() else {}
        for scope, observed in (("run", native_run), ("initialization", initialized)):
            for name, expected_value in expected_options.items():
                actual = observed.get(name)
                checks[f"{scope}.{name}"] = (
                    type(actual) is type(expected_value) and actual == expected_value
                )
    step_counts = []
    for reference in metadata["requests"]:
        length = reference["prompt_length"]
        prefill_steps = length // block if block > 1 else 0
        decode_steps = (length % block if block > 1 else length) + len(reference["tokens"]) - 1
        step_counts.append(
            {
                "prefill_steps": prefill_steps,
                "decode_steps": decode_steps,
                "launches": prefill_steps + decode_steps,
            }
        )
    if prefill_head_policy is not None or block_prefill:
        expected_heads = sum(
            len(reference["tokens"]) if prefill_head_policy == "last-token" else counts["launches"]
            for reference, counts in zip(metadata["requests"], step_counts, strict=True)
        )
        observed_heads = native_run.get("head_runs")
        checks["head_runs"] = type(observed_heads) is int and observed_heads == expected_heads
    if read_logits:
        checks["reference.logits_contract"] = (
            type(metadata.get("vocabulary")) is int
            and metadata["vocabulary"] > 0
            and metadata.get("logits_dtype") == "<f2"
            and metadata.get("logits_scope") == "last_consumed_token"
        )
    if block_prefill:
        for name in ("launches", "prefill_steps", "decode_steps", "head_runs"):
            initial = initialized.get(name)
            checks[f"initialization.{name}"] = type(initial) is int and initial == 0
            expected_count = (
                expected_heads
                if name == "head_runs"
                else sum(counts[name] for counts in step_counts)
            )
            observed = native_run.get(name)
            checks[f"run.{name}"] = type(observed) is int and observed == expected_count
    for index, reference in enumerate(metadata["requests"]):
        result = json.loads((device / f"measured.{index}.json").read_text())
        runs.append(result)
        prefix = reference["consumed"]
        repeat_of = reference.get("repeat_of")
        if repeat_of is not None and (type(repeat_of) is not int or not 0 <= repeat_of < index):
            raise ValueError("repeat_of must refer to an earlier request")
        length = reference["prompt_length"]
        launches = step_counts[index]["launches"]
        transfers = {
            "uploads": launches
            * (4 if variant == "compiled-fullgraph" else 3 if variant == "compiled-decoder" else 1)
            + (reset_uploads_per_request if index else 0),
            "reads": len(reference["tokens"]),
        }
        if mapped:
            transfers = _mapped_slot_transfers(
                metadata, launches, len(reference["tokens"]), index > 0
            )
        checks[f"request.{index}"] = (
            result.get("tokens") == reference["tokens"]
            and result.get("consumed") == prefix
            and result.get("launches") == launches
            and all(result.get(name) == value for name, value in transfers.items())
        )
        if block_prefill:
            checks[f"reference.{index}.consumed"] = (
                type(length) is int
                and bool(reference["tokens"])
                and type(prefix) is int
                and prefix == length + len(reference["tokens"]) - 1
                and 1 <= length <= prefix <= metadata["capacity"]
            )
            for name, expected_count in (
                step_counts[index]
                | {
                    "prefill_block": block,
                    "head_runs": len(reference["tokens"])
                    if prefill_head_policy == "last-token"
                    else launches,
                }
            ).items():
                observed = result.get(name)
                checks[f"request.{index}.{name}"] = (
                    type(observed) is int and observed == expected_count
                )
        if cache_readback == "none":
            checks[f"request.{index}.read_bytes"] = result.get("read_bytes") == transfers.get(
                "read_bytes", 4 * len(reference["tokens"])
            )
            continue
        if read_logits:
            logits_key = f"measured.{index}.logits"
            try:
                expected_path = fixture / f"expected.{index}.logits.bin"
                actual_path = device / f"{logits_key}.bin"
                expected_bytes = metadata.get("vocabulary", 0) * 2
                if (
                    not checks["reference.logits_contract"]
                    or expected_path.stat().st_size != expected_bytes
                    or actual_path.stat().st_size != expected_bytes
                ):
                    raise ValueError("missing or wrong-size final logits")
                checks[logits_key] = compare_arrays(
                    np.fromfile(actual_path, dtype="<f2"),
                    np.fromfile(expected_path, dtype="<f2"),
                    **TOLERANCES["device"],
                )["passed"]
                if repeat_of is not None:
                    checks[logits_key + ".reset"] = (
                        actual_path.read_bytes()
                        == (device / f"measured.{repeat_of}.logits.bin").read_bytes()
                    )
            except (OSError, ValueError, TypeError):
                checks[logits_key] = False
        for layer in cache_layers:
            for name in ("keys", "values"):
                key = f"measured.{index}.layer.{layer}.{name}"
                expected = np.fromfile(
                    fixture / f"expected.{index}.{layer}.{name}.bin", dtype="<f2"
                )
                actual = np.fromfile(
                    device / f"{key}.bin",
                    dtype="<f4" if variant in ("compiled-decoder", "compiled-fullgraph") else "<f2",
                )
                # Ignore unconsumed slots while checking every head in the valid prefix.
                actual = actual.reshape(metadata["heads"], metadata["capacity"], metadata["width"])
                if block_prefill or mapped:
                    checks[key + ".suffix"] = bool(np.all(actual[:, prefix:].view("u2") == 0))
                actual = actual[:, :prefix].reshape(-1)
                checks[key] = compare_arrays(actual, expected, **TOLERANCES["device"])["passed"]
                if repeat_of is not None:
                    previous = (
                        np.fromfile(
                            device / f"measured.{repeat_of}.layer.{layer}.{name}.bin",
                            dtype="<f4"
                            if variant in ("compiled-decoder", "compiled-fullgraph")
                            else "<f2",
                        )
                        .reshape(metadata["heads"], metadata["capacity"], metadata["width"])[
                            :, :prefix
                        ]
                        .reshape(-1)
                    )
                    checks[key + ".reset"] = np.array_equal(actual.view("u2"), previous.view("u2"))
    return checks, runs


def evaluate_benchmark(
    device,
    metadata,
    block,
    *,
    warmups,
    repeats,
    variant,
    decoder_group_size,
    prefill_head_policy,
    kv_transfer=None,
):
    """Validate every resident repetition without requesting diagnostic tensor readback."""
    mapped = _mapped_slot_enabled(metadata, variant, block, kv_transfer)
    if block not in (1, 4, 8, 16) or not 0 <= warmups <= 5 or not 1 <= repeats <= 20:
        raise ValueError("invalid resident benchmark counts or prefill block")
    if prefill_head_policy not in ("every-token", "last-token"):
        raise ValueError("invalid prefill head policy")
    manifest = json.loads((device / "benchmark.json").read_text())
    native = json.loads((device / "run.json").read_text())
    initialized = json.loads((device / "initialization.json").read_text())
    checks = {
        "benchmark.completed": manifest.get("completed") is True,
        "benchmark.schema": manifest.get("schema_version") == 1,
        "benchmark.warmups": manifest.get("warmups") == warmups,
        "benchmark.repeats": manifest.get("repeats") == repeats,
        "benchmark.request_count": manifest.get("request_count") == len(metadata["requests"]),
        "released": native.get("released") is True and native.get("status") == "executed",
        "closed_rejected": native.get("closed_checks") == native.get("closed_rejections") == 1,
        "diagnostic_reads": native.get("diagnostic_reads") == 0,
        "diagnostic_read_bytes": native.get("diagnostic_read_bytes") == 0,
        "cache_readback": native.get("cache_readback") == "none",
        "graph_layout": native.get("graph_layout") == "partitioned",
        "reference.prefill_block": metadata.get("prefill_block") == block,
    }
    for segment in ("const", "nonconst"):
        peak = native.get(f"{segment}_peak_bytes")
        checks[f"memory.{segment}"] = (
            native.get(f"{segment}_live_bytes") == 0 and type(peak) is int and 0 <= peak <= 2**30
        )
    options = {
        "variant": variant,
        "decoder_group_size": decoder_group_size,
        "prefill_head_policy": prefill_head_policy,
        "prefill_block": block,
    }
    if kv_transfer is not None:
        options["kv_transfer"] = kv_transfer
    for scope, observed in (("run", native), ("initialization", initialized)):
        for name, expected in options.items():
            actual = observed.get(name)
            checks[f"{scope}.{name}"] = type(actual) is type(expected) and actual == expected
    reset_uploads = (
        0
        if not mapped
        and variant in ("compiled-ffn-finish", "compiled-decoder", "compiled-fullgraph")
        else 2 * metadata["layers"]
    )
    checks["reset_uploads"] = native.get("kv_reset_uploads_per_request") == reset_uploads
    expected_records, measured = [], []
    total_counts = dict(launches=0, prefill_steps=0, decode_steps=0, head_runs=0)
    capacity_checks = 0
    for index, reference in enumerate(metadata["requests"]):
        length = reference["prompt_length"]
        prefill = length // block if block > 1 else 0
        decode = (length % block if block > 1 else length) + len(reference["tokens"]) - 1
        counts = {
            "launches": prefill + decode,
            "prefill_steps": prefill,
            "decode_steps": decode,
            "head_runs": len(reference["tokens"])
            if prefill_head_policy == "last-token"
            else prefill + decode,
        }
        for kind, count in (("warmup", warmups), ("measured", repeats)):
            for iteration in range(count):
                name = f"{kind}.{index}.{iteration}"
                try:
                    result = json.loads((device / f"{name}.json").read_text())
                except (OSError, ValueError):
                    result = {}
                expected_records.append(
                    {
                        "name": name,
                        "request_index": index,
                        "iteration": iteration,
                        "prompt_length": length,
                        "kind": kind,
                        "result_file": f"{name}.json",
                        "tokens_match": True,
                    }
                )
                expected_result = counts | {
                    "tokens": reference["tokens"],
                    "consumed": reference["consumed"],
                    "prefill_block": block,
                    "uploads": counts["launches"]
                    * (
                        4
                        if variant == "compiled-fullgraph"
                        else 3
                        if variant == "compiled-decoder"
                        else 1
                    )
                    + (reset_uploads if len(expected_records) > 1 else 0),
                    "reads": len(reference["tokens"]),
                    "read_bytes": 4 * len(reference["tokens"]),
                }
                if mapped:
                    expected_result.update(
                        _mapped_slot_transfers(
                            metadata,
                            counts["launches"],
                            len(reference["tokens"]),
                            len(expected_records) > 1,
                        )
                    )
                checks[name] = all(
                    result.get(key) == value for key, value in expected_result.items()
                )
                for field in total_counts:
                    total_counts[field] += counts[field]
                capacity_checks += reference["consumed"] == metadata["capacity"]
                if kind == "measured":
                    measured.append(result | {"request_index": index, "iteration": iteration})
    checks["benchmark.records"] = manifest.get("runs") == expected_records
    for field, expected in total_counts.items():
        checks[f"run.{field}"] = native.get(field) == expected
        checks[f"initialization.{field}"] = initialized.get(field) == 0
    checks["capacity_rejected"] = (
        native.get("capacity_checks") == native.get("capacity_rejections") == capacity_checks
    )
    return checks, measured


def evaluate_rebuild(device, metadata):
    """Check release and exact final outputs across two fresh SDK Contexts."""
    summary = json.loads((device / "lifecycle.json").read_text())
    checks = {
        "lifecycle.completed": summary
        == {
            "cycles": 2,
            "reused_context": False,
            "recreated_context": True,
            "released": True,
        }
    }
    for cycle in range(2):
        directory = device / f"cycle.{cycle}"
        native = json.loads((directory / "run.json").read_text())
        checks[f"lifecycle.{cycle}.released"] = (
            native.get("released") is True
            and native.get("status") == "executed"
            and native.get("closed_checks") == native.get("closed_rejections") == 1
        )
        for segment in ("const", "nonconst"):
            checks[f"lifecycle.{cycle}.{segment}"] = (
                native.get(f"{segment}_live_bytes") == 0
                and 0 <= native.get(f"{segment}_peak_bytes", 2**40) <= 2**30
            )
        if cycle:
            for index in range(len(metadata["requests"])):
                baseline = device / "cycle.0" / f"measured.{index}.logits.bin"
                actual = directory / baseline.name
                checks[f"lifecycle.{cycle}.request.{index}.exact_logits"] = (
                    baseline.is_file()
                    and actual.is_file()
                    and baseline.read_bytes() == actual.read_bytes()
                )
    return checks
