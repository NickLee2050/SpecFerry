"""Independent official-model expectations for OPT IO and resident generation."""

import hashlib
import json

import numpy as np
import torch
from transformers import GenerationConfig

from specferry.validation.arrays import compare_file, save_tensor
from specferry.validation.capabilities import compare_arrays
from specferry.validation.device import fingerprint, write_json

from .config import TOLERANCES
from .export import verify_export
from .generation import lifecycle, validate_tokens, write_config
from .reference import capture_layers, load_model, source_record


def record(directory, name, tensor, observations):
    observations[name] = save_tensor(directory / (name + ".bin"), tensor)


def prepare_selection(directory):
    """Exact independent cases for ties, negative maxima, block edges and the short tail."""
    directory.mkdir(parents=True, exist_ok=False)
    deployment = directory / "deployment"
    deployment.mkdir()
    table = np.zeros((19, 8), dtype="<f2")
    table[[3, 9], 1] = 2  # Tie across blocks: lowest token ID wins.
    table[18, 2] = 3  # Last valid row in the short tail.
    table[[17, 18], 3] = 2  # Tie inside the short tail.
    table[:, 4] = -np.arange(1, 20, dtype=np.float16)
    table[9, 5] = 4
    table[[7, 8], 6] = 3  # Adjacent tokens straddling a block boundary.
    weights = {
        "decoder.embed_tokens.weight": table,
        "decoder.embed_positions.weight": np.zeros((10, 8), dtype="<f2"),
        "decoder.project_in.weight": np.eye(8, dtype="<f2"),
        "decoder.project_out.weight": np.eye(8, dtype="<f2"),
    }
    index = ["specferry-np101-weights 1"]
    with (deployment / "weights.bin").open("wb") as stream:
        for name, value in weights.items():
            stream.write(bytes((-stream.tell()) % 64))
            offset, data = stream.tell(), value.tobytes()
            stream.write(data)
            shape = ",".join(map(str, reversed(value.shape)))
            index.append(
                f"{name} F16 {shape} {offset} {len(data)} {hashlib.sha256(data).hexdigest()}"
            )
    (deployment / "weights.index").write_text("\n".join(index) + "\n")
    (directory / "components.txt").write_text("specferry-opt-components 1\n8 16 2 1 8 1e-5\n")
    (directory / "model.txt").write_text("specferry-opt-model 1\n8 19 8 2 8 2 2 1\n")
    observations, tokens = {}, []
    lookup_tokens = [0, 3, 7, 8, 9, 17, 18]
    (directory / "cases.txt").write_text("".join(f"{t} {i}\n" for i, t in enumerate(lookup_tokens)))
    for column, token in enumerate(lookup_tokens):
        hidden = np.eye(8, dtype="<f2")[column : column + 1]
        logits = hidden @ table.T
        save_tensor(directory / f"hidden.{column}.bin", torch.from_numpy(hidden))
        for name, value in (("embedding", table[token]), ("projected", hidden), ("logits", logits)):
            record(directory, f"{name}.{column}", torch.from_numpy(value), observations)
        tokens.append(int(logits.argmax()))
    metadata = {
        "mode": "io",
        "count": len(tokens),
        "observations": observations,
        "tokens": tokens,
        "tolerances": {"atol": 0, "rtol": 0},
        "reference": "exact NumPy identity projections and enumerated FP16 row scores",
    }
    write_json(directory / "reference.json", metadata)
    return metadata


@torch.inference_mode()
def prepare(
    deployment, checkpoint, directory, mode, layers=24, capacity=512, steps=8, input_ids=None
):
    verify_export(deployment)
    manifest = json.loads((deployment / "deployment-manifest.json").read_text())
    model, source = load_model(checkpoint)
    if source["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("reference checkpoint and deployment have different sources")
    directory.mkdir(parents=True, exist_ok=False)
    components = write_config(directory, manifest["config"], capacity)
    if not 1 <= layers <= components.layers or (mode == "teacher" and not 1 <= steps <= capacity):
        raise ValueError("invalid layer or step count")
    observations, tokens = {}, []
    decoder = model.model.decoder
    if mode == "io":
        cases = [
            (token, min(position, capacity - 1))
            for token, position in (
                (0, 0),
                (2, 1),
                (4095, 2),
                (4096, 3),
                (49152, 4),
                (50271, capacity - 1),
            )
        ]
        (directory / "cases.txt").write_text(
            "".join(f"{token} {position}\n" for token, position in cases)
        )
        # Capture actual final-layer inputs for the output path, not random large vectors.
        captured = {}
        with capture_layers({23: decoder.layers[23]}, captured):
            model(input_ids=torch.tensor([[2, 2061, 32, 52, 259, 13, 1768, 116]]))
        hidden_sequence = captured["layer.23.output"]
        for index, (token, position) in enumerate(cases):
            hidden = hidden_sequence[:, index : index + 1]
            save_tensor(directory / f"hidden.{index}.bin", hidden)
            embedding = decoder.project_in(decoder.embed_tokens(torch.tensor([[token]])))
            embedding += decoder.embed_positions.weight[position + 2].view(1, 1, -1)
            projected = decoder.project_out(hidden)
            logits = model.lm_head(projected)
            for name, value in (
                ("embedding", embedding),
                ("projected", projected),
                ("logits", logits),
            ):
                record(directory, f"{name}.{index}", value, observations)
            tokens.append(int(logits.argmax(-1).item()))
        count = len(cases)
    elif mode == "teacher":
        # Fixed BOS + prompt prefix. Long trajectories deliberately repeat token IDs.
        seed = [2, 2061, 32, 52, 259, 13, 1768, 116]
        input_ids = input_ids or [seed[index % len(seed)] for index in range(steps)]
        steps = len(input_ids)
        validate_tokens(input_ids, manifest["config"]["vocab_size"], capacity)
        (directory / "tokens.txt").write_text(" ".join(map(str, input_ids)) + "\n")
        decoder.layers = decoder.layers[:layers]
        selected = dict(enumerate(decoder.layers))
        cache = None
        for index, token in enumerate(input_ids):
            captured = {}
            with capture_layers(selected, captured):
                result = model(
                    input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True
                )
            cache = result.past_key_values
            record(directory, f"teacher.{index}.embedding", captured["layer.0.input"], observations)
            for layer in selected:
                record(
                    directory,
                    f"teacher.{index}.layer.{layer}",
                    captured[f"layer.{layer}.output"],
                    observations,
                )
                for name in ("keys", "values"):
                    prefix = getattr(cache.layers[layer], name)[0]
                    record(directory, f"teacher.{index}.layer.{layer}.{name}", prefix, observations)
            record(
                directory,
                f"teacher.{index}.projected",
                decoder.project_out(captured[f"layer.{layers - 1}.output"]),
                observations,
            )
            record(directory, f"teacher.{index}.logits", result.logits, observations)
            tokens.append(int(result.logits.argmax(-1).item()))
        count = steps
    else:
        raise ValueError("unknown generation validation mode")
    metadata = {
        "mode": mode,
        "layers": layers,
        "count": count,
        "capacity": capacity,
        "observations": observations,
        "tokens": tokens,
        "reference": source_record(),
        "tolerances": TOLERANCES["device"],
        "manifest_sha256": fingerprint(deployment / "deployment-manifest.json"),
        "weight_payload_bytes": sum(
            item["bytes"]
            for item in manifest["tensors"]
            if not item["name"].startswith("decoder.layers.")
            or int(item["name"].split(".")[2]) < layers
        ),
        "kv_payload_bytes": components.state_bytes(range(layers)),
        "fixture_sha256": {p.name: fingerprint(p) for p in sorted(directory.iterdir())},
    }
    write_json(directory / "reference.json", metadata)
    return metadata


def read_tokens(path):
    try:
        return [int(value) for value in path.read_text().split()]
    except (OSError, ValueError):
        return []


@torch.inference_mode()
def reference_generation(checkpoint, request, maximum_new_tokens):
    """Independent checkpoint-policy oracle; never part of device decoding."""
    maximum = min(maximum_new_tokens, request["capacity"] - len(request["tokens"]) + 1)
    expected = []
    if maximum:
        model, _ = load_model(checkpoint)
        # The reference loader constructs the module directly from model config.
        # Attach the checkpoint's generation policy, then let generate() resolve
        # its own defaults without forcing do_sample=False.
        model.generation_config = GenerationConfig.from_pretrained(
            checkpoint, local_files_only=True
        )
        if model.generation_config.get_generation_mode().value != "greedy_search":
            raise ValueError("CPU token equality requires greedy checkpoint defaults")
        ids = torch.tensor([request["tokens"]], dtype=torch.long)
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=maximum,
        )
        expected = output[0, ids.shape[1] :].tolist()
    return {"tokens": expected, "reference": source_record()}


def compare_generation(checkpoint, request, generation, maximum_new_tokens):
    """Optional validation after native execution; never part of device decoding."""
    reference = reference_generation(checkpoint, request, maximum_new_tokens)
    expected = reference["tokens"]
    actual = generation["tokens"]
    mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    if mismatch is None and len(actual) != len(expected):
        mismatch = min(len(actual), len(expected))
    return {
        "passed": actual == expected,
        "expected_tokens": expected,
        "first_mismatch": mismatch,
        "reference": reference["reference"],
    }


def evaluate(fixture, actual, metadata, evidence):
    checks = {}
    for name, spec in metadata["observations"].items():
        if name.endswith((".keys", ".values")):
            continue
        checks[name] = compare_file(
            actual / (name + ".bin"), fixture / (name + ".bin"), spec, metadata["tolerances"]
        )
        if metadata["mode"] == "teacher":
            index = int(name.split(".")[1])
            for repeat in ("reset", "final", "fresh"):
                if repeat != "reset" and index != metadata["count"] - 1:
                    continue
                checks[repeat + name] = compare_file(
                    actual / (name.replace("teacher.", repeat + ".", 1) + ".bin"),
                    actual / (name + ".bin"),
                    spec,
                    {"atol": 0, "rtol": 0},
                )
    if metadata["mode"] == "teacher":
        checks.update(compare_cache_prefixes(fixture, actual, metadata))
    execution_path = actual / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.is_file() else {}
    if metadata["mode"] == "io":
        checks["tokens"] = {"passed": read_tokens(actual / "tokens.txt") == metadata["tokens"]}
        checks["lifecycle"] = {
            "passed": (
                evidence.get("returncode") == 0
                and evidence.get("process_group_exited") is True
                and evidence.get("device_recovery_required") is False
                and execution.get("status") == "executed"
                and execution.get("released") is True
                and execution.get("bounds_rejected") is True
                and execution.get("cases") == metadata["count"]
            )
        }
    else:
        count, layers = metadata["count"], metadata["layers"]
        for repeat in ("teacher", "reset", "final", "fresh"):
            expected = (
                metadata["tokens"] if repeat in ("teacher", "reset") else metadata["tokens"][-1:]
            )
            checks[repeat + ".tokens"] = {
                "passed": read_tokens(actual / (repeat + ".tokens.txt")) == expected
            }
        checks["lifecycle"] = {
            "passed": (
                lifecycle(evidence, execution)
                and execution.get("bounds_rejected") is True
                and execution.get("reset_rejected_stale") is True
                and execution.get("steps") == count * 4
                and execution.get("cache_writes") == count * 4 * layers
                and execution.get("uploads") == count * 4 * (layers + 2)
                and execution.get("upload_bytes") == count * 16 * (layers + 2)
                and execution.get("reads") == count * 2 + 2
                and execution.get("read_bytes") == (count * 2 + 2) * 4
            )
        }
    failed = [name for name, item in checks.items() if not item["passed"]]
    return {
        "status": "numerical_pass" if not failed else "failed",
        "checks": checks,
        "first_failures": failed[:10],
        "execution": execution,
        "evidence": evidence,
        "hardware_execution_proven": False,
        "device_residency_proven": False,
    }


def compare_cache_prefixes(fixture, actual, metadata):
    """Check independent CPU prefixes, exact append preservation, and reset/fresh reuse."""
    checks = {}
    for name, spec in metadata["observations"].items():
        if not name.endswith((".keys", ".values")):
            continue
        index = int(name.split(".")[1])
        heads, length, width = spec["shape"]
        expected = np.fromfile(fixture / f"{name}.bin", dtype="<f2").reshape(heads, length, width)
        for repeat in ("teacher", "reset", "final", "fresh"):
            if repeat in ("final", "fresh") and index != metadata["count"] - 1:
                continue
            label = name.replace("teacher.", repeat + ".", 1)
            path = actual / f"{label}.bin"
            elements = heads * metadata["capacity"] * width
            if not path.is_file() or path.stat().st_size != elements * 2:
                checks[label] = {"passed": False, "reason": "missing or invalid cache size"}
                continue
            data = np.fromfile(path, dtype="<f2")
            prefix = data.reshape(heads, metadata["capacity"], width)[:, :length]
            checks[label] = compare_arrays(prefix, expected, **metadata["tolerances"])
            checks[label]["valid_length"] = length
            if repeat in ("teacher", "reset") and index:
                previous = actual / f"{label.replace(f'.{index}.', f'.{index - 1}.', 1)}.bin"
                old = (
                    np.fromfile(previous, dtype="<f2")
                    if previous.is_file() and previous.stat().st_size == data.nbytes
                    else np.array([])
                )
                checks[label + ".history"] = {
                    "passed": bool(
                        old.size == data.size
                        and np.array_equal(
                            prefix[:, :-1].view(np.uint16),
                            old.reshape(heads, metadata["capacity"], width)[:, :index].view(
                                np.uint16
                            ),
                        )
                    )
                }
    return checks
