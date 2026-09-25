"""A/B/A CPU token and KV oracle for experimental fixed-graph inference."""

import json

import numpy as np
import torch
from transformers import GenerationConfig

from specferry.data.checkpoint import sha256
from specferry.validation.arrays import compare_arrays
from specferry.validation.device import write_json

from .config import EPSILON, TOLERANCES
from .reference import load_model


def validate_prefix_fixture(deployment, directory, layers):
    """Reject stale, full-model or malformed references before opening the device."""
    metadata = json.loads((directory / "reference.json").read_text())
    manifest = json.loads((deployment / "deployment-manifest.json").read_text())
    config = manifest["config"]
    if (
        metadata.get("mode") != "prefix"
        or metadata.get("layers") != layers
        or not 1 <= layers <= config["num_hidden_layers"]
        or metadata.get("count") != 2
        or not 2 <= metadata["capacity"] <= config["max_position_embeddings"]
        or metadata.get("manifest_sha256") != sha256(deployment / "deployment-manifest.json")
    ):
        raise ValueError("prefix fixture must match the deployment, layer count and two steps")
    checks = metadata.get("reference_checks", {})
    if not checks or not all(item["passed"] for item in checks.values()):
        raise ValueError("prefix fixture lacks passing CPU consistency checks")
    hashes = metadata["fixture_sha256"]
    for name in ("components.txt", "model.txt", "tokens.txt"):
        if hashes.get(name) != sha256(directory / name):
            raise ValueError(f"prefix fixture changed: {name}")
    components = (directory / "components.txt").read_text().split()
    dimensions = [
        config["hidden_size"],
        config["ffn_dim"],
        config["num_attention_heads"],
        layers,
        metadata["capacity"],
    ]
    if (
        len(components) != 8
        or components[:2] != ["specferry-opt-components", "1"]
        or list(map(int, components[2:7])) != dimensions
        or float(components[7]) != EPSILON
    ):
        raise ValueError("prefix native configuration differs from the CPU reference")
    model = (directory / "model.txt").read_text().split()
    if model != [
        "specferry-opt-model",
        "1",
        str(config["word_embed_proj_dim"]),
        str(config["vocab_size"]),
        str(config["max_position_embeddings"]),
        "2",
        "4096",
        str(config["bos_token_id"]),
        str(config["eos_token_id"]),
        str(config["pad_token_id"]),
    ]:
        raise ValueError("prefix input/output configuration differs from the checkpoint")
    tokens = [int(value) for value in (directory / "tokens.txt").read_text().split()]
    if len(tokens) != 2 or any(not 0 <= token < config["vocab_size"] for token in tokens):
        raise ValueError("prefix fixture needs two valid token IDs")
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    embedding, vocabulary = config["word_embed_proj_dim"], config["vocab_size"]
    expected = {}
    for step in range(2):
        shapes = {
            "lookup": [1, 1, embedding],
            "input_projection": [1, 1, hidden],
            "position_embedding": [1, 1, hidden],
            "embedding": [1, 1, hidden],
            "projected": [1, 1, embedding],
            "logits": [1, 1, vocabulary],
        }
        for layer in range(layers):
            shapes[f"layer.{layer}"] = [1, 1, hidden]
            for name in ("keys", "values"):
                shapes[f"layer.{layer}.{name}"] = [heads, step + 1, hidden // heads]
        expected.update({f"teacher.{step}.{name}": shape for name, shape in shapes.items()})
    for name, shape in expected.items():
        spec = metadata["observations"].get(name, {})
        size = int(np.prod(shape)) * 2
        path = directory / f"{name}.bin"
        if (
            spec != {"shape": shape, "dtype": "<f2", "bytes": size}
            or not path.is_file()
            or path.stat().st_size != size
            or hashes.get(path.name) != sha256(path)
        ):
            raise ValueError(f"invalid prefix reference shape, dtype or checksum: {name}")
        if not np.isfinite(np.fromfile(path, dtype="<f2")).all():
            raise ValueError(f"nonfinite prefix reference: {name}")
    predictions = metadata["tokens"]
    if len(predictions) != 2 or any(
        int(np.fromfile(directory / f"teacher.{step}.logits.bin", dtype="<f2").argmax()) != token
        for step, token in enumerate(predictions)
    ):
        raise ValueError("prefix token expectations disagree with truncated logits")
    return metadata


@torch.inference_mode()
def prepare(checkpoint, request, directory, maximum):
    model, _ = load_model(checkpoint)
    model.generation_config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    if request["selection"]["mode"] != "greedy":
        raise ValueError("experimental graph requires greedy checkpoint policy")
    requests = [
        request["tokens"],
        request["tokens"][: max(1, len(request["tokens"]) // 2)],
        request["tokens"],
    ]
    references = []
    for index, ids in enumerate(requests):
        count = min(maximum, request["capacity"] - len(ids) + 1)
        generated = model.generate(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long),
            max_new_tokens=count,
        )[0, len(ids) :].tolist()
        trajectory = ids + generated[:-1]
        cache = model(input_ids=torch.tensor([trajectory]), use_cache=True).past_key_values
        for layer, state in enumerate(cache.layers):
            for name in ("keys", "values"):
                getattr(state, name)[0].contiguous().numpy().tofile(
                    directory / f"expected.{index}.{layer}.{name}.bin"
                )
        references.append(
            {"tokens": generated, "consumed": len(trajectory), "prompt_length": len(ids)}
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
        "layers": config.num_hidden_layers,
        "heads": config.num_attention_heads,
        "width": config.hidden_size // config.num_attention_heads,
        "capacity": request["capacity"],
    }
    write_json(directory / "reference.json", metadata)
    return metadata


def evaluate(fixture, device, metadata, block):
    checks, runs = {}, []
    for index, reference in enumerate(metadata["requests"]):
        result = json.loads((device / f"measured.{index}.json").read_text())
        runs.append(result)
        prefix = reference["consumed"]
        length = reference["prompt_length"]
        launches = length // block + length % block + len(reference["tokens"]) - 1
        checks[f"request.{index}"] = (
            result["tokens"] == reference["tokens"]
            and result["consumed"] == prefix
            and result["launches"] == launches
            and result["uploads"] == launches
            and result["reads"] == len(reference["tokens"])
        )
        for layer in range(metadata["layers"]):
            for name in ("keys", "values"):
                key = f"measured.{index}.layer.{layer}.{name}"
                expected = np.fromfile(
                    fixture / f"expected.{index}.{layer}.{name}.bin", dtype="<f2"
                )
                actual = np.fromfile(device / f"{key}.bin", dtype="<f2")
                # Ignore unconsumed slots while checking every head in the valid prefix.
                actual = actual.reshape(metadata["heads"], metadata["capacity"], metadata["width"])[
                    :, :prefix
                ].reshape(-1)
                checks[key] = compare_arrays(actual, expected, **TOLERANCES["device"])["passed"]
                if index == 2:
                    previous = (
                        np.fromfile(device / f"measured.0.layer.{layer}.{name}.bin", dtype="<f2")
                        .reshape(metadata["heads"], metadata["capacity"], metadata["width"])[
                            :, :prefix
                        ]
                        .reshape(-1)
                    )
                    checks[key + ".reset"] = np.array_equal(actual.view("u2"), previous.view("u2"))
    return checks, runs
