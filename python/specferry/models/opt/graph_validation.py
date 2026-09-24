"""A/B/A CPU token and KV oracle for experimental fixed-graph inference."""

import json

import numpy as np
import torch
from transformers import GenerationConfig

from specferry.validation.arrays import compare_arrays
from specferry.validation.device import write_json

from .config import TOLERANCES
from .reference import load_model


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
