"""Host request preparation; all model arithmetic remains in the native SDK path."""

import json
from dataclasses import replace

import transformers
from transformers import AutoTokenizer, GenerationConfig

from specferry.validation.device import fingerprint, write_json

from .config import validate_config
from .export import verify_export


def resolve_selection(checkpoint, *, sample=False, seed=0):
    """Resolve checkpoint defaults and reject policies the native path cannot execute."""
    path = checkpoint / "generation_config.json"
    config = GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
    # Transformers 5.16.1 leaves omitted fields as None until generate() fills
    # them. Use its pinned defaults instead of inventing our own model policy.
    defaults = GenerationConfig._get_default_generation_params()
    config.update(**defaults, defaults_only=True)
    baseline = GenerationConfig(**defaults)
    if sample:
        # Preserve the explicitly requested, previously validated sampling mode.
        config.update(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)

    # Length is controlled by --max-new-tokens; special IDs are checked against
    # the deployment below. Sampling filters are inactive during greedy decoding.
    handled = {
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "transformers_version",
        "max_length",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
    }
    unsupported = [
        name
        for name, value in config.to_dict().items()
        if not name.startswith("_")
        and name not in handled
        and value != getattr(baseline, name, None)
    ]
    mode = config.get_generation_mode().value
    if mode not in ("greedy_search", "sample"):
        unsupported.append(f"generation_mode={mode}")
    if config.do_sample:
        unsupported.extend(
            name
            for name, expected in {"temperature": 1.0, "top_k": 0, "top_p": 1.0}.items()
            if getattr(config, name) != expected
        )
    if unsupported:
        raise ValueError("unsupported checkpoint generation settings: " + ", ".join(unsupported))
    if type(seed) is not int or not 0 <= seed < 2**32 or (seed and not config.do_sample):
        raise ValueError("seed must be uint32 and is only meaningful with sampling")
    return {
        "mode": "multinomial" if config.do_sample else "greedy",
        "origin": "explicit_sample" if sample else "model_default",
        "source": str(path.resolve()),
        "source_sha256": fingerprint(path),
        "transformers_version": transformers.__version__,
        "resolved_config": config.to_dict(),
        "seed": seed if config.do_sample else None,
        "temperature": config.temperature if config.do_sample else None,
        "top_k": config.top_k if config.do_sample else None,
        "top_p": config.top_p if config.do_sample else None,
    }


def write_config(directory, config, capacity=512, block_rows=4096):
    components = replace(validate_config(config), capacity=capacity)
    components.write(directory / "components.txt")
    directory.joinpath("model.txt").write_text(
        "specferry-opt-model 1\n"
        f"{config['word_embed_proj_dim']} {config['vocab_size']} "
        f"{config['max_position_embeddings']} 2 {block_rows} "
        f"{config['bos_token_id']} {config['eos_token_id']} {config['pad_token_id']}\n"
    )
    return components


def validate_tokens(tokens, vocabulary, capacity):
    if len(tokens) > capacity or any(
        type(token) is not int or not 0 <= token < vocabulary for token in tokens
    ):
        raise ValueError("tokens exceed the vocabulary or cache capacity")
    return tokens


def prepare_request(
    deployment, checkpoint, directory, prompt, capacity=512, *, sample=False, seed=0
):
    verify_export(deployment)
    manifest = json.loads((deployment / "deployment-manifest.json").read_text())
    selection = resolve_selection(checkpoint, sample=sample, seed=seed)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if selection["resolved_config"][name] != manifest["config"][name]:
            raise ValueError(f"generation {name} differs from the native model contract")
    directory.mkdir(parents=True, exist_ok=False)
    components = write_config(directory, manifest["config"], capacity)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if not tokens:
        tokens = [manifest["config"]["bos_token_id"]]
    validate_tokens(tokens, manifest["config"]["vocab_size"], capacity)
    (directory / "tokens.txt").write_text(" ".join(map(str, tokens)) + "\n")
    metadata = {
        "prompt": prompt,
        "tokens": tokens,
        "capacity": capacity,
        "manifest_sha256": fingerprint(deployment / "deployment-manifest.json"),
        "batch_size": 1,
        "selection": selection,
        "attention_policy": "every supplied token is attended; no padded batches",
        "consumed_policy": "last predicted token is returned but not yet appended to KV",
        "weight_bytes": sum(item["bytes"] for item in manifest["tensors"]),
        "kv_bytes": components.state_bytes(range(components.layers)),
        "hardware_execution_proven": False,
    }
    write_json(directory / "request.json", metadata)
    return metadata, tokenizer


def lifecycle(evidence, execution):
    return (
        evidence.get("returncode") == 0
        and evidence.get("process_group_exited") is True
        and evidence.get("device_recovery_required") is False
        and execution.get("status") == "executed"
        and execution.get("phase") == "complete"
        and execution.get("released") is True
    )
