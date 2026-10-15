"""Bounded text-generation preparation and a sequential pinned CPU oracle."""

import json

import numpy as np
import torch
from transformers import GenerationConfig, Qwen3_5TextConfig

from specferry.data.checkpoint import sha256
from specferry.validation.device import write_json

from .checkpoint import inventory
from .components import Components
from .export import verify_export
from .model import encode, load_text_model, load_tokenizer, source_record
from .precision import precision


def prepare_request(
    deployment,
    checkpoint,
    directory,
    prompt,
    capacity,
    maximum,
    repeats,
    embedding_lookup="gather-nd",
    matmul_feed_forward_layer=9,
    token_readback="mapped",
):
    if token_readback not in ("sdk", "mapped"):
        raise ValueError("unsupported Qwen token readback transport")
    if embedding_lookup not in ("gather", "gather-nd"):
        raise ValueError("unsupported Qwen embedding lookup operator")
    if type(matmul_feed_forward_layer) is not int or not 0 <= matmul_feed_forward_layer < 24:
        raise ValueError("Qwen MatMul feed-forward layer must be a zero-based layer in 0..23")
    verify_export(deployment)
    manifest = json.loads((deployment / "deployment-manifest.json").read_text())
    config = Qwen3_5TextConfig(**manifest["text_config"])
    generation_file = checkpoint / "generation_config.json"
    generation = (
        GenerationConfig.from_pretrained(checkpoint, local_files_only=True)
        if generation_file.exists()
        else GenerationConfig.from_model_config(config)
    )
    # Transformers 5.16.1 resolves omitted fields inside generate(), as in the
    # accepted OPT entry. Keep the same pinned defaults for this adapter.
    defaults = GenerationConfig._get_default_generation_params()
    generation.update(**defaults, defaults_only=True)
    baseline = GenerationConfig(**defaults)
    handled = {
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "max_length",
        "max_new_tokens",
        "transformers_version",
        "output_attentions",
        "output_hidden_states",
    }
    unsupported = [
        name
        for name, value in generation.to_dict().items()
        if not name.startswith("_")
        and name not in handled
        and value != getattr(baseline, name, None)
    ]
    if unsupported:
        raise ValueError("unsupported Qwen generation settings: " + ", ".join(unsupported))
    if generation.output_attentions or generation.output_hidden_states:
        raise ValueError("Qwen entry does not return all internal activations or attention maps")
    if generation.eos_token_id is None or not isinstance(generation.eos_token_id, int):
        raise ValueError(
            "Qwen entry requires resolved greedy, single-EOS defaults; unsupported policy is not substituted"
        )
    components = Components.from_text_config(manifest["text_config"], capacity=capacity)
    tokenizer = load_tokenizer(checkpoint)
    rendered, ids = encode(tokenizer, prompt)
    if not 1 <= ids.shape[1] <= capacity or not 1 <= maximum <= 32 or not 1 <= repeats <= 3:
        raise ValueError("Qwen prompt/output/repeat count exceeds the bounded entry")
    directory.mkdir(parents=True, exist_ok=False)
    components.write(directory)
    tokens = ids[0].tolist()
    (directory / "tokens.txt").write_text(" ".join(map(str, tokens)) + "\n")
    (directory / "embedding-lookup.txt").write_text(embedding_lookup + "\n")
    (directory / "token-readback.txt").write_text(token_readback + "\n")
    (directory / "matmul-feed-forward-layer.txt").write_text(str(matmul_feed_forward_layer) + "\n")
    (directory / "policy.txt").write_text(
        f"{config.vocab_size} {capacity} {generation.bos_token_id or 0} {generation.eos_token_id} {maximum} {repeats}\n"
    )
    request = dict(
        prompt=prompt,
        rendered=rendered,
        tokens=tokens,
        capacity=capacity,
        maximum=maximum,
        repeats=repeats,
        embedding_lookup=embedding_lookup,
        matmul_feed_forward_layer=matmul_feed_forward_layer,
        token_readback=token_readback,
        deployment_sha256=sha256(deployment / "deployment-manifest.json"),
        selection=dict(
            origin="generation_config"
            if generation_file.exists()
            else "pinned_reference_model_defaults",
            effective=generation.to_dict(),
            empty_prompt="rejected; BOS sentinel is unused",
        ),
        reference=source_record(),
    )
    request["selection"]["effective"]["max_new_tokens"] = maximum
    request["selection"]["overrides"] = {"max_new_tokens": maximum}
    write_json(directory / "request.json", request)
    return request, tokenizer


@torch.inference_mode()
def prepare_reference(checkpoint, request, directory):
    torch.set_num_threads(4)
    model = load_text_model(checkpoint, inventory(checkpoint), "deployment-fp16")
    cache, captured, tokens = None, {}, []

    def capture(module, args, output):
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[23].register_forward_hook(capture)
    count = 0
    with precision("deployment-fp16"):
        for token in request["tokens"]:
            result = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            count += 1
        for index in range(request["maximum"]):
            token = int(result.logits[0, -1].argmax())
            tokens.append(token)
            if token == request["selection"]["effective"]["eos_token_id"]:
                reason = "eos"
                break
            if index + 1 == request["maximum"]:
                reason = "max_new_tokens"
                break
            if count == request["capacity"]:
                reason = "capacity"
                break
            result = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            count += 1
    handle.remove()
    snapshots = {
        "hidden": captured["hidden"],
        "recurrent": cache.layers[22].recurrent_states[0],
        "keys": cache.layers[23].keys,
        "values": cache.layers[23].values,
    }
    contract = {}
    for name, tensor in snapshots.items():
        array = tensor.detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise ValueError("Qwen CPU reference contains nonfinite values")
        array.tofile(directory / f"expected.{name}.bin")
        contract[name] = dict(shape=list(array.shape), dtype=array.dtype.str)
    metadata = dict(
        tokens=tokens,
        consumed=count,
        stop_reason=reason,
        snapshots=contract,
        reference=request["reference"],
        prompt_length=len(request["tokens"]),
    )
    write_json(directory / "reference.json", metadata)
    return metadata
