"""Official OPT CPU baseline and tensor hooks; no NP101 arithmetic is used here."""

import inspect
from contextlib import contextmanager

import torch
import transformers
from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM
from transformers.models.opt import modeling_opt

from specferry.data.checkpoint import sha256
from specferry.reference.tensors import compare, save_arrays
from specferry.validation.device import write_json

from .checkpoint import load_checkpoint
from .config import EPSILON, TOLERANCES


def source_record():
    path = inspect.getfile(modeling_opt)
    return {
        "path": path,
        "sha256": sha256(path),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }


def load_model(root):
    weights, metadata = load_checkpoint(root)
    config = OPTConfig(**metadata["config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = OPTForCausalLM(config)
    state = {f"model.{name}": tensor for name, tensor in weights.items()}
    state["lm_head.weight"] = weights["decoder.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    model.eval().requires_grad_(False)
    if model.lm_head.weight.data_ptr() != model.model.decoder.embed_tokens.weight.data_ptr():
        raise ValueError("OPT reference lost the shared embedding/head storage")
    if any(
        value.dtype != torch.float16 or value.device.type != "cpu" for value in model.parameters()
    ):
        raise ValueError("OPT reference requires original FP16 parameters on CPU")
    if any(
        layer.self_attn_layer_norm.eps != EPSILON or layer.final_layer_norm.eps != EPSILON
        for layer in model.model.decoder.layers
    ):
        raise ValueError("official OPT LayerNorm epsilon changed")
    return model, metadata


@contextmanager
def capture_layers(layers, captured):
    """Capture official boundaries; derive only the reference's explicit query scaling."""
    handles = []

    def before(key):
        def save(module, args):
            captured[key] = args[0].detach().clone()

        return save

    def after(key):
        def save(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[key] = value.detach().clone()
            if key.endswith(".mixer") and isinstance(output, tuple) and output[1] is not None:
                captured[key.removesuffix("mixer") + "probabilities"] = output[1].detach().clone()

        return save

    for index, layer in layers.items():
        prefix = f"layer.{index}."
        handles.append(layer.register_forward_pre_hook(before(prefix + "input")))
        handles.append(layer.register_forward_hook(after(prefix + "output")))
        for name, module in (
            ("mixer", layer.self_attn),
            ("attention_norm", layer.self_attn_layer_norm),
            ("fc1", layer.fc1),
            ("mlp", layer.fc2),
        ):
            handles.append(module.register_forward_hook(after(prefix + name)))
        handles.append(
            layer.self_attn_layer_norm.register_forward_pre_hook(
                before(prefix + "attention_residual")
            )
        )
        handles.append(
            layer.final_layer_norm.register_forward_pre_hook(before(prefix + "ffn_residual"))
        )
        handles.append(layer.fc2.register_forward_pre_hook(before(prefix + "activation")))
        for name in ("q", "k", "v"):
            handles.append(
                getattr(layer.self_attn, name + "_proj").register_forward_hook(after(prefix + name))
            )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    model, metadata = load_model(root)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    prompts = ("What are we having for dinner?", "The capital of France is", "Once upon a time,")
    checks, generations, trace = {}, [], {}
    selected = {i: model.model.decoder.layers[i] for i in (0, 1, 2, 3, 23)}
    for prompt_index, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        cache, cached_logits = None, []
        for step in range(ids.shape[1]):
            captured = {}
            with capture_layers(selected, captured):
                result = model(
                    input_ids=ids[:, step : step + 1], past_key_values=cache, use_cache=True
                )
            cache = result.past_key_values
            cached_logits.append(result.logits)
            for name, value in captured.items():
                trace[f"prompt.{prompt_index}.token.{step}.{name}"] = value
        full = model(input_ids=ids, use_cache=True)
        checks[f"prompt.{prompt_index}.logits"] = compare(
            torch.cat(cached_logits, dim=1), full.logits, **TOLERANCES["cpu"]
        )
        for index, (cached, whole) in enumerate(
            zip(cache.layers, full.past_key_values.layers, strict=True)
        ):
            for name in ("keys", "values"):
                checks[f"prompt.{prompt_index}.layer.{index}.{name}"] = compare(
                    getattr(cached, name), getattr(whole, name), **TOLERANCES["cpu"]
                )
        generated = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Teacher forcing compares cached decode with a fresh full-prefix forward.
        sequence, decode_cache = ids, full.past_key_values
        for index, token in enumerate(generated[0, ids.shape[1] : ids.shape[1] + 4]):
            sequence = torch.cat((sequence, token.reshape(1, 1)), dim=1)
            decoded = model(
                input_ids=token.reshape(1, 1), past_key_values=decode_cache, use_cache=True
            )
            decode_cache = decoded.past_key_values
            fresh = model(input_ids=sequence, use_cache=False)
            checks[f"prompt.{prompt_index}.decode.{index}"] = compare(
                decoded.logits[:, -1], fresh.logits[:, -1], **TOLERANCES["cpu"]
            )
        text = tokenizer.decode(generated[0, ids.shape[1] :], skip_special_tokens=True)
        if not text.strip() or "\ufffd" in text:
            raise ValueError("OPT generation produced empty or invalid text")
        generations.append(
            {
                "prompt": prompt,
                "input_ids": ids[0].tolist(),
                "output_ids": generated[0, ids.shape[1] :].tolist(),
                "text": text,
            }
        )
    save_arrays(output / "trace.npz", trace)
    write_json(output / "inventory.json", metadata)
    report = {
        "status": "passed" if all(item["pass"] for item in checks.values()) else "failed",
        "checks": checks,
        "generations": generations,
        "reference": source_record(),
        "tolerances": TOLERANCES,
        "weights": "native FP16, unchanged",
        "device": "host-cpu",
        "embedding_head_share_storage": True,
        "hardware_execution_proven": False,
        "trace_sha256": sha256(output / "trace.npz"),
    }
    write_json(output / "reference.json", report)
    return report
