"""Prepare the official Qwen FP16 regression trace; no NP101 access."""

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as impl

from specferry.reference.tensors import finite
from specferry.validation.device import write_json as write

from .checkpoint import inventory
from .model import encode, load_text_model, load_tokenizer
from .precision import TOLERANCES, precision
from .trace import Capture, cache_tensors, compare, save_arrays, tensor_info


def forward(model, ids, cache=None, keep=1):
    result = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=keep)
    finite(result.logits, "logits")
    return result


def alignment(model, ids, directory, mode, capture_intermediates=True):
    checkpoints = (1, 2, 4, 8)
    capture = Capture()
    cache, seq, seq_logits = None, {}, []
    from contextlib import nullcontext

    with capture.attach(model) if capture_intermediates else nullcontext():
        for i in range(8):
            capture.step = i + 1
            result = forward(model, ids[:, i : i + 1], cache)
            cache = result.past_key_values
            seq_logits.append(result.logits.detach().clone())
            if i + 1 in checkpoints:
                seq[i + 1] = cache_tensors(cache)
                if capture_intermediates:
                    capture.save(cache_tensors(cache, (0, 3)), "cache")
    if capture_intermediates:
        save_arrays(directory / "layer-0-3-sequential.npz", capture.values)
        write(
            directory / "trace-index.json", {k: tensor_info(v) for k, v in capture.values.items()}
        )
        fixed = fixed_input_recurrence(capture.values)
        write(directory / "fixed-input-recurrence.json", fixed)
        if not all(x["state"]["pass"] and x["output"]["pass"] for x in fixed):
            raise ValueError("fixed-input recurrent reference comparison failed")
    results = []
    for count in checkpoints:
        # No custom prefill: invoke the unmodified official model forward.
        output = forward(model, ids[:, :count], keep=0)
        normal = cache_tensors(output.past_key_values)
        if normal.keys() != seq[count].keys():
            raise ValueError("prefill and sequential paths expose different cache fields")
        reference_logits = torch.cat(seq_logits[:count], dim=1)
        budget = TOLERANCES["fp16_pipeline" if mode == "deployment-fp16" else "fp32_pipeline"]
        logit_result = compare(output.logits, reference_logits, **budget)
        states = {}
        for name, value in normal.items():
            state_budget = (
                TOLERANCES["fp32_state_from_fp16_pipeline"]
                if mode == "deployment-fp16" and value.dtype == torch.float32
                else budget
            )
            states[name] = compare(value, seq[count][name], **state_budget)
        save_arrays(directory / f"prefill-{count}-states.npz", normal)
        save_arrays(directory / f"sequential-{count}-states.npz", seq[count])
        results.append(
            {
                "tokens": count,
                "logits": logit_result,
                "states": states,
                "pass": logit_result["pass"] and all(s["pass"] for s in states.values()),
            }
        )
    write(directory / "prefill-alignment.json", results)
    return results


def fixed_input_recurrence(values):
    """Check the documented recurrence against real official function IO.

    Each case receives identical saved Q/K/V, gates, and previous state, so it
    must meet the small FP32 operator budget even in an FP16 model pipeline.
    """
    results = []
    for step in (2, 4, 8):
        prefix = f"token.{step:04d}.layer.0.torch_recurrent_gated_delta_rule"
        q, k, v = (values[f"{prefix}.input.{i}"] for i in range(3))
        q, k = impl.l2norm(q), impl.l2norm(k)
        q, k, v = (x[:, 0].float() for x in (q, k, v))
        q = q / q.shape[-1] ** 0.5
        decay = values[f"{prefix}.kwargs.g"][:, 0].float().exp()
        beta = values[f"{prefix}.kwargs.beta"][:, 0].float()
        previous = values[f"{prefix}.kwargs.initial_state"]
        decayed = previous.float() * decay[..., None, None]
        correction = beta[..., None] * (v - torch.einsum("bhk,bhkv->bhv", k, decayed))
        state = decayed + torch.einsum("bhk,bhv->bhkv", k, correction)
        expected_output = values[f"{prefix}.output.0"]
        out = torch.einsum("bhk,bhkv->bhv", q, state)[:, None].to(expected_output.dtype)
        results.append(
            {
                "consumed_tokens": step,
                "state": compare(
                    state, values[f"{prefix}.output.1"], **TOLERANCES["fixed_input_fp32_state"]
                ),
                "output": compare(out, expected_output, **TOLERANCES["fixed_input_fp16_output"]),
            }
        )
    return results


@torch.inference_mode()
def run(root, output, threads=4):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    model = load_text_model(root, inventory(root), "deployment-fp16")
    tokenizer = load_tokenizer(root)
    _, ids = encode(tokenizer, "中国的首都是哪里？请只回答城市名。")
    with precision("deployment-fp16"):
        checks = alignment(model, ids, output, "deployment-fp16")
    passed = all(item["pass"] for item in checks)
    write(output / "reference.json", {"status": "passed" if passed else "failed", "checks": checks})
    return int(not passed)
