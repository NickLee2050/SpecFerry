"""Strict text-only loading and explicit CPU precision policies."""

import inspect
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

from .checkpoint import sha256


def load_text_model(root: Path, metadata: dict, mode: str):
    if mode not in ("original-fp32", "deployment-fp16"):
        raise ValueError(f"unknown reference mode: {mode}")
    config = Qwen3_5TextConfig(**metadata["text_config"])
    config._attn_implementation = "eager"
    config.use_cache = True
    if not config.tie_word_embeddings:
        raise ValueError("first DLM must have tied embedding and LM head")
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    state = {}
    for filename in sorted({t["file"] for t in metadata["tensors"] if t["group"] == "text"}):
        with safe_open(root / filename, framework="pt", device="cpu") as weights:
            for entry in metadata["tensors"]:
                if entry["group"] != "text" or entry["file"] != filename:
                    continue
                tensor = weights.get_tensor(entry["source_name"])
                target_dtype = (
                    torch.float32
                    if mode == "original-fp32" or entry["dtype"] == "F32"
                    else torch.float16
                )
                tensor = tensor.to(target_dtype)
                if list(tensor.shape) != entry["shape"] or not torch.isfinite(tensor).all():
                    raise ValueError(f"invalid text tensor: {entry['source_name']}")
                state[entry["text_name"]] = tensor
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    # Meta construction also creates nonpersistent RoPE buffers; materialize them
    # through the exact installed reference class rather than approximating mRoPE.
    with torch.device("cpu"):
        model.model.rotary_emb = implementation.Qwen3_5TextRotaryEmbedding(config)
    if any(t.device.type != "cpu" for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("reference model contains non-CPU or unmaterialized tensors")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("LM head and embedding are not tied")
    model.eval().requires_grad_(False)
    return model


def load_tokenizer(root):
    return AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)


def encode(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if not rendered.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("unexpected non-thinking chat-template suffix; inspect tokenizer contract")
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return rendered, torch.tensor([ids], dtype=torch.long)


def source_record():
    source = Path(inspect.getfile(implementation))
    return {"path": str(source), "sha256": sha256(source)}
