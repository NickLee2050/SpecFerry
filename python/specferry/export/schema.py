"""Exact parameter contract for the fixed Qwen3.5-0.8B text checkpoint."""


def expected_text_tensors(config: dict) -> dict:
    required = {
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "vocab_size": 248320,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("text configuration differs from the fixed first DLM")
    layers = ["full_attention" if layer % 4 == 3 else "linear_attention" for layer in range(24)]
    if config.get("layer_types") != layers or not config.get("tie_word_embeddings"):
        raise ValueError("unexpected layer types or untied embedding/head")
    tensors = {
        "model.embed_tokens.weight": ([248320, 1024], "BF16"),
        "model.norm.weight": ([1024], "BF16"),
    }
    for layer, kind in enumerate(layers):
        prefix = f"model.layers.{layer}."
        entries = {
            "input_layernorm.weight": ([1024], "BF16"),
            "post_attention_layernorm.weight": ([1024], "BF16"),
            "mlp.gate_proj.weight": ([3584, 1024], "BF16"),
            "mlp.up_proj.weight": ([3584, 1024], "BF16"),
            "mlp.down_proj.weight": ([1024, 3584], "BF16"),
        }
        if kind == "linear_attention":
            entries.update(
                {
                    "linear_attn.in_proj_qkv.weight": ([6144, 1024], "BF16"),
                    "linear_attn.in_proj_z.weight": ([2048, 1024], "BF16"),
                    "linear_attn.in_proj_a.weight": ([16, 1024], "BF16"),
                    "linear_attn.in_proj_b.weight": ([16, 1024], "BF16"),
                    "linear_attn.out_proj.weight": ([1024, 2048], "BF16"),
                    "linear_attn.conv1d.weight": ([6144, 1, 4], "BF16"),
                    "linear_attn.dt_bias": ([16], "BF16"),
                    "linear_attn.A_log": ([16], "F32"),
                    "linear_attn.norm.weight": ([128], "F32"),
                }
            )
        else:
            entries.update(
                {
                    "self_attn.q_proj.weight": ([4096, 1024], "BF16"),
                    "self_attn.k_proj.weight": ([512, 1024], "BF16"),
                    "self_attn.v_proj.weight": ([512, 1024], "BF16"),
                    "self_attn.o_proj.weight": ([1024, 2048], "BF16"),
                    "self_attn.q_norm.weight": ([256], "BF16"),
                    "self_attn.k_norm.weight": ([256], "BF16"),
                }
            )
        tensors.update({prefix + name: value for name, value in entries.items()})
    return tensors


def validate_text_entries(inventory: dict) -> list[dict]:
    expected = expected_text_tensors(inventory["text_config"])
    entries = [entry for entry in inventory["tensors"] if entry["group"] == "text"]
    names = [entry["text_name"] for entry in entries]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("text tensor names do not match the complete 24-layer contract")
    for entry in entries:
        shape, dtype = expected[entry["text_name"]]
        if entry["shape"] != shape or entry["dtype"] != dtype:
            raise ValueError(f"text tensor shape/dtype mismatch: {entry['text_name']}")
    return sorted(entries, key=lambda entry: entry["text_name"])
