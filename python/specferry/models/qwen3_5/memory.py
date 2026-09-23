"""Historical Qwen full-model planning resources, not measured board occupancy."""

from specferry.export.memory import memory_budget as summarize_memory

from .components import BASELINE


def resources(records, capacity=512, duplicate_head=False):
    if not 1 <= capacity <= 512:
        raise ValueError("initial context capacity must be in [1, 512]")
    embedding = next(record for record in records if record["name"] == "model.embed_tokens.weight")
    config = BASELINE
    delta_layers = config.layer_types.count("delta")
    attention_layers = config.layer_types.count("attention")
    return {
        "recurrent_state_double_buffer": 2
        * delta_layers
        * config.delta_heads
        * config.key_dim
        * config.value_dim
        * 4,
        "conv_state_double_buffer": 2
        * delta_layers
        * config.channels
        * config.convolution_width
        * 2,
        "compact_kv_cache": attention_layers * capacity * config.kv_slot_bytes,
        "rope_cos_sin_fp32": 2 * capacity * config.rotary_dim * 4,
        "position_ids_int32": capacity * 4,
        "attention_mask_fp32": capacity * 4,
        "head_block_logits_fp16": 4096 * 2,
        "head_block_winners": ((248320 + 4095) // 4096) * (4 + 4),
        "optional_second_embedding_layout": embedding["bytes"] if duplicate_head else 0,
    }


def memory_budget(
    records,
    capacity=512,
    pool_bytes=None,
    sdk_overhead_bytes=None,
    workspace_bytes=None,
    duplicate_head=False,
):
    return summarize_memory(
        records,
        resources(records, capacity, duplicate_head),
        capacity=capacity,
        pool_bytes=pool_bytes,
        sdk_overhead_bytes=sdk_overhead_bytes,
        workspace_bytes=workspace_bytes,
    )
