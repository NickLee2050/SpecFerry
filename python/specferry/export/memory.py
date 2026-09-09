"""Known memory lower bound and explicit unresolved device allocation costs."""


def memory_budget(
    records: list[dict],
    capacity=512,
    pool_bytes=None,
    sdk_overhead_bytes=None,
    workspace_bytes=None,
    duplicate_head=False,
) -> dict:
    if not 1 <= capacity <= 512:
        raise ValueError("initial context capacity must be in [1, 512]")
    for value in (pool_bytes, sdk_overhead_bytes, workspace_bytes):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("memory counts must be nonnegative integers")
    weight_bytes = sum(record["bytes"] for record in records)
    embedding = next(record for record in records if record["name"] == "model.embed_tokens.weight")
    known = {
        "text_weights": weight_bytes,
        "recurrent_state_double_buffer": 2 * 18 * 16 * 128 * 128 * 4,
        "conv_state_double_buffer": 2 * 18 * 6144 * 4 * 2,
        "compact_kv_cache": 6 * 2 * capacity * 2 * 256 * 2,
        "rope_cos_sin_fp32": 2 * capacity * 64 * 4,
        "position_ids_int32": capacity * 4,
        "attention_mask_fp32": capacity * 4,
        "head_block_logits_fp16": 4096 * 2,
        "head_block_winners": ((248320 + 4095) // 4096) * (4 + 4),
        "optional_second_embedding_layout": embedding["bytes"] if duplicate_head else 0,
    }
    unknown = []
    estimates = {}
    for name, value in (
        ("sdk_packing_and_copies", sdk_overhead_bytes),
        ("activations_graphs_and_workspace", workspace_bytes),
    ):
        if value is None:
            unknown.append(name)
        else:
            estimates[name] = value
    lower_bound = sum(known.values())
    total = lower_bound + sum(estimates.values())
    over_pool = pool_bytes is not None and total > pool_bytes
    return {
        "status": "exceeds_pool" if over_pool else "unverified",
        "context_capacity": capacity,
        "known_bytes": known,
        "estimated_bytes": estimates,
        "unknown_components": unknown,
        "known_lower_bound_bytes": lower_bound,
        "total_with_estimates_bytes": total,
        "reported_pool_bytes": pool_bytes,
        "headroom_against_pool_bytes": None if pool_bytes is None else pool_bytes - total,
        "pool_is_free_memory": False,
        "actual_device_allocation_verified": False,
        "device_residency_verified": False,
        "notes": [
            "Allocation and graph verification must measure SDK copies and workspace.",
            "An SDK tensor allocation does not establish physical device residency.",
            "The tied embedding/head is counted once; alternate layouts must be explicit.",
            "No per-token host weight streaming is included in this design.",
        ],
    }
