"""Known memory lower bound and explicit unresolved device allocation costs."""


def memory_budget(
    records: list[dict],
    resources: dict[str, int],
    *,
    capacity: int,
    pool_bytes=None,
    sdk_overhead_bytes=None,
    workspace_bytes=None,
) -> dict:
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError("context capacity must be positive")
    for value in (pool_bytes, sdk_overhead_bytes, workspace_bytes):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("memory counts must be nonnegative integers")
    if any(type(value) is not int or value < 0 for value in resources.values()):
        raise ValueError("resource bytes must be nonnegative integers")
    if any(type(record["bytes"]) is not int or record["bytes"] < 0 for record in records):
        raise ValueError("weight bytes must be nonnegative integers")
    if "text_weights" in resources:
        raise ValueError("weight payload is counted separately from resource buffers")
    known = {"text_weights": sum(record["bytes"] for record in records), **resources}
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
            "Physical weight records are counted once; additional copies must be explicit.",
        ],
    }
