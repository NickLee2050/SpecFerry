"""Compatibility entry for the retained Qwen trace."""

from specferry.models.qwen3_5.trace import (
    Capture,
    cache_tensors,
    compare,
    finite,
    save_arrays,
    tensor_info,
    tensors,
)

__all__ = ["Capture", "cache_tensors", "compare", "finite", "save_arrays", "tensor_info", "tensors"]
