"""Tensor snapshots and numerical comparisons, independent of model architecture."""

import numpy as np
import torch


def tensors(value, prefix):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from tensors(item, f"{prefix}.{i}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f"{prefix}.{key}")


def finite(value, name):
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"NaN/Inf: {name}, shape={list(value.shape)}, dtype={value.dtype}")


def cache_tensors(cache, layers=None):
    result = {}
    for i, layer in enumerate(cache.layers):
        if layers is not None and i not in layers:
            continue
        for field in ("keys", "values", "conv_states", "recurrent_states"):
            for name, value in tensors(getattr(layer, field, None), f"layer.{i}.{field}"):
                finite(value, name)
                result[name] = value.detach().clone()
    return result


def tensor_info(value):
    finite(value, "tensor_info")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "max_abs": value.float().abs().max().item() if value.numel() else 0.0,
    }


def save_arrays(path, values):
    arrays = {}
    for key, value in values.items():
        finite(value, key)
        arrays[key] = value.detach().cpu().numpy()
    np.savez(path, **arrays)


def compare(actual, expected, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("comparison shape/dtype mismatch")
    finite(actual, "comparison.actual")
    finite(expected, "comparison.expected")
    a, b = actual.float(), expected.float()
    difference = (a - b).abs()
    bad = difference > atol + rtol * b.abs()
    return {
        "pass": not bad.any().item(),
        "shape": list(a.shape),
        "dtype": str(actual.dtype),
        "max_abs_error": difference.max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "mismatches": bad.sum().item(),
        "elements": a.numel(),
        "atol": atol,
        "rtol": rtol,
    }
