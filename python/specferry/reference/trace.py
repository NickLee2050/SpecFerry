"""Capture real intermediate tensors without changing model arithmetic."""

from contextlib import contextmanager

import numpy as np
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as impl


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


class Capture:
    def __init__(self):
        self.values = {}
        self.step = 0
        self.active_layer = None

    def save(self, value, name):
        for key, tensor in tensors(value, f"token.{self.step:04d}.{name}"):
            finite(tensor, key)
            original, call = key, 1
            while key in self.values:
                key = f"{original}.call{call}"
                call += 1
            self.values[key] = tensor.detach().clone()

    @contextmanager
    def attach(self, model):
        handles = []
        originals = {}
        for i in (0, 3):
            for name, module in model.model.layers[i].named_modules():
                label = f"layer.{i}" + ("." + name if name else "")

                def before(module, args, kwargs, label=label, i=i, name=name):
                    if not name:
                        self.active_layer = i
                    self.save(args, label + ".input")
                    # Masks intentionally contain -inf; only capture positional constants here.
                    self.save(kwargs.get("position_embeddings"), label + ".position_embeddings")

                def after(module, args, output, label=label, name=name):
                    self.save(output, label + ".output")
                    if not name:
                        self.active_layer = None

                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after))
        # These functional calls do not appear as child modules. Capture Q/K/V,
        # decay, beta, convolution and recurrent outputs at their true call sites.
        for name in (
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "l2norm",
            "apply_rotary_pos_emb",
        ):
            original = getattr(impl, name)
            originals[name] = original

            def wrapped(*args, _original=original, _name=name, **kwargs):
                label = f"layer.{self.active_layer}.{_name}"
                active = self.active_layer in (0, 3)
                if active:
                    self.save(args, label + ".input")
                    self.save(kwargs, label + ".kwargs")
                result = _original(*args, **kwargs)
                if active:
                    self.save(result, label + ".output")
                return result

            setattr(impl, name, wrapped)
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
            for name, original in originals.items():
                setattr(impl, name, original)


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
