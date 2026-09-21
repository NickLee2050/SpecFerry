"""Capture real intermediate tensors without changing model arithmetic."""

from contextlib import contextmanager

from transformers.models.qwen3_5 import modeling_qwen3_5 as impl

from specferry.reference.tensors import (
    cache_tensors,
    compare,
    finite,
    save_arrays,
    tensor_info,
    tensors,
)


class Capture:
    def __init__(self, layers=(0, 3)):
        self.layers = tuple(layers)
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
        for i in self.layers:
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
                active = self.active_layer in self.layers
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


__all__ = ["Capture", "cache_tensors", "compare", "finite", "save_arrays", "tensor_info", "tensors"]
