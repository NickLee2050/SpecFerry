"""Small synthetic decoder fixtures exercising the production shared operators."""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as reference

from specferry.export.weights import write_native_index
from specferry.validation.arrays import save_tensor
from specferry.validation.device import write_json

from .components import Components
from .model import source_record
from .precision import TOLERANCES
from .validation_decoder import DecoderCache, reference_sequence


def prepare(directory: Path, steps=8):
    """Use official Qwen arithmetic with different dimensions, not device-derived expectations."""
    torch.manual_seed(701)
    config = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": True,
        },
    )
    config._attn_implementation = "eager"
    components = Components.from_text_config(config.to_dict(), capacity=8)
    selected = [0, 1]
    layers = {}
    records = []
    weights = directory / "weights"
    fixture = directory / "fixture"
    weights.mkdir(parents=True)
    fixture.mkdir()
    with (weights / "weights.bin").open("xb") as packed:
        for index in selected:
            layer = reference.Qwen3_5DecoderLayer(config, index)
            state = {}
            for name, tensor in layer.state_dict().items():
                dtype = (
                    torch.float32
                    if name.endswith(("A_log", "linear_attn.norm.weight"))
                    else torch.float16
                )
                # These fixtures do not train or modify the downloaded checkpoint.
                if name.endswith("A_log"):
                    value = torch.zeros_like(tensor, dtype=dtype)
                elif name.endswith("norm.weight") and "linear_attn" not in name:
                    value = torch.zeros_like(tensor, dtype=dtype)
                elif name == "linear_attn.norm.weight":
                    value = torch.ones_like(tensor, dtype=dtype)
                else:
                    value = (torch.randn(tensor.shape) * 0.04).to(dtype)
                state[name] = value
                raw = value.contiguous().view(torch.uint8).numpy().tobytes()
                packed.write(bytes((-packed.tell()) % 64))
                record = {
                    "name": f"model.layers.{index}.{name}",
                    "dtype": "F32" if dtype == torch.float32 else "F16",
                    "sdk_shape": list(reversed(value.shape)),
                    "offset": packed.tell(),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                packed.write(raw)
                records.append(record)
            layer.load_state_dict(state, strict=True, assign=True)
            layers[index] = layer.eval().requires_grad_(False)
    write_native_index(weights, records)
    components.write(fixture)
    (fixture / "layers.txt").write_text("0 1\n")
    inputs = [(torch.randn(1, 1, components.hidden) * 0.2).half() for _ in range(steps)]
    for index, hidden in enumerate(inputs):
        save_tensor(fixture / f"input.{index}.bin", hidden)
    trace = directory / "empty-trace.npz"
    np.savez(trace)
    rotary = reference.Qwen3_5TextRotaryEmbedding(config)
    cache = DecoderCache(components, selected)
    observations = []
    for name, count, final_only in (
        ("zero", steps, False),
        ("reset", min(8, steps), False),
        ("final", min(32, steps), True),
        ("fresh", min(8, steps), False),
    ):
        if name == "fresh":
            cache = DecoderCache(components, selected)
        else:
            cache.reset()
        values, _ = reference_sequence(
            layers, rotary, cache, inputs[:count], fixture, name, trace, final_only
        )
        observations.extend(values)
    metadata = {
        "kind": "synthetic_component_regression",
        "steps": steps,
        "first_layer": 0,
        "layers": selected,
        "component_config": asdict(components),
        "transfers": components.transfers(selected),
        "tolerances": TOLERANCES,
        "observations": observations,
        "reference": source_record(),
        "memory_payload": {"state_bytes": components.state_bytes(selected)},
        "device_deployment": False,
    }
    write_json(fixture / "reference.json", metadata)
    return metadata
