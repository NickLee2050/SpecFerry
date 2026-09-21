"""Frozen Qwen reference error budgets and arithmetic policy."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as impl

TOLERANCES = json.loads(Path(__file__).with_name("tolerances.json").read_text())


@contextmanager
def precision(mode):
    original = impl.l2norm
    if mode == "deployment-fp16":

        def l2_fp32(x, dim=-1, eps=1e-6):
            value = x.float()
            return (value * torch.rsqrt(value.square().sum(dim=dim, keepdim=True) + eps)).to(
                x.dtype
            )

        impl.l2norm = l2_fp32
    try:
        yield
    finally:
        impl.l2norm = original
