"""Compatibility entry for the retained Qwen runner."""

from specferry.models.qwen3_5.runner import (
    TOLERANCES,
    alignment,
    forward,
    generate,
    precision,
    run,
    teacher_forcing,
)

__all__ = ["TOLERANCES", "alignment", "forward", "generate", "precision", "run", "teacher_forcing"]
