"""Compatibility entry for the retained Qwen checkpoint inspector."""

from specferry.models.qwen3_5.checkpoint import MODEL_ID, REVISION, inventory, local_path, sha256

__all__ = ["MODEL_ID", "REVISION", "inventory", "local_path", "sha256"]
