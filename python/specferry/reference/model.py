"""Compatibility entry for the retained Qwen model."""

from specferry.models.qwen3_5.model import encode, load_text_model, load_tokenizer, source_record

__all__ = ["encode", "load_text_model", "load_tokenizer", "source_record"]
