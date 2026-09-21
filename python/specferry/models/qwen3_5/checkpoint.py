"""Identity and text/vision/MTP mapping for the retained Qwen checkpoint."""

from pathlib import Path

from specferry.data.checkpoint import inventory as inspect_safetensors
from specferry.data.checkpoint import local_path, sha256

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def inventory(root: Path):
    result = inspect_safetensors(root, repo_id=MODEL_ID, revision=REVISION)
    groups = {
        name: {"tensors": 0, "parameters": 0, "bytes": 0} for name in ("text", "vision", "mtp")
    }
    for tensor in result["tensors"]:
        name = tensor["source_name"]
        if name.startswith("model.language_model."):
            group, mapped = "text", "model." + name.removeprefix("model.language_model.")
        elif name.startswith("model.visual."):
            group, mapped = "vision", None
        elif name.startswith("mtp."):
            group, mapped = "mtp", None
        else:
            raise ValueError(f"unclassified checkpoint tensor: {name}")
        tensor.update(group=group, text_name=mapped)
        groups[group]["tensors"] += 1
        for field in ("parameters", "bytes"):
            groups[group][field] += tensor[field]
    config = result.pop("config")
    result.update(
        groups=groups,
        text_config=config["text_config"],
        tied_embedding_head=config["text_config"].get(
            "tie_word_embeddings", config.get("tie_word_embeddings")
        ),
    )
    return result


__all__ = ["MODEL_ID", "REVISION", "inventory", "local_path", "sha256"]
