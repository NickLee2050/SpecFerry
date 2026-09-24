#!/usr/bin/env python3
"""Download pinned, allowlisted checkpoints with the Hub's resume support."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from specferry.data.checkpoint import sha256 as sha256_file
from specferry.validation.device import write_json


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str = "main"


# Only the first DLM is enabled. Uncomment individual TLMs when their phase starts.
MODELS = (
    ModelSpec("facebook/opt-350m", "08ab08cc4b72ff5593870b5d527cf4230323703c"),
    # ModelSpec("facebook/opt-13b"),  # Prospective TLM; enable only after selection.
    # ModelSpec("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    # ModelSpec("Qwen/Qwen3.5-4B"),
    # ModelSpec("Qwen/Qwen3.5-9B"),
    # ModelSpec("Qwen/Qwen3.5-27B"),
    # ModelSpec("Qwen/Qwen3.5-35B-A3B"),
)
ALLOWED_REPOS = frozenset(
    {"facebook/opt-350m", "facebook/opt-13b"}
    | {f"Qwen/Qwen3.5-{size}" for size in ("0.8B", "4B", "9B", "27B", "35B-A3B")}
)
FILE_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "LICENSE",
    "LICENSE.*",
    "README.md",
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / ".cache" / "models"


def validate_models(models: tuple[ModelSpec, ...]) -> None:
    if not models:
        raise ValueError("No models are enabled in MODELS.")
    seen = set()
    for model in models:
        if "llama" in model.repo_id.casefold():
            raise ValueError(f"Llama models are excluded by project policy: {model.repo_id}")
        if model.repo_id not in ALLOWED_REPOS:
            raise ValueError(f"Model is outside the download allowlist: {model.repo_id}")
        if model.repo_id in seen:
            raise ValueError(f"Duplicate model: {model.repo_id}")
        if not model.revision:
            raise ValueError(f"Missing revision: {model.repo_id}")
        seen.add(model.repo_id)


def expected_files(info) -> list[dict]:
    # Prefer safetensors when available; older official OPT releases provide only
    # PyTorch weights. Select one format rather than downloading duplicate weights.
    safe_weights = {
        item.rfilename for item in info.siblings if item.rfilename.endswith(".safetensors")
    }
    pytorch_weights = {
        item.rfilename for item in info.siblings if fnmatch(item.rfilename, "pytorch_model*.bin")
    }
    weight_files = safe_weights or pytorch_weights
    excluded_index = (
        "pytorch_model.bin.index.json" if safe_weights else "model.safetensors.index.json"
    )
    files = []
    for item in info.siblings:
        name = item.rfilename
        if name == excluded_index:
            continue
        if name not in weight_files and not any(
            fnmatch(name, pattern) for pattern in FILE_PATTERNS
        ):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError(f"Unsafe repository path: {name}")
        lfs = getattr(item, "lfs", None)
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files.append({"path": name, "size": getattr(item, "size", None), "sha256": digest})
    names = {item["path"] for item in files}
    if "config.json" not in names or not weight_files:
        raise ValueError("The repository does not contain config.json and supported weights.")
    if "tokenizer.json" not in names and not {"vocab.json", "merges.txt"} <= names:
        raise ValueError("The repository requires tokenizer.json or vocab.json with merges.txt.")
    return sorted(files, key=lambda item: item["path"])


def verify_files(destination: Path, files: list[dict]) -> None:
    names = {item["path"] for item in files}
    for item in files:
        path = destination / item["path"]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty download: {path}")
        if item["size"] is not None and path.stat().st_size != item["size"]:
            raise ValueError(f"Download size mismatch: {path}")
        if item["sha256"] and sha256_file(path) != item["sha256"]:
            raise ValueError(f"Download SHA256 mismatch: {path}")
        if item["path"].endswith((".safetensors.index.json", ".bin.index.json")):
            index = json.loads(path.read_text(encoding="utf-8"))
            if not set(index["weight_map"].values()).issubset(names):
                raise ValueError(f"Weight index refers to an unselected shard: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--http-only", action="store_true")
    args = parser.parse_args()
    validate_models(MODELS)
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.http_only:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, snapshot_download

    for model in MODELS:
        destination = args.output / model.repo_id
        print(f"{model.repo_id}@{model.revision} -> {destination}")
        if args.dry_run:
            continue
        info = HfApi().model_info(model.repo_id, revision=model.revision, files_metadata=True)
        files = expected_files(info)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / "download-manifest.json"
        manifest.unlink(missing_ok=True)
        snapshot_download(
            model.repo_id,
            revision=info.sha,
            local_dir=destination,
            allow_patterns=[entry["path"] for entry in files],
            max_workers=args.workers,
        )
        verify_files(destination, files)
        write_json(
            manifest,
            {
                "status": "complete",
                "repo_id": model.repo_id,
                "resolved_revision": info.sha,
                "files": files,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
