#!/usr/bin/env python3
"""Download the enabled Qwen3.5 checkpoints, with pinned revisions and resume support."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str = "main"


# Only the first DLM is enabled. Uncomment individual TLMs when their phase starts.
MODELS = (
    ModelSpec("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    # ModelSpec("Qwen/Qwen3.5-4B"),       # First two-machine experiment.
    # ModelSpec("Qwen/Qwen3.5-9B"),       # Medium dense experiment.
    # ModelSpec("Qwen/Qwen3.5-27B"),      # Large dense experiment.
    # ModelSpec("Qwen/Qwen3.5-35B-A3B"),  # MoE experiment; all experts are downloaded.
)
ALLOWED_REPOS = frozenset(
    f"Qwen/Qwen3.5-{size}" for size in ("0.8B", "4B", "9B", "27B", "35B-A3B")
)
FILE_PATTERNS = (
    "*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja",
    "LICENSE", "LICENSE.*", "README.md",
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
            raise ValueError(f"Model is outside the Qwen3.5 download allowlist: {model.repo_id}")
        if model.repo_id in seen:
            raise ValueError(f"Duplicate model: {model.repo_id}")
        if not model.revision:
            raise ValueError(f"Missing revision: {model.repo_id}")
        seen.add(model.repo_id)


def write_json(path: Path, value: dict) -> None:
    """Publish a complete JSON file; interrupted downloads cannot create a success marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(info) -> list[dict]:
    files = []
    for item in info.siblings:
        name = item.rfilename
        if not any(fnmatch(name, pattern) for pattern in FILE_PATTERNS):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError(f"Unsafe repository path: {name}")
        lfs = getattr(item, "lfs", None)
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files.append({"path": name, "size": getattr(item, "size", None), "sha256": digest})
    names = {item["path"] for item in files}
    if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
        raise ValueError("The repository does not contain config.json and safetensors weights.")
    if "tokenizer.json" not in names:
        raise ValueError("The repository does not contain tokenizer.json.")
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
        if item["path"].endswith(".safetensors.index.json"):
            index = json.loads(path.read_text(encoding="utf-8"))
            if not set(index["weight_map"].values()).issubset(names):
                raise ValueError(f"Weight index refers to an unselected shard: {path}")


def download_one(model: ModelSpec, output: Path, file_workers: int, range_workers: int = 1) -> Path:
    from huggingface_hub import HfApi, snapshot_download

    destination = output.joinpath(*model.repo_id.split("/"))
    lock_path = destination / "download-lock.json"
    manifest_path = destination / "download-manifest.json"
    revision = model.revision
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["repo_id"] != model.repo_id or lock["requested_revision"] != model.revision:
            raise ValueError(f"Download configuration changed; choose a new --output directory: {destination}")
        revision = lock["resolved_revision"]
    # Resolve once, then use the immutable SHA for both metadata and all file downloads.
    info = HfApi().model_info(model.repo_id, revision=revision, files_metadata=True)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise ValueError("The Hub did not return an immutable commit SHA.")
    files = expected_files(info)
    lock = {
        "repo_id": model.repo_id,
        "requested_revision": model.revision,
        "resolved_revision": info.sha,
    }
    write_json(lock_path, lock)
    # Revalidate every run; retain downloaded data so the Hub can resume/reuse it.
    manifest_path.unlink(missing_ok=True)
    print(f"Downloading {model.repo_id}@{info.sha} -> {destination}", flush=True)
    selected = [item["path"] for item in files]
    if range_workers > 1:
        from range_download import download_ranges
        from huggingface_hub import hf_hub_url
        for item in files:
            if item['path'].endswith('.safetensors') and item['sha256'] and item['size']:
                download_ranges(hf_hub_url(model.repo_id, item['path'], revision=info.sha),
                                destination / item['path'], item['size'], item['sha256'], range_workers)
                selected.remove(item['path'])
    snapshot_download(
        repo_id=model.repo_id, revision=info.sha, local_dir=destination,
        allow_patterns=selected, max_workers=file_workers,
    )
    verify_files(destination, files)
    write_json(manifest_path, {
        **lock, "status": "complete", "files": files,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "huggingface_hub_version": importlib.metadata.version("huggingface_hub"),
    })
    print(f"Verified {model.repo_id}: {len(files)} files", flush=True)
    return destination


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="download root (default: repository .cache/models)")
    parser.add_argument("--parallel-models", type=positive_int, default=1,
                        help="simultaneously download this many enabled models")
    parser.add_argument("--file-workers", type=positive_int, default=4,
                        help="concurrent file downloads per model")
    parser.add_argument("--http-only", action="store_true",
                        help="disable Xet transfers when that service is unreachable")
    parser.add_argument("--range-workers", type=positive_int, default=1,
                        help="parallel HTTP byte ranges per public weight file (default: disabled)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without dependencies, network requests, or file writes")
    args = parser.parse_args(argv)
    try:
        validate_models(MODELS)
    except ValueError as error:
        parser.error(str(error))
    output = args.output.expanduser().resolve()
    if args.dry_run:
        print(json.dumps({
            "output": str(output),
            "parallel_models": min(args.parallel_models, len(MODELS)),
            "file_workers_per_model": args.file_workers,
            "http_only": args.http_only,
            "range_workers": args.range_workers,
            "models": [{"repo_id": model.repo_id, "revision": model.revision} for model in MODELS],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.http_only:
        # The Hub reads this at import time, before any download workers start.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Install download dependencies: python -m pip install -r requirements-download.txt", file=sys.stderr)
        return 2
    failed = []
    with ThreadPoolExecutor(max_workers=min(args.parallel_models, len(MODELS))) as pool:
        pending = {pool.submit(download_one, model, output, args.file_workers, args.range_workers): model for model in MODELS}
        for future in as_completed(pending):
            model = pending[future]
            try:
                future.result()
            except Exception as error:
                failed.append(model.repo_id)
                print(f"FAILED {model.repo_id}: {error}", file=sys.stderr, flush=True)
    if failed:
        print("Incomplete downloads: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
