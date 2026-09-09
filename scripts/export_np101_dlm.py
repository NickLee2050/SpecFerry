#!/usr/bin/env python3
"""Export the fixed text checkpoint and record a provisional NP101 memory budget."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from specferry.export.memory import memory_budget
from specferry.export.schema import validate_text_entries
from specferry.export.weights import verify_export, write_native_index, write_weight_pack
from specferry.reference.checkpoint import MODEL_ID, REVISION, inventory, sha256
from specferry.validation.device import write_json

ROOT = Path(__file__).resolve().parents[1]


def export(args) -> dict:
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; use --verify-only or a new directory")
    metadata = inventory(args.model.resolve())
    entries = validate_text_entries(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        records = write_weight_pack(args.model.resolve(), entries, staging, args.chunk_bytes)
        write_native_index(staging, records)
        budget = memory_budget(
            records,
            args.context_capacity,
            args.pool_bytes,
            args.sdk_overhead_bytes,
            args.workspace_bytes,
        )
        capability = None
        if args.capabilities:
            capability = {
                "path": str(args.capabilities.resolve()),
                "sha256": sha256(args.capabilities),
                "status": json.loads(args.capabilities.read_text()).get("status"),
            }
        manifest = {
            "format": "specferry-np101-weights-v1",
            "status": "complete",
            "repo_id": MODEL_ID,
            "revision": REVISION,
            "source_manifest_sha256": metadata["download_manifest_sha256"],
            "text_config": metadata["text_config"],
            "tensors": records,
            "aliases": {"lm_head.weight": "model.embed_tokens.weight"},
            "files": {
                name: {"bytes": (staging / name).stat().st_size, "sha256": sha256(staging / name)}
                for name in ("weights.bin", "weights.index")
            },
            "precision_policy": "BF16 weights -> FP16; native FP32 retained; no norm folding",
            "source_groups": metadata["groups"],
            "memory_budget": budget,
            "capability_report": capability,
            "deployment_ready": False,
            "deployment_blockers": [
                "operator_hardware_acceptance",
                "measured_memory_fit",
                "device_weight_layout_validation",
            ],
        }
        write_json(staging / "deployment-manifest.json", manifest)
        write_json(staging / "memory-budget.json", budget)
        verification = verify_export(staging)
        # Publish the directory only after complete pack and per-tensor validation.
        staging.rename(output)
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/models/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/np101/Qwen3.5-0.8B")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--capabilities", type=Path)
    parser.add_argument("--context-capacity", type=int, default=512)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument(
        "--pool-bytes", type=int, help="reported pool size; not measured free memory"
    )
    parser.add_argument(
        "--sdk-overhead-bytes", type=int, help="explicit estimate, not a measurement"
    )
    parser.add_argument("--workspace-bytes", type=int, help="explicit estimate, not a measurement")
    args = parser.parse_args()
    try:
        result = verify_export(args.output) if args.verify_only else export(args)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, FloatingPointError) as error:
        print(f"weight export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
