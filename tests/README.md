# Validation commands

Run from the repository root with `conda activate SpecFerry`. Use a new output
folder each time. Host unit tests and all `--prepare-only` commands do not open NP101.
Board commands serialize through one lock; timeout, signal or lingering children
leave a recovery marker. Do not delete that marker to force a retry.

## Small board checks

```bash
python scripts/check_np101.py selection --output .cache/runs/selection
python scripts/check_np101.py conv --output .cache/runs/conv
python scripts/check_np101.py kv --output .cache/runs/kv
python scripts/check_np101.py cache --output .cache/runs/cache
python scripts/check_np101.py cache-block --output .cache/runs/cache-block
python scripts/check_np101.py sampling --output .cache/runs/sampling
```

`selection` covers embedding/head boundary indices and ties using synthetic weights.
`kv` checks the component path's cache; `cache` and `cache-block` check the experimental
2D indexed append, attention, masking and reset. Sampling checks the promoted FP32
path, not the known-bad direct FP16 operator. Native executables remain in `build/tests`.
A numerical pass does not prove exclusive NPU execution or physical device residency.

## OPT and Qwen references

```bash
python scripts/check_np101.py opt --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 --steps 2 --capacity 16 --prepare-only --output .cache/runs/opt-prepared
python scripts/check_np101.py opt --fixture .cache/runs/opt-prepared/fixture \
  --output .cache/runs/opt-slice
python scripts/check_np101.py teacher --layer-count 24 --steps 2 --capacity 16 \
  --prepare-only --output .cache/runs/teacher-prepared
python scripts/check_np101.py teacher --fixture .cache/runs/teacher-prepared/fixture \
  --timeout 600 --output .cache/runs/teacher

python scripts/export_np101_dlm.py --verify-only
python scripts/reference_dlm.py --output .cache/runs/qwen-cpu
python scripts/check_np101.py qwen --trace .cache/runs/qwen-cpu/layer-0-3-sequential.npz \
  --layers 0 1 2 3 --steps 2 --prepare-only --output .cache/runs/qwen-prepared
python scripts/check_np101.py qwen --fixture .cache/runs/qwen-prepared/fixture \
  --output .cache/runs/qwen-slice
```

OPT `teacher` compares embedding, decoder outputs, logits and every valid KV prefix;
repeated reset/fresh trajectories must match exactly. Qwen retains FP16 projection
and FP32 recurrent-state reference policies. No Qwen full-model deployment is claimed.

## Experimental graph diagnosis

```bash
python scripts/check_np101.py lookup --output .cache/runs/lookup
python scripts/check_np101.py layer --fixture .cache/runs/opt-prepared/fixture \
  --output .cache/runs/layer
python scripts/check_np101.py prefix --fixture .cache/runs/teacher-prepared/fixture \
  --layer-count 1 --timeout 360 --output .cache/runs/prefix
python scripts/generate_opt.py --backend graph --block 1 --capacity 16 \
  --max-new-tokens 4 --output .cache/runs/graph-decode
```

Run one case at a time after device recovery. `layer` requires layer 0 with at least
two reference steps. For real-weight lookup, prepare `io --prepare-only`, then pass
its fixture to `lookup --fixture`. Partial-prefix predictions are diagnostic, not meaningful text.
The production path no longer performs automatic weight readback while compiling;
lookup/layer tests explicitly verify their shared weight banks after execution.

Add `--sdk-timing` to a compute command for public API begin/end records in
`sdk-calls.tsv`. Add `--trace-driver` to `check_np101.py` for raw `driver.strace`
(`strace` must be installed). These are diagnostics, not throughput measurements.
Failure details and native progress go to `device/sdk.log`.

## Memory diagnostics

```bash
python scripts/check_np101_memory.py capacity --mib 64 --storage constant --dtype F16 \
  --readback none --output .cache/runs/capacity
python scripts/check_np101_memory.py capacity --mib 1024 --storage mutable --dtype F32 \
  --readback all --output .cache/runs/integrity
python scripts/check_np101_memory.py weights --deployment .cache/np101/opt-350m \
  --storage constant --output .cache/runs/weights
python scripts/check_np101_memory.py growth --mode advance --iterations 128 \
  --output .cache/runs/host-growth
python scripts/check_np101_memory.py accounting --mib 8 --storage mutable \
  --output .cache/runs/accounting
```

Application const/nonconst tensor payloads are independently capped at 1 GiB.
`--readback none` measures allocation/release only; `all` scans every retained block,
including those after corruption. `readback-blocks.jsonl` records logical byte/page
ranges, not physical addresses. Corruption is a nonzero integrity result even when
allocation succeeds. `--storage` and `--dtype` are explicit, never fallback policies.

Compare `growth --mode fixed`, `same`, and `advance` to isolate revalidation growth.
`accounting` reports SDK counters, not proven physical allocation. Both print their
native tables. For model-independent state allocation alongside weights, add
`--state-spec tests/fixtures/qwen3_5_allocation_states.txt`; it describes shapes only.

## Entry-point consolidation

Former `check_np101_{opt,decoder,generation,graph_pipeline,graph_cache,sampling}.py`
commands are now explicit cases of `check_np101.py`. Former allocation/capacity
commands are `check_np101_memory.py weights/capacity`. Acceptance timing is
`generate_opt.py --warmups ... --repeats ... --compare-cpu`; the experimental graph
uses `--backend graph`. Historical reports retain their original schemas and commands
in Git/history; they are not inputs to the new runner.
