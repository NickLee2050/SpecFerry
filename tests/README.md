# Validation commands

Run from the repository root after [environment setup](../README.md). Current acceptance,
precision exceptions, performance and raw evidence live in [NP101 status](../docs/np101-status.md);
[TODO](../TODO.md) owns future work. Completed state/structural probes and historical report
bodies are recoverable from `.cache/runs/cleanup-20261001/before-source.tar.gz`.

## Device discipline

Each board command runs one bounded child process and uses a new output directory. Board
checks are excluded from ordinary CTest. Timeouts, signals and lingering children leave a
recovery marker: resolve the fault before another board command; do not delete it to force
retries. Stop on kernel hang/BUG or residual ownership. Before intentionally reproducing
P3/selection hang, cold-restart both host and NP101. Sudo needs explicit authorization.
CPU references and `--prepare-only` do not open NP101; model wrappers accept `--cpu-threads N`.

## Generation and performance

```bash
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --compare-cpu --output .cache/runs/opt-tokens
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --benchmark --warmups 1 --repeats 3 --sdk-summary \
  --output .cache/runs/opt-benchmark
```

Graph defaults to K/group1 and token-only readback. A/B/A or a JSON request matrix via
`--request-tokens-file FILE` shares a resident model; explicit `repeat_of` identifies repeats.
Block4 consumes four known consecutive tokens from one request, with block1 for tails/decode;
weight/KV storage and both execution sets remain resident. Block4 currently requires K,
group1, every-token and capacity divisible by four. Other graph candidates use block1.
Checkpoint defaults govern generation; sampling is explicit and supported by the component
FP32-logit path. Token-only success does not establish full logits/KV correctness.

Use `--cache-readback logits` for complete final logits, or
`--cache-readback kv --cache-layer-range 1 4` for K/V of one-based inclusive layers 1–4.
The range changes diagnostics only; all model layers execute. These are separate scopes,
without automatic readback expansion on failure. Full all-layer generation readback was
retired; iterate explicit KV ranges only when the acceptance scope requires it.

Benchmark repeats each original input on the same model. Warmups are checked but excluded
from statistics; `benchmark.json` maps names to input/repeat, and samples retain actual EOS,
capacity stop, output and consumed counts. Benchmark requires token-only readback and is
incompatible with lifecycle diagnostics.

`--sdk-summary` and `--sdk-timing` are mutually exclusive. Summary aggregates steady-clock
spans by request/phase/graph-role/API in memory and writes `sdk-summary.tsv` after release;
no per-call disk flush. `nested_count` marks overlap; `completed` is normal C++ return,
with SDK status still checked by its caller. Native mode is `SPECFERRY_SDK_TIMING=summary`.
Primary `graph_metrics` sums outer RunGraph plus independent ProcessGraph, including SDK
completion waits but excluding initialization, Setup/Verify (including legacy runtime
Verify), reset, explicit IO and release. First prediction belongs to prefill; decode steps
are `consumed - prompt_length`. Existing `metrics` reports host generation wall time.
Neither is pure NPU arithmetic time. Failed checks have no accepted performance metrics.

## Native generation contract

Both binaries take only three paths:

```text
build/bin/specferry_opt_generate DEPLOYMENT REQUEST OUTPUT
build/bin/specferry_opt_graph_generate DEPLOYMENT REQUEST OUTPUT
```

The Python launcher prepares REQUEST and `run-options.txt`; the graph CPU token oracle is
fixed at `REQUEST/expected.txt`. Options use unique `key value` lines; unknown/duplicate
keys and invalid combinations fail before SDK access. The first row below is common to both
binaries; the remaining rows are graph-only. Component also accepts `seed` for sampling;
omitting it selects greedy. Omitted options use these defaults:

| Keys | Default / choices |
|---|---|
| `max_new_tokens`, `warmups`, `repeats` | `8`, `0`, `1` |
| `block` | `1`; graph also supports `4` under the restrictions above |
| `variant` | `merged-kv`; candidates `merged-kv-local-const`, `merged-kv-local-mutable` |
| `decoder_group_size`, `prefill_head` | `1`, `every-token`; block1 candidates `2`, `last-token` |
| `readback` | `none`; `logits`, or `kv` with explicit `cache_first_layer` / `cache_last_layer` |
| `benchmark`, `lifecycle` | `0`; graph accepts `1`, mutually exclusive |

Example `run-options.txt` for a benchmark:

```text
block 4
max_new_tokens 8
benchmark 1
warmups 1
repeats 3
```

Graph layout is fixed to partitioned. Public step/close/capacity guards and payload limits
remain active; LastToken skips intermediate prompt heads while public step and decode still
produce a complete prediction. Coarse diagnostics retain numerical assertions and explicit
SDK ownership/release, rather than inferring correctness from normal exit.

## Model and lifecycle diagnostics

```bash
python scripts/check_np101_model.py prefix-split --layer-count 2 --steps 2 --capacity 16 \
  --prepare-only --output .cache/runs/prefix-prepared
python scripts/check_np101_model.py prefix-split --layer-count 2 --readback outputs \
  --fixture .cache/runs/prefix-prepared/fixture --output .cache/runs/prefix
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --request-tokens-file /absolute/path/requests.json --max-new-tokens 2 \
  --lifecycle-check --cache-readback logits --output .cache/runs/rebuild
```

`prefix-split` defaults to complete logits/selection for two teacher steps. Explicit
`outputs-kv` adds current KV, bit-exact history and zero suffix; `boundaries` with
`--boundary-layers 8 16 24` adds selected decoder outputs. A boundary list may cover all
layers when fault investigation requires it; full-intermediate
readback modes were retired. `prefix-split-repeat` rebuilds Context/model twice. Teacher
checks do not replace generation reset/isolation, capacity acceptance or hardware evidence.
K/group2 and WK/WcK are explicit candidates, not automatically accepted configurations.
Native prefix execution takes `np101_graph_pipeline_check DEPLOYMENT FIXTURE OUTPUT`;
`FIXTURE/run-options.txt` carries `layers`, `readback` (`outputs`, `outputs-kv`, `boundaries`),
comma-separated `boundary_layers`, `variant`, `decoder_group_size`, `repeat` and `check_fixture`.
The wrapper writes these options during preparation; use its `--prepare-only` path for offline
fixture validation instead of launching a native board command.

Lifecycle now checks two model lifetimes, each with a fresh Context, complete final logits,
close rejection and zero live storage. Same-Context rebuild and the two-layer state-only
entry have been retired after diagnosis; historical failures remain in raw evidence.
Lifecycle automatically enables the pinned compilation guard, rejecting missing/zero
operation entries before RunGraph. Ordinary requests keep one resident Context/model.

`--trace-verify` enables the explicit `np101_verify_trace` diagnostic (build its CMake
target first). It records Verify stages, nodes/operation entries and execution replay /
submit / completion observations. It gates fixed v105 OpenVX/GAL hashes, rejects inherited
preloads and does not modify SDK objects. These traces are neither throughput baselines
nor sufficient numerical/hardware proof by themselves. Only targeted fault investigation
should use them; no automatic detail tracing on failure.

Retained Qwen3.5 regression uses the cached checkpoint:

```bash
python scripts/export_np101_dlm.py --verify-only
python scripts/reference_dlm.py --output .cache/runs/qwen-cpu
python scripts/check_np101_model.py qwen --trace .cache/runs/qwen-cpu/layer-0-3-sequential.npz \
  --layers 0 1 2 3 --steps 2 --prepare-only --output .cache/runs/qwen-prepared
python scripts/check_np101_model.py qwen --fixture .cache/runs/qwen-prepared/fixture \
  --output .cache/runs/qwen-slice
```

## Operator, cache and memory diagnostics

Use one selected case per process; [P1–P10](../docs/np101-sdk-driver-validation-report.md)
contains precise expected/actual results, crash/hang restrictions and reproduction arguments.

| Launcher | Retained cases / responsibility |
|---|---|
| `check_np101.py` | lookup, projection, sampling, fp16-preservation, conv: atomic synthetic checks |
| `check_np101.py` | kv, cache-append, cache-packed, cache-packed64, cache-packed-block4: fixed writes, independent readers and causal attention |
| `check_np101_memory.py` | capacity (allocation only), integrity (all retained bytes), accounting (SDK counters), growth (fixed/rebound RSS), weights (exported pack readback) |
| `check_np101_model.py` | OPT/Qwen decoder regression, teacher component reference, selection/io composition and prefix integration |

Memory payload remains separately capped at 1 GiB const/nonconst. Only explicitly authorized
v106 synthetic P5 capacity/integrity probes use `--probe-limit` up to 4096 MiB; after refusal
scan all successfully written blocks, including those after mismatches. Full retained-data
coverage is not physical board coverage. RSS/counters cannot prove leakage or transfer cost.
Packed block4 uses a calibrated fixed writable `vxReshapeTensor` alias because ordinary
RESHAPE does not guarantee writeback; see the current status API exception and evidence.

## Offline checks and outputs

```bash
python -m unittest discover -s tests -t .
ctest --test-dir build --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

Compute reports use `result.json` and `device/`; memory reports use the output directory.
`execution-evidence.json` records command, boot/binary/SDK identity, return state and resource
usage; `sdk.log` retains SDK messages. Optional calls TSV and driver ioctl tracing are
fault diagnostics. Raw historical files keep their original names/IDs; current wrappers
must not be used to claim reproduction of retired graph structures.
