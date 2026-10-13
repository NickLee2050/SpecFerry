# Validation

Run from the repository root in Conda `SpecFerry`, after [environment setup](../README.md).
[Status](../docs/np101-status.md) owns accepted scope/precision/performance,
[TODO](../TODO.md) unfinished work, and [incidents](../docs/np101-recovery-incidents.md) device gates.
Historical evidence and retired programs are archived; they are not current test commands.

## Test layers

| Layer | Entry | Purpose |
|---|---|---|
| Offline Python | `python -m unittest discover -s tests -t .` | Checkpoint/export, generation policy, arrays/budgets, benchmark counts, KV/lifecycle and recovery guards; device calls are mocked |
| Offline native | `ctest --test-dir build --output-on-failure` | Generation, timing, segmented memory, allocation readback and data contracts; never opens NP101 |
| Model acceptance | `scripts/generate_opt.py` | Complete token/logit/KV correctness, lifecycle and resident performance; explicit CPU oracle |
| Compiled acceptance | `scripts/check_np101_compiled.py` | Pinned NBG/reference plan, coarse output checks, release/budget and delayed health gate |
| Targeted diagnostics | `check_np101.py`, `check_np101_memory.py`, `check_np101_model.py` | One selected operator/cache/memory/model question; no automatic sweep |

Python tests are grouped by contract: `test_recovery`/`test_device_runner`, `test_graph_options`,
`test_graph_benchmark`, `test_graph_cache`, `test_generation_cli`, `test_compiled_validation`,
checkpoint/export/policy tests and common array/allocation checks. Fixtures are shared without
sharing expected values with the code being checked. All 5 native CTest cases carry `unit;cpu` labels.

```bash
python -m unittest discover -s tests -t .
ctest --test-dir build -L cpu --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

## Device discipline

Read the ledger before every run; check boot, recovery marker, full-boot kernel and device owner.
Serialize bounded processes, monitor faults and inspect at least 45 seconds after release.
Correct outputs, empty ownership or `recovery done` do not override faults. Preserve markers.
`--prepare-only` validates files/options without kernel or device access.

The deleted repository convolution and its `conv`/`recovery` entries must never be restored.
`vendor-conv` is an explicit diagnostic, cannot clear the marker, and requires agreement.
Future fault recovery requires confirmed host/NP101 cold restart plus agreed controls and numerical,
execution, budget, release and delayed-kernel checks. A passing workload proves only its scope.
`--sudo-read-only` means authorized journalctl/fuser reads; it does not grant permission itself.

## Complete generation and benchmarking

```bash
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --compare-cpu --output .cache/runs/opt-tokens
python scripts/generate_opt.py --backend graph --block 16 --capacity 64 \
  --variant merged-kv-fc1-split2 --prefill-head last-token \
  --max-new-tokens 16 --benchmark --warmups 1 --repeats 3 \
  --output .cache/runs/opt-benchmark
```

Graph defaults: merged-kv, block1, every-token head, greedy, token-only readback. Blocks4/8/16
consume known consecutive input from one request; tails/decode use block1, capacity must divide
by block. Block8 model execution remains unvalidated. `last-token` skips unused prompt heads.
Explicit component sampling uses `--sample --seed`; unsupported checkpoint settings fail.
`--request-tokens-file` contains a JSON list of requests sharing one resident model.

Benchmark validates warmups and excludes them; `benchmark.json` maps every measured file to its
input/repeat, preserving actual EOS/capacity stops. Report actual output count/generation wall and
post-first-token decode separately. Preparation/Setup/Verify/reset/extra diagnostics are excluded;
required controls/token reads are included. CPU user/system time overlaps wall and cannot be added.

For diagnostics, `--cache-readback logits` reads final logits; `kv` requires one-based inclusive
`--cache-layer-range FIRST LAST`. All layers still execute. `--readback-transport mapped` is an
explicit ordinary nonquantized rank1/2/3 Map/Unmap path with stride-aware copies and no SDK fallback;
use only calibrated scopes. `sdk` remains default; selected-token IO is unchanged.
`--lifecycle-check` releases the old model and final Context before rebuilding, compares logits,
close rejection and live memory; benchmark/lifecycle modes are separate acceptance runs.

Native generation keeps three paths: `DEPLOYMENT REQUEST OUTPUT`. REQUEST contains unique
`key value` settings in `run-options.txt`; unknown/duplicate/incompatible keys fail before SDK access.
Common keys: max_new_tokens/warmups/repeats; graph also block/variant/prefill_head/readback/
readback_transport/cache_first_layer/cache_last_layer/benchmark/lifecycle; component alone seed.
Use the Python launcher to prepare these files instead of reconstructing options by hand.

## Compiled model acceptance

```bash
python scripts/check_np101_compiled.py --plan /absolute/path/plan.json \
  --case fullgraph --prepare-only --output .cache/runs/compiled-check
python scripts/check_np101_compiled.py --plan /absolute/path/plan.json \
  --case fullgraph --output .cache/runs/compiled-check
```

One selected case per process; this replaces date-specific run/summarize scripts. The plan has
schema_version=1, pinned SDK directory/module srcversion/library hashes, named file_sets and cases.
Each case states binary/arguments/request, file_sets, kind (fullgraph/partitioned), measured repeats,
expected stop reason and separate const/nonconst payload bounds. Reference JSON/logits and optional
KV files must be hashed; fixed historical binaries also have binary_sha256. Paths can be absolute
or repository-relative, and `{output}` denotes the child output directory. No data path is defaulted.

Fullgraph uses the maintained `np101_compiled_model_check CONTRACT REQUEST OUTPUT` CMake target:
accepted M1/C64, prompt17/output16, one warmup plus one measured request, final logits/last-layer
FP32 KV and release only. CONTRACT is the existing SDK-order 52-input/50-output text contract.
Its NBG and contract hashes belong in the plan. Ordinary and FFN-TTGT artifacts use the same target.
Layered compiled regressions retain their previously validated frozen binary identities until the
formal exporter/loader is completed; their source evidence is in the archive, not duplicated programs.
The consolidated local plan is `.cache/runs/commit-cleanup-20261013/compiled-opt-plan.json`.

The runner checks identities before device access, uses one lock through delayed checks, validates
warmup and every measured output, and emits result.json with independent numerical checks and
performance only on acceptance. Optional cache contract states dtype/shape/reference paths;
`device_reset` checks the stored coarse reset result and zero reset uploads.
`--audit-verify` uses only pinned v105 Verify snapshots with execution tracing off; it refuses unknown
SDK identities and checks complete graph/operation scans. NBG internal operators are not enumerated.

## Diagnostic selection

| Entry | Retained cases |
|---|---|
| `check_np101.py` | lookup, MatMul+Add/mapped-readback projection, sampling, FP16 preservation; explicit vendor diagnostic |
| `check_np101.py` | kv, cache-append rank2/3, cache-packed/64/block4: writes, sharing and causal dependencies |
| `check_np101_memory.py` | capacity, integrity, accounting, growth, weights; capacity alone is not integrity |
| `check_np101_model.py` | OPT/Qwen regression, teacher reference and selection/IO isolation |

Raw FCL/forced-const/VX projection experiments are retired. Keep failures in the SDK report and
ledger; successful NBG import does not validate them. Quantization research is closed.
Const/nonconst each remain ≤1GiB. Only explicitly authorized v106 synthetic capacity/integrity probes
may use --probe-limit up to4096MiB; after refusal scan all successfully written blocks. Logical bytes,
RSS and SDK counters do not prove physical allocation, leakage or bandwidth.

FFN collections retain `np101_projection_check OUTPUT MODE DEPLOYMENT FIXTURE`: ffn-full,
ffn-linear, ffn-linear-fc1-split2, or ffn-linear-batched with 4/8/16 after FIXTURE. Reuse teacher
attention-norm/MLP fixtures; 24 different weights, four warmup sweeps and 3×8 measured sweeps,
final-layer coarse mapped output. Preparation/IO/release excluded; not model output throughput.

Qwen3.5 keeps its cached checkpoint and existing `export_np101_dlm.py --verify-only`, reference_dlm
and check_np101_model qwen paths. No new downloads or unsupported configurations are enabled.

## Tracing and records

`--sdk-summary`/`--sdk-timing` are mutually exclusive. Summary aggregates request/phase/role/API
in memory and writes after release; nested counts cannot be added. `--trace-verify` is explicit pinned
v105 diagnosis; SPECFERRY_VERIFY_EXECUTION_TRACE=0 keeps only Verify, summary enables fixed
in-memory Process/Submit/Replay/Wait/HAL data (4096 graph calls). Interrupted collections are discarded.
Profiler is explicit and separate; follow the off/on/off method and known perturbation in the status page.

Standard output: result.json, device/execution-evidence.json (command/boot/binary/SDK/exit/resources),
sdk.log, measured generation files, budgets/release and health snapshots. Raw original reports,
NBG/ONNX/metadata, checkpoints and cold-restart evidence stay under ignored cache paths. Temporary
sources/scripts are archived by hash before deletion; do not run historical commands with current binaries.
