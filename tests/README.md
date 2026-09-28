# Validation commands

Run from the repository root after [environment setup](../README.md):

```bash
conda activate SpecFerry
cmake -S . -B build
cmake --build build -j 4
```

Every board command runs **one case in one child process**, with its own inputs and
new output directory. There is no automatic sweep or prerequisite board test.
Timeouts, signals and lingering children leave a recovery marker; resolve the device
fault before another board command. Do not delete the marker to force a retry.
Before intentionally reproducing the selection hang, cold-restart both host and NP101.

## Test boundaries

| Entry / case | Responsibility | Reference / dependencies |
|---|---|---|
| `check_np101.py lookup` | One indexed row lookup; exact FP16 bytes, unchanged table | Synthetic table; GATHER or EMBEDDING_LOOKUP, one per process |
| `check_np101.py projection` | One biased linear projection | Identity matrix; FCL or MatMul+Add, one per process |
| `check_np101.py sampling` | Categorical sampling range, distribution, seed/counter repeatability | Synthetic logits; one precision per process; NumPy evaluator |
| `check_np101.py fp16-preservation` | Preserve FP16 bits through addition or selection | One ADD or SELECT node; exact synthetic bits, no model |
| `check_np101.py conv` | CONV2D→ReLU→MaxPool reference chain | Deterministic CPU golden, derived from the vendor demo |
| `check_np101.py kv` | View-based cache writes visible to a fixed reader | Synthetic values; no attention |
| `check_np101.py cache-attention` | Indexed append consumed by attention, causal mask and reset | Integration check; block 1 or 4 |
| `check_np101.py cache-append` | One indexed write into a preallocated cache | One TENSORSTACKCONCAT; explicit 2-D or 3-D configuration; no reshape/attention |
| `check_np101_memory.py capacity` | Allocation and release; no readback claim | Synthetic 8 MiB blocks |
| `check_np101_memory.py integrity` | Retain all blocks, then scan written bytes | Same allocator, independent full-readback experiment |
| `check_np101_memory.py accounting` | SDK counter delta and release | One tensor; readback is a validity check, not a capacity scan |
| `check_np101_memory.py growth` | Host RSS across fixed/rebound copy graphs | Preallocated views; fixed/same/advance, one per process |
| `check_np101_memory.py weights` | Deployment pack loading and readback | Explicit exported pack, optional state shapes; no model execution |
| `check_np101_model.py` | Model composition, cached CPU references and layer boundaries | OPT / Qwen adapters; integration tests |

Atomic operator checks do not load a model or run other tests. Composed graph/model
checks intentionally retain their end-to-end assertions: they test connections that
single-operator checks cannot establish. No separate test is named after an issue ID.

## Operators and cache

Choose and run one command; the following is a menu, not an unattended batch:

```bash
python scripts/check_np101.py lookup --operator embedding --sdk-timing --timeout 60 \
  --output .cache/runs/lookup-embedding
python scripts/check_np101.py projection --operator matmul-add --sdk-timing --timeout 60 \
  --output .cache/runs/projection-control
python scripts/check_np101.py sampling --dtype F16_TO_F32 --classes 8 --samples 4096 \
  --sdk-timing --output .cache/runs/sampling
python scripts/check_np101.py fp16-preservation --operator select --sdk-timing --timeout 30 \
  --output .cache/runs/fp16-select
python scripts/check_np101.py conv --sdk-timing --output .cache/runs/conv
python scripts/check_np101.py kv --sdk-timing --output .cache/runs/kv
python scripts/check_np101.py cache-append --rank 2 --sdk-timing --timeout 30 \
  --output .cache/runs/cache-append-2d
python scripts/check_np101.py cache-attention --block 1 --sdk-timing --output .cache/runs/cache
```

- Lookup: `--operator gather` changes only the lookup operator/argument order. The
  FP16 `[8,8]` table and INT32 index are initialized before Verify; ten row checks cover
  all indices, repeats, signed zero and subnormal bits. This is a new controlled case,
  not identical to the historical model selection test. The no-GATHER alternative
  has **not** been accepted on a recovered board.
- Projection: `--operator fcl` tests the known Verify crash separately. Both modes use
  FP16 `[8,1]`, identity weights `[8,8]`, bias `[8]=0.25`, and expect 1.25 then -0.25.
  The program validates its actual node list and tensor references before Verify;
  unknown native modes fail before device access. Run the control first and FCL last.
- Sampling: `--dtype F32` and `--dtype F16` are explicit comparison variants. Direct
  FP16 currently returns invalid indices; a normal native exit is insufficient.
  Python checks all seven sample files and returns nonzero on numerical failure.
- Cache/attention: `--block 4` checks grouped prefill and one-token tails. This uses
  2D cache updates; it does not reproduce the historical rank-three Setup crash.
- Cache append: `--rank 2` writes `[8,1]` into a preallocated `[8,8]` cache;
  `--rank 3` writes `[8,1,2]` into `[8,8,2]`. Each graph contains just one
  TENSORSTACKCONCAT(axis=1), with no application RESHAPE node. Two indexed writes
  check the new slot, previous values and untouched suffix; capacity never grows.
  The 3-D configuration is a diagnostic whose SDK support is unconfirmed. Run it
  last, separately from the 2-D control; it may crash during SetupGraph.
- FP16 preservation: compare `--operator select` with `--operator add` in separate
  output directories. Each tests 16 exact FP16 values, including 10 subnormals of
  both signs. One input contributes the value and the other zero at each element.
  The v105 ADD run flushed all 10 subnormals; SELECT preserved all 16 values.
  The same SELECT executable on v106 after a cold restart also preserved the values,
  but Verify took 28.982 s and kernel logs reported NP hang/recovery. Check kernel
  health as well as numerical results. The same ADD executable on v106 also flushed
  all 10 subnormals; its input and output bytes match v105, with no new kernel hang.
  Run ADD separately before the known SELECT hang reproduction.
  `device/sdk.log` prints expected/actual bits; `expected.bin`, `actual.bin`,
  `left.bin` and `right.bin` retain the bytes. Numerical failure returns nonzero
  after releasing SDK resources; it is distinct from a signal or timeout.

The operator wrapper uses only the Python standard library except sampling's NumPy
validation. Native executables are in `build/tests` and print usage on invalid arguments.
A wrapper is shared across cases because it adds process isolation, deadlines, recovery
protection and reports; there are no redundant per-operator Python launcher files.

## Memory

```bash
python scripts/check_np101_memory.py capacity --mib 64 --storage constant --dtype F16 \
  --output .cache/runs/capacity
python scripts/check_np101_memory.py integrity --mib 1024 --storage constant --dtype F16 \
  --output .cache/runs/integrity
python scripts/check_np101_memory.py accounting --mib 8 --storage mutable --dtype F16 \
  --output .cache/runs/accounting
python scripts/check_np101_memory.py growth --mode fixed --iterations 128 \
  --output .cache/runs/host-growth
python scripts/check_np101_memory.py weights --deployment .cache/np101/opt-350m \
  --storage constant --output .cache/runs/weights
```

Application const/nonconst payloads are independently capped at 1 GiB. The synthetic
`capacity` and `integrity` diagnostics accept explicit `--probe-limit` to raise their
requested ceiling to at most 4096 MiB; this does not change model memory budgets.
For a v106 near-limit scan, after a passing 64 MiB control, run:

```bash
python scripts/check_np101_memory.py integrity --mib 4096 --probe-limit --storage constant --dtype F16 \
  --timeout 180 --output .cache/runs/v106-integrity-limit-f16-constant
```

Allocation stops at the requested ceiling or the first SDK allocation/upload refusal.
All successfully written blocks remain allocated during readback. The scan checks
every byte of every retained block, including blocks after a mismatch; release follows
the entire scan. This is coverage of retained tensors, not all physical board memory.
The terminal summary reports retained/requested bytes, scan coverage, mismatch counts
and whether the next block was refused. `full_scan_completed` is separate from numerical
success; an incomplete scan cannot establish that only the reported locations are bad.
The v105 comparison stays within 1 GiB: use identical 1024 MiB targets, storage and
precision on both versions. Failure to allocate is recorded separately from byte errors.

`capacity` performs the same initial uploads as `integrity` but does not read them back; it does
not prove usable memory or an exact physical limit. `integrity` scans every retained
block, including blocks after corruption. `readback-blocks.jsonl` contains logical
byte/page ranges, not physical addresses. Allocation refusal and corruption have
separate result fields. Storage and precision are never changed as fallback policies.

Compare `growth --mode fixed`, `same`, and `advance` separately. SDK counters do not
prove physical allocation; RSS does not prove a leak or host/device transfer cost.
The growth log ends with a MiB delta between the first and last completed copy;
use that same measurement interval for both driver versions.
For extra state allocation with weights, add
`--state-spec tests/fixtures/qwen3_5_allocation_states.txt` (shapes only).

## Model integration

```bash
python scripts/check_np101_model.py selection --prepare-only --output .cache/runs/selection-inputs
python scripts/check_np101_model.py opt --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 --steps 2 --capacity 16 --prepare-only --output .cache/runs/opt-prepared
python scripts/check_np101_model.py opt --fixture .cache/runs/opt-prepared/fixture \
  --output .cache/runs/opt-slice
python scripts/check_np101_model.py prefix --layer-count 1 --steps 2 --capacity 16 \
  --prepare-only --output .cache/runs/prefix-prepared
python scripts/check_np101_model.py prefix --fixture .cache/runs/prefix-prepared/fixture \
  --layer-count 1 --sdk-timing --output .cache/runs/prefix
```

`selection` tests synthetic OPT input/output composition, boundary IDs and ties;
`io` uses actual weights. `lookup` tests packed lookup/input projection/position
embedding; `layer` checks layer 0 with at least two reference steps. For real-weight
lookup, prepare `io --prepare-only`, then pass that fixture to `lookup --fixture`.
`teacher --layer-count N` compares embedding, N decoder layers, logits, valid KV
prefixes and reset/fresh trajectories. These overlap operators by composition, not
by duplicating the isolated operator implementations.

`prefix` needs its own one-layer/two-token truncated-model reference. Full-model
teacher fixtures are rejected. It reads existing input and KV tensors without
adding SDK nodes or graph outputs; passing it does not accept decoder outputs or
LM head. Default deadline: prefix 360 s (historical Verify about 158 s), others 120 s.
The native preflight also avoids device access:

```bash
build/tests/np101_graph_pipeline_check .cache/np101/opt-350m \
  .cache/runs/prefix-prepared/fixture .cache/runs/prefix-preflight prefix 1 --check-fixture
```

Retained Qwen regression:

```bash
python scripts/export_np101_dlm.py --verify-only
python scripts/reference_dlm.py --output .cache/runs/qwen-cpu
python scripts/check_np101_model.py qwen --trace .cache/runs/qwen-cpu/layer-0-3-sequential.npz \
  --layers 0 1 2 3 --steps 2 --prepare-only --output .cache/runs/qwen-prepared
python scripts/check_np101_model.py qwen --fixture .cache/runs/qwen-prepared/fixture \
  --output .cache/runs/qwen-slice
```

## Results and offline checks

Compute reports: `result.json`, `device/sdk.log`, `device/execution-evidence.json`.
Memory reports use the output directory directly. Evidence records the command,
boot ID, executable and SDK-file SHA256, elapsed process time, return code and recovery
state. `--sdk-timing` adds public API begin/end records in `device/sdk-calls.tsv`;
`--trace-driver` adds ioctl traces when strace is installed. Sampling results include
out-of-range counts and observed/expected frequencies. These are diagnostics, not
throughput measurements or proof of exclusive NPU execution.

```bash
python -m unittest discover -s tests -t .
ctest --test-dir build --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

CPU checks and `--prepare-only` do not open NP101. For operator cases,
`--prepare-only` prints the native command; model cases generate/validate fixtures.
Board diagnostics are deliberately not registered with ordinary CTest.

Shared C++ byte IO is in `native/np101/binary_io.hpp` relative to this directory;
float encoding, output setup and error handling are in `native/np101/test_support.*`.
Tests retain graph construction, assertions and explicit SDK ownership/release.
No automatic run-all command is provided for known crash/hang cases.

Migration: model cases moved from `check_np101.py` to `check_np101_model.py`;
`lookup-embedding/gather` became `lookup --operator ...`, projection variants became
`projection --operator ...`, and `cache/cache-block` became `cache-attention --block 1/4`.
Memory `capacity --readback none/all` became `capacity/integrity` respectively.
Historical reports keep their original commands and are not rewritten.
