# Model-independent memory diagnostics

Build the current repository and activate the `SpecFerry` Conda environment:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4
conda activate SpecFerry
```

These three experiments need no model download. Run them serially, with read/write
access to `/dev/galcore`, using a fresh output directory for each command. Python
only launches the native SDK program and records logs, binary/source/library hashes,
process lifetime and the recovery guard. A signal, timeout, API/readback failure or
incomplete release is a reason to stop and inspect the evidence before another run.
A completed byte-mismatch scan is separately identified.

## Host allocation growth

```bash
python scripts/check_np101_memory.py --output .cache/runs/memory-fixed growth --mode fixed
python scripts/check_np101_memory.py --output .cache/runs/memory-same growth --mode same
python scripts/check_np101_memory.py --output .cache/runs/memory-advance growth --mode advance
```

`tests/native/np101/memory_growth_check.cpp` creates one copy node, a 2 KiB source,
and a 16 KiB parent with eight preallocated views. `fixed` never changes the binding;
`same` binds the same destination again; `advance` cycles through the eight views.
It verifies the graph whenever the SDK invalidates it, then executes the copy.
There is no model, per-iteration tensor/view creation or growing application container.
The final parent is compared byte for byte, including untouched slots.

`--iterations` defaults to 128 (range 1..4096). The log prints iteration, phase,
current host RSS in bytes and phase seconds, followed by observations after graph,
view and context release. `summary.json` also retains the RSS growth and phase totals.
Compare the slope after the first execution. RSS can
include allocator retention; it does not by itself establish a leak or board usage.
The direct view/copy calls reproduce the existing KV append mechanism, whose purpose
is to avoid copying the entire cache through the documented whole-tensor update path.

## SDK device-memory accounting

```bash
python scripts/check_np101_memory.py --output .cache/runs/accounting-1m accounting --mib 1
python scripts/check_np101_memory.py --output .cache/runs/accounting-8m accounting --mib 8
python scripts/check_np101_memory.py --output .cache/runs/accounting-32m accounting --mib 32 --storage mutable
```

`tests/native/np101/memory_accounting_check.cpp` creates **one** FP16 or FP32 tensor, checks
all bytes and releases it. It prints payload bytes, SDK `gpu_memory.currentSize`
before/during/after allocation, and the delta/payload ratio. Use several sizes to
separate fixed overhead from proportional overhead. `--mib` accepts 1..64 and
`--storage` accepts `constant` or `mutable`; mutable backing is forced by upload.
Use `--dtype F32` to check whether the ratio also occurs without FP16 storage.
No arithmetic graph or model workspace is involved.

Only this diagnostic process enables `VIV_MEMORY_PROFILE=1`. It uses the declared
HAL counter API because the operator guide has no accounting interface. A sentinel
rejects SDK success without populated output. This is an **SDK counter observation**,
not physical board occupancy or proof of a duplicated tensor. Counter semantics
still require explanation. Inference performance measurements keep profiling off.

## Current segmented payload policy (2026-09-24)

The common native allocation helpers enforce **1 GiB per `is_const` category**
across all graphs in a context. Retained references and storage reshapes share one
budget reservation; their temporary SDK wrapper backing is charged until replaced.
The charge is returned after the last participating graph releases it. Capacity
and full-weight checks also reject oversized requests before opening the device.
An application-budget rejection is not reported as an SDK capacity measurement.

`memory-budget.json` records live/peak application payload per category. These are
conservative explicit-allocation counts, not physical memory measurements: SDK
internal tensors, virtual intermediates, workspace, padding and hidden copies are
unknown. No driver, allocation address or SDK `is_const` flag is changed. Shared
WeightBank weights remain non-const, so they share that limit with KV and ordinary
intermediates. OPT-350M's weight pack is 631.707 MiB; graph overhead still matters.

The former corruption offset is **below** 1 GiB of logical payload. This policy
cannot blacklist it or establish data integrity. Pipeline diagnostics now read
every shared-weight byte before SetupGraph, after SetupGraph, after VerifyGraph,
and after execution. See [pipeline checks](np101-graph-pipeline.md).

After the user-confirmed cold boot, the first synthetic lookup reached three exact
weight-readback phases with const/non-const peaks of 76/1020 bytes, then failed to
complete. Kernel logs report two NP101 hangs, including one after process exit.
The segmented policy therefore has no complete hardware acceptance yet. A new
recovery marker blocks further tests; see the [cold-boot record](np101-graph-pipeline.md).

## Large retained allocation integrity

Reuse `tests/native/np101/capacity_check.cpp`; do not introduce a second scanner:

```bash
python scripts/check_np101_capacity.py --target-mib 64 --readback all \
  --output .cache/runs/integrity-small
python scripts/check_np101_capacity.py --target-mib 1024 --readback all --timeout 600 \
  --output .cache/runs/integrity-large
# Optional independent configurations: --storage mutable and/or --dtype F32.
```

All 8 MiB blocks remain alive until the full readback finishes. Deterministic finite
FP16/FP32 patterns distinguish different blocks and element positions. The scanner
checks every byte and continues after mismatches; an SDK readback failure interrupts
it. `readback-blocks.jsonl` records exact affected ranges and logical pages;
`capacity.json` and `summary.json` distinguish allocation, integrity and release.
The console summarizes damaged blocks and their ranges. Offsets are zero-based,
end-exclusive **payload offsets, not physical addresses**.

To measure allocation capacity without integrity, use the same tool with
`--target-mib 1024 --readback none`. This separate mode cannot establish correct data
storage. Larger targets are currently disabled; see [the capacity record](np101-capacity.md)
for the historical 2840 MiB observation and its matching corruption map.

## Direct native invocation

For a C++-only run after setting the SDK library path:

```bash
export LD_LIBRARY_PATH=/usr/lib/ljmicro
VIV_MEMORY_PROFILE=0 build/tests/np101_memory_growth_check advance 128
VIV_MEMORY_PROFILE=1 build/tests/np101_memory_accounting_check 8 constant
mkdir -p .cache/runs/integrity-native
VIV_MEMORY_PROFILE=0 build/tests/np101_capacity_check \
  .cache/runs/integrity-native/capacity.json constant F16 1024 all
```

Direct invocation bypasses the Python device lock and timeout/recovery guard: keep
one process on the device and inspect its exit before proceeding. Memory-growth and
accounting programs return 0 for a completed measurement with correct readback and
normal release, and 2 for an incomplete experiment. Accounting returns 1 if the
completed readback differs. A completed experiment does not
mean its growth/ratio is acceptable. Capacity returns nonzero for integrity failure;
its reports distinguish a complete mismatch scan from abnormal execution.

## Observations on 2026-09-23 (SDK/driver package 1.0.6)

The model-independent experiments reproduce all three observations:

- With one copy node and 128 iterations, `fixed` has no RSS growth after its first
  execution. `same` and `advance` grow by about 16 MiB over the following 127 copies.
  The observed increases occur across the `vxVerifyGraph` phase, not binding or
  processing. The application has no growing allocation container. Readback and
  normal release pass. This narrows the reproducing call sequence; it does not yet
  identify an allocator bug, ownership error or driver root cause. The roughly
  128 KiB per verification is consistent in scale with 24 KV graphs producing
  about 3 MiB per full-model consumed token; this is not an allocation-stack proof.
- For 1, 8 and 32 MiB FP16 tensors, both constant and mutable storage produce
  `gpu_memory` deltas of `2 * payload + 4,672` bytes in this run. For 8 MiB that is
  16,781,888 counted bytes. Readback passes, the post-context counter returns to
  its pre-allocation baseline, and the SDK exit dump reports zero outstanding
  GPU-accounted bytes. Physical occupancy and the extra accounting remain unexplained.
- Retaining 355 eight-MiB tensors reproduces 24 mismatched bytes in block 109,
  for FP16/FP32 and constant/mutable storage. The three local byte ranges remain
  `[1274680,1274688)`, `[1274744,1274752)`, `[1274808,1274816)`. Corresponding payload
  starts are 915,632,952; 915,633,016; 915,633,080. All other bytes match and release
  completes. Four 64 MiB controls pass. This agrees with the existing capacity scan.

Do not combine these into a proven common cause: host growth, SDK GPU accounting
and data corruption have different observable behaviors. The programs make each
one reproducible without a model and keep their conclusions separate.

The related [startup-wait diagnostic](np101-sdk-timing.md) follows the same run-directory,
SDK identity and recovery conventions. Use it separately if long waits recur after
cold start; the memory observations do not establish the cause of those waits.

Local raw evidence (ignored by Git): `.cache/runs/acceptance-20260923-memory/`
contains the initial minimal experiment snapshots, including archived draft source;
`.cache/runs/memory-integrated-{fixed,same,advance,accounting}/` verifies the integrated
entry points and shared resource helpers. The repository capacity scanner retains
its earlier complete matrix under `.cache/runs/capacity-map-20260923/`.

The final build and host regressions pass. Additional FP32 accounting controls
and a repeat of the three small growth controls were deferred when the user
cancelled a separate generation test and activated the device recovery guard.
The subsequent cold-start O0–O2 work resumed generation/latency checks; it did not
repeat this independent memory matrix.
The accounting ratio above is measured for FP16 only; FP32 support in the new
accounting command has compiled but has not yet been exercised on the device.
The FP16/FP32 retained-byte corruption matrix is already complete.
