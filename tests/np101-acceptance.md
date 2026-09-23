# Complete OPT acceptance and timing observations

The acceptance runner checks the complete OPT-350M SDK path, repeatability across
resident requests and fresh processes, and application transfer contracts. It
reports host-observed timings separately from formal NPU performance acceptance.

## Run

Use the `SpecFerry` Conda environment, a current build and a verified OPT export.
Use a new output directory each time:

```bash
cmake --build build -j 4
python scripts/check_np101_acceptance.py --prepare-only \
  --output .cache/runs/opt-acceptance-prepared
python scripts/check_np101_acceptance.py --output .cache/runs/opt-acceptance
```

`--prepare-only` writes checkpoint-default policies, tokenized requests, independent
CPU generation results, exact selection fixtures and all-layer teacher-forced
expectations. It does not open the board. The execution command prepares its own
fixtures in a fresh directory to keep one complete evidence set.

The default suite uses capacity 512, eight teacher-forced steps, two fixed prompts
and eight new tokens. For each prompt it creates two fresh native processes; each
loads one complete model, runs one warmup, then three measured requests on that
same allocation. Each request resets the valid KV prefix without clearing/copying
its storage. Every warmup and measured token sequence must match the official CPU
model exactly. All processes must release and exit normally.

To run a shorter bounded comparison:

```bash
python scripts/check_np101_acceptance.py --max-new-tokens 4 \
  --warmups 1 --repeats 2 --cycles 2 --output .cache/runs/opt-acceptance-short
```

Use repeatable `--prompt TEXT` to choose prompts. `--capacity` and `--steps` accept
1–512; teacher steps are capped at the selected capacity. `--max-new-tokens` accepts
1–512, `--warmups` 0–5, `--repeats` 2–20, and `--cycles` 2–5. `--timeout` bounds each
native process. Longer contexts and trajectories should be separate named runs;
a short suite does not establish long-context quality or capacity-bound behavior.
A one-token result has no decode interval and reports decode throughput as null.

The suite uses the checkpoint's default greedy policy. It rejects a sampling
checkpoint instead of silently changing its strategy to make exact repeat checks
pass. Optional sampling remains available through `generate_opt.py --sample` and
its separate [sampling diagnostic](np101-sampling.md).

`--model`, `--checkpoint`, `--binary`, `--io-binary`, `--sdk-lib` and `--shader-header`
allow explicit local paths. The runner snapshots both native executables, records
source/library hashes, and rejects boot or SDK-library changes between stages.
It uses the existing device lock and recovery guard. Any failed correctness gate,
readback/transfer contract, signal, timeout or incomplete release stops the suite;
it never retries a failed native process automatically.

Exit codes:

- `0`: preparation succeeded, or numerical checks passed with explicit `--diagnostic`.
- `1`: preparation, execution or numerical/lifecycle checks failed.
- `2`: numerical checks passed, but hardware acceptance remains open.

## Measurement boundaries

The native `benchmark` mode reuses production `Model::generate`; it contains no
alternative model arithmetic. It disables the streaming callback and writes
reports only between requests. Measurement processes run without strace and with
`VIV_VX_ENABLE_PRINT_TARGET=0`; active `VIV_VX_PROFILE` or `VIV_MEMORY_PROFILE`
settings are rejected. Other SDK
internal activity is still included in the wall-clock observation.

| Metric | Boundary |
|---|---|
| Initialization | Context creation and complete model construction; excludes host weight-file verification |
| Prefill | Consuming all prompt IDs, after request reset and before the first head invocation |
| First token | Prefill plus the first device prediction/readback; excludes initialization, tokenization and reset |
| Decode interval | Consuming the previous predicted ID through the next device prediction/readback |
| Request time | Prefill and generation after reset, including native control overhead |
| Prefill throughput | Total measured prompt tokens divided by total measured prefill time |
| Decode throughput | Number of post-first-token intervals divided by their summed duration |

Warmups are validated but excluded from timing distributions. Reports retain each
request, the sample count, min/median/mean/p95/max and initialization separately.
Percentiles use linear interpolation; a few repeats are observations, not robust
population estimates. Do not compare these host times with a device-only kernel
benchmark or claim network/end-to-end serving latency.

The native runner samples current process RSS before/after model initialization,
after every request and after release, and records peak RSS separately. RSS may
remain elevated because the host allocator retains pages; it is not physical
board memory usage or proof of a device leak. Known weight/KV payload is reported
separately. SDK layouts/workspace and physical device peak remain null until a
supported measurement exists. Two fresh processes check complete lifetimes; they
do not by themselves establish leak-free long-running device allocation.

## Reports and evidence gates

- `suite.json`: policies, reference identities, prompts and the requested matrix.
- `selection/validation.json`: exact tiny selection and sharing correctness.
- `teacher/validation.json`: full-model intermediate, token, reset and recreation checks.
- `cycle-N-request-M/`: executable evidence, native execution/host RSS, each warmup
  and measured generation, and `benchmark.json`.
- `acceptance.json`: incremental suite status, failed/completed stages, measurements
  and the explicit remaining hardware gates.

The benchmark checks 104 uploaded control bytes per consumed token, four returned
bytes per prediction and 24 KV appends per consumed token. Warmups count toward
these totals. No hidden/cache/logit readback is allowed in the measured path.
Normal release, clean process exit and untraced execution are required before
publishing timing metrics. CPU references run outside device measurements.

No report flag or SDK success is accepted as a substitute for hardware proof.
The implemented result is `hardware_pending` after numerical success:

1. Identify the execution backend and completion of every kernel, including any
   software fallback or other on-chip engine.
2. Establish physical weight/KV residency and absence of SDK-internal per-token
   host state transfers.
3. Measure device peak, layout/workspace copies and allocation release.
4. Complete representative long-context checks and formal performance acceptance.

The local SDK review found target enumeration/affinity in `/usr/inc/VX/vx_ext_target.h`
and graph/node timing in `vx_types.h`. Target enumeration describes available
kernels; timing counters alone do not identify the physical backend. Node local/global
memory in `vx_khr_node_memory.h` is not a model residency counter. HAL profiler
headers contain counters but lack a verified NP101 model-allocation interpretation
and node-to-backend correlation procedure here. The vendor operator guide supplies
no such observation workflow. These APIs are not silently introduced into the
production path. Ask the chip team for a supported profiler/trace procedure and
memory-counter semantics, then validate and integrate it before closing these gates.

## Validation record

Earlier SDK numerical results remain in [the generation record](np101-generation.md); they do not replace
revalidation after a driver or library change.

On 2026-09-22 with package 1.0.6, the exact selection preflight and 24-layer,
capacity-512, eight-step teacher suite pass (23 and 491 checks respectively).
The new runner then passes two prompts × two fresh model processes, with one
warmup and two measured four-token continuations per process. All 48 returned
token IDs across 12 requests match CPU; those requests consume 120 tokens.
Application transfer totals, reset, release and process exit all pass. The suite
returns 2 / `hardware_pending`, as intended.

Unprofiled host-wall observations, with only four measured requests per prompt:

| Prompt | Prompt tokens | First-token median, two fresh processes | Decode-interval median, two fresh processes |
|---|---:|---|---|
| `The capital of France is` | 6 | 9.100 s / 9.210 s | 1.621 s / 1.614 s |
| `For dinner tonight, I will cook` | 8 | 12.198 s / 12.155 s | 1.605 s / 1.610 s |

Initialization ranges from 6.078 to 6.719 seconds. Observed decode throughput is
0.615–0.624 tokens/s. These short runs do not establish NPU throughput, long-context
performance or steady-state memory use. In fact, host RSS grows by about 3 MiB per
consumed token: 56,627,200 bytes over the two measured France requests and
69,214,208 bytes over the two dinner requests, repeated in both fresh processes.
This is tracked as `NP101-MEM-003` rather than declared a proven device leak.

A separate HAL accounting investigation used the declared/exported
`gcoOS_GetMemoryProfileInfo` from `HAL/gc_hal_base.h`. It was kept in local cached
diagnostic source, outside the production runtime. The need is allocation
observability, which the operator guide does not supply. The probe first checks a
small tensor, uses nonnull output storage, and never adopts the API as a default
inference path:

- With profiling disabled, status is success but the entire 80-byte output structure
  retains a sentinel pattern. Zero-initializing it and trusting success would report
  fabricated zero usage.
- With `VIV_MEMORY_PROFILE=1`, an 8 MiB constant tensor passes byte readback; SDK
  `gpu_memory.currentSize` rises by 16 MiB + 4 KiB and falls after graph release.
  Thus this count cannot be equated directly with unique tensor payload or physical
  resident bytes. The exit dump reaches zero in this control.
- In a separate full-model profiled run, all three nine-token requests return the
  four CPU-expected IDs. `gpu_memory.currentSize` stays at 1,454,438,928 bytes
  through the requests; its peak is 1,462,554,944 bytes and the final exit dump
  reports zero outstanding GPU-accounted bytes.
- `system_memory.currentSize` increases by 28,313,280 bytes per repeated request.
  Even the SDK exit dump reports 84,939,840 bytes and 1,296 allocations outstanding
  after 27 consumed tokens: 3,145,920 bytes and 48 allocations per token. This
  supports investigating retained/unreleased SDK host allocations; the originating
  call and ownership error are not yet identified. The per-slot KV graph
  re-verification path is a candidate, not an established cause.

Profiled observations are excluded from timing distributions. The benchmark runner
also rejects an active `VIV_MEMORY_PROFILE`. The chip team still needs to confirm
counter semantics, physical pool accounting and per-kernel backend evidence.

Local evidence (ignored by Git):

- `.cache/runs/acceptance-20260922-environment/`: host/SDK identity.
- `.cache/runs/acceptance-20260922-selection/` and
  `.cache/runs/acceptance-20260922-model/`: preliminary independent SDK health gates.
- `.cache/runs/acceptance-20260922-suite/`: complete suite, per-process snapshots,
  CPU references and raw timing/resource records. `acceptance-review.json`
  re-evaluates retained records with the final host-RSS-growth summary checks.
- `.cache/runs/acceptance-20260922-memory-profile/`: cached probe sources/binaries,
  8 MiB control, full-model counter observations and SDK exit dumps.

No sudo, driver modification or full-Qwen allocation retry was used. The separate
convolution SIGFPE and large-allocation data corruption remain unresolved; they
were not reproduced again during these OPT checks. Full formal acceptance stays
open for hardware evidence, memory-growth diagnosis and long-context validation.
Long-context/stress acceptance is deferred until the per-token allocation growth
is explained or corrected. The final build, 81 Python tests, three CTests and
Ruff/clang-format checks pass; these host checks do not close the hardware gates.
