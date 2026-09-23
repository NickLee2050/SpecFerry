# SDK latency and inference progress

Use the current build and the `SpecFerry` Conda environment. Run one board process
at a time, with a fresh output directory. No sudo is needed after the administrator
has granted access to `/dev/galcore`.

## Small startup check

```bash
python scripts/check_np101_acceptance.py --selection-only \
  --output .cache/runs/selection-startup
```

This uses seven exact cases with synthetic FP16 weights (embedding width 8,
vocabulary 19); it requires no downloaded model. It checks lookup, projections,
head/argmax, byte results, invalid inputs and explicit cleanup. It is the same
selection gate used by full-model acceptance, not another arithmetic implementation.
`--prepare-only` creates its fixture without accessing the device.

The selection timeout defaults to 120 seconds; the other SDK processes retain
`--timeout` (default 1800 seconds). A completed selection taking over
`--selection-warning` (default 30 seconds), or a traced galcore ioctl of at least
one second on the native submitting thread, is reported as a latency anomaly.
Correct bytes do not clear this flag. Full acceptance stops before loading the model
if selection is anomalous.
The thresholds identify outliers in this tiny fixture, not a general SDK latency SLA.

On a known slow startup, collect a bounded diagnostic instead of repeatedly
starting the full model:

```bash
python scripts/check_np101_acceptance.py --selection-only --sdk-timing calls \
  --selection-timeout 1800 --output .cache/runs/selection-calls
```

This explicitly allows enough time for the previously observed repeated waits;
it does not disable the process timeout or recovery guard. Inspect:

- `selection.json`: independent numerical checks, lifecycle, latency flag and
  boot/SDK/binary identities. The fixture fixes all dimensions and precision.
- `selection/sdk-calls.tsv`: flushed begin/end records, request/component/API and
  returned status or pointer. An unmatched begin identifies the last instrumented
  call that has not returned. Begin timestamps are process-relative; end times
  are durations. SDK internal commands and kernel execution are not inferred.
- `selection/driver.strace`: completed galcore ioctl durations; `execution-evidence.json`
  reports the native entry thread's count, sum, maximum and number lasting at least
  one second. SDK/shader worker threads are reported separately: a background wait
  spanning the process lifetime is not treated as a stalled submission. Thread
  durations can overlap and must not be added as sequential request time.
- `selection/sdk.log`: original SDK messages, preserved unchanged.

Per-call file writes and tracing affect timing. This mode is a fault-localization
tool, not a throughput benchmark. If the cold-start anomaly cannot be reproduced,
retain the earlier logs and mark it unconfirmed; do not claim that the driver was fixed.

## Whole-model call timing

```bash
python scripts/check_np101_acceptance.py --lengths 32 --sdk-timing summary \
  --output .cache/runs/opt-32-sdk-time --diagnostic
```

The command uses all 24 OPT layers and the checkpoint generation policy. Defaults
are two fresh processes, one warmup and three measured requests per process, with
16 predicted tokens. All correctness gates still run. SDK call observation is
off by default; use an otherwise identical `--sdk-timing off` run to assess overhead.
`--sdk-timing calls` is restricted to the standalone selection diagnostic.

`summary` accumulates public SDK call times in native memory. It adds no tensor
readback, profiler, kernel trace or per-token file output. Each completed request
publishes a cumulative `sdk-timing.tsv` outside its measured interval. Rows contain
request, phase, component, API, call count, total seconds, maximum seconds and C++
exception count. SDK status values are handled by existing callers; the exception
column is not a count of all negative SDK return codes.

Phases separate initialization, prefill, first prediction, decode and normal
explicit release. Components separate embedding lookup/projection, each decoder
layer's QKV, KV and Attention/FFN, output projection and LM head. Attention and FFN
remain one SDK graph call; this work does not split it just to obtain finer timing.
Tensor upload/readback calls appear separately within their owning components.
Only the main submitting thread's selected call boundaries are observed; internal
SDK calls, workers and kernel time are outside this instrument's scope.

`benchmark.json` retains the rows and measured phase/component totals. Warmup,
initialization and release rows are excluded from these measured totals. Acceptance
also checks RunGraph/copy/upload/readback call counts against actual consumed and
predicted tokens. `results.md` prints phase totals, API counts/times/maxima and the
largest components next to TTFT, tokens/s and generated text. Do not add these API
times again to request time: they are portions of the same host wall interval. The remainder includes
unobserved calls, host work and observer overhead.

## Progress and abnormal exits

The Python supervisor prints a heartbeat every ten seconds (`--progress-interval`,
zero disables). It shows the current native phase, such as `warmup.0` or `measured.2`,
process elapsed time and phase elapsed time when available. Older binaries or
initialization may only provide a process heartbeat. Native progress snapshots are
replaced atomically; heartbeat reads do not enter the device or measured C++ path.
The starting execution evidence is written before waiting, so an unfinished run
still identifies its process, boot, command and SDK libraries.

A native signal, timeout or remaining process still requires device recovery.
Ctrl+C terminates the supervised process group and records a user interruption,
even if a child signal handler exits zero; it does not leave an unobserved SDK child.
The supervisor does not interpret a successful numerical check as hardware health,
automatically reboot the board, change SDK settings, or continue past the recovery guard.

## Related independent diagnostics

| Observation | Entry point | Record |
|---|---|---|
| Long startup waits | Selection command above; reduce further only if it reproduces after cold start | This document |
| Host allocation growth on graph revalidation | `check_np101_memory.py ... growth` | [Memory diagnostics](np101-memory.md) |
| SDK device-memory accounting versus payload | `check_np101_memory.py ... accounting` | [Memory diagnostics](np101-memory.md) |
| Fixed retained-byte corruption | `check_np101_capacity.py --readback all ...` | [Memory diagnostics](np101-memory.md) |

These are separate runnable experiments. Existing resource helpers, evidence and
recovery handling are shared; no common root cause is assumed.

## Cold-start baseline, 2026-09-23

The user cold-started the host/board and restored device permissions. Boot ID is
`b4805626-9314-440e-9574-5121b031b50b`. The isolated first selection passes in about
2.4 seconds (`.cache/runs/o0-cold-selection-20260923/`). The subsequent full suite
in `.cache/runs/o0-cold-32-20260923/` passes selection, teacher, boundary, KV and
A/B/A checks, then two fresh 32-token model processes with one warmup and three
measured 16-token continuations each. All tokens match CPU and release normally.

| Process | Initialization | Decode tokens/s | TTFT median | KV write share of measured request time |
|---|---:|---:|---:|---:|
| 0 | 47.990 s | 0.3695 | 82.350 s | 20.06% |
| 1 | 47.759 s | 0.3728 | 82.135 s | 19.81% |

Both native binary hashes, the five recorded SDK library hashes and recorded runtime
options match the pre-cold-start run `.cache/runs/opt-32-20260923-145020/`.
That earlier run had 36 slow submitting-thread ioctls in selection (maximum
30.721 seconds), five in teacher, then measured about 0.655 tokens/s and a
45-second TTFT. The cold suite's traced submission maxima are 0.076 seconds in
selection, 0.113 seconds in teacher/boundary and below 0.138 seconds in KV checks.
Generation is untraced, so no per-ioctl conclusion is drawn for those requests.

SDK worker threads can wait for almost the entire process lifetime. They are
retained separately in the derived trace summary; including them in a submission
latency guard would produce a false alarm. The final parser and its regression
test handle interleaved unfinished/resumed calls and this distinction.

The fixed approximately 30-second submitting-thread waits did not recur in these
cold-start checks. The separate low-throughput problem remains, and KV's measured
share increased relative to the earlier run. No driver repair or root cause is
claimed. Further reduction of a waiting reproducer is conditional on recurrence.

## SDK observation comparison, 2026-09-23

After O0 passed, the new binary ran serially with observation off and then summary
on, using the already prepared CPU fixture. Both runs use the same native binary,
boot, SDK hashes, 32-token input, capacity 64, one warmup and three measured
16-token continuations. No CPU reference preparation or build overlapped them.
All token IDs, transfer counts, cache writes and normal release checks pass.
Summary API counts also match the actual consumed and predicted token counts.

| Mode | Initialization | Decode tokens/s | TTFT median | Three measured requests |
|---|---:|---:|---:|---:|
| off | 46.579 s | 0.3789 | 81.003 s | 361.854 s |
| summary | 45.777 s | 0.3812 | 80.349 s | 359.497 s |

Summary's decode rate is 0.63% higher and total request time 0.65% lower in this
ordered pair. This small sample shows no conspicuous observer penalty; it does
not establish a speedup or an exact overhead bound. Initialization and warmup
are excluded from the following measured totals:

| Public SDK call | Count | Total time | Share of request time | Maximum call |
|---|---:|---:|---:|---:|
| `vsi_nn_RunGraph` | 7,146 | 254.910 s | 70.91% | 0.1171 s |
| `vxVerifyGraph` | 3,384 | 48.757 s | 13.56% | 0.0359 s |
| `vsi_nn_CopyDataToTensor` | 3,666 | 33.721 s | 9.38% | 0.0295 s |
| `vxProcessGraph` | 3,384 | 21.536 s | 5.99% | 0.0213 s |
| `vsi_nn_ConvertTensorToData` | 48 | 0.337 s | 0.09% | 0.0210 s |
| KV parameter rebinding, both slots | 6,768 | 0.037 s | 0.01% | 0.000054 s |

The observed SDK calls cover 359.299 of 359.497 measured seconds, approximately
99.94%. Prefill accounts for 240.944 SDK seconds, first prediction for 0.388 and
decode for 117.967. No measured instrumented SDK call exceeds 0.118 seconds.
These are synchronous public API wall times, not a decomposition of launch,
transfer, SDK CPU work and kernel execution inside each call.

Each consumed token uploads 26 four-byte scalar values. Across the three measured
requests this is only 14,664 application bytes, yet the corresponding upload calls
take 33.721 seconds. Each ordinary decode step currently executes 52 `RunGraph`
calls plus 24 KV `ProcessGraph` calls, and revalidates 24 KV graphs. These counts
and durations motivate O3/O4's shared-cache and whole-graph work; they do not
predict its speedup or justify changing the cache mechanism without validation.
Measured host RSS still grows by about 423 MiB in each mode; MEM-003 remains open.

Evidence: `.cache/runs/o1-timing-off-20260923/` and
`.cache/runs/o1-timing-summary-20260923/`. Each contains readable `results.md`,
`benchmark.json`, native outputs and execution evidence. The final standalone
begin/end trace also passes in about 2.3 seconds under
`.cache/runs/o1-selection-final-20260923/`. O2's further waiting-reproducer
reduction was not triggered; the three independent memory diagnostics are
retained without repeating their known failures.
