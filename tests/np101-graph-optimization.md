# Fixed-graph OPT experiments

The experimental executable is `build/bin/specferry_opt_graph_generate`.
`generate_opt.py` continues to use the existing component-graph executable.
**The new path has compiled and passed host tests, but whole-model device
validation has failed.** On 2026-09-24 both small indexed-cache gates passed;
the complete decode graph returned zero tokens and zero KV. See the dated record below. Do not switch the
default generator or claim a throughput improvement from this implementation.
Later [pipeline diagnostics](np101-graph-pipeline.md) passed separate lookup and
one-layer KV checks, but a combined one-layer prefix timed out during verification.
That latest incident has an active recovery marker; no further device runs followed.

## Construction and contracts

- `native/models/opt/layer.*` constructs projections and post-norm decoder tails;
  both execution paths reuse these functions. Model ordering and checkpoint names
  remain in the OPT adapter. Shared projection/normalization operators accept
  `[width,tokens]`; one-column behavior is retained.
- `GraphModel` builds one complete decode graph and, for block sizes above one,
  one fixed prefill graph. Each contains token/position lookup, all 24 layers,
  cache updates, projection out, vocabulary head and greedy selection. A prefill
  block predicts from its last column; intermediate block predictions are unused.
- `WeightBank` owns one set of weight chunks. Consumer graphs retain references;
  application writes finish before compilation. Mutable SDK storage permits the
  existing retained-tensor API; weights remain read-only to inference. Matching
  payload counts do not prove the absence of SDK-private copies or workspace.
- Each attention head owns one K and one V allocation `[head_dim,capacity]`.
  `TENSORSTACKCONCAT` appends one column. A block of B tokens packs contiguous
  columns into `[head_dim*B,1]` and views the same destination allocation as
  `[head_dim*B,capacity/B]`. Capacity must be divisible by B. Both graphs retain
  the same parents. There is no application full-cache copy or second cache bank.
- `storage_reshape` uses the installed public `vxReshapeTensor` API. Its header
  specifies shared memory with a changed shape. It is outside the chip team's
  operator guide: the reason for this experimental use is to let block append
  and attention address the same allocation without full-cache reshaping copies.
  Dependency ordering and alias visibility on this SDK remain unvalidated.
- Every query in a prefill block gets its own causal limit. An incomplete final
  block uses single-token decode graphs, without padding or extra cache writes.
  Reset changes only logical position; a shorter request must mask old suffixes.
- One INT32 vector upload supplies token IDs, base position and append slot per
  launch. A prediction reads one INT32. Graph validity is checked after upload;
  invalidation is an error, with no per-token revalidation fallback. SDK-internal
  transfers are outside these application counters.
- The candidate supports checkpoint-default greedy decoding only; preparation
  rejects a different checkpoint policy. Existing explicit sampling is retained
  in the default generator. This experiment does not validate new model variants.

The cache writer rejects rank-three/multi-column/dtype mismatches before SDK node
construction. Only individual public node fields are changed; SDK private state
is preserved. Operator availability in `demo/ref_op_api_guide.md` does not establish
support for these shapes or their alias relationships.

## Offline preparation

From the repository root, with Conda `SpecFerry` activated:

```bash
cmake --build build -j 4
python scripts/check_np101_optimization.py --prepare-only \
  --output .cache/runs/graph-prepared
python -m unittest discover -s tests -t . -v
ctest --test-dir build --output-on-failure
```

Preparation opens no device. It verifies the OPT export/checkpoint relationship,
loads the official CPU model once, and saves token and all-layer KV expectations.
The same references serve repeated A/B/A requests and both capacity cases. The
32-token input is a natural-text prefix from `fixtures/opt_continuation.txt`.
`prepared.json` records the manifest and fixture hashes, generation policy,
reference-library identity, expected tokens/text and dimensions. A changed export
or fixture is rejected before device execution.

Existing local preparation from this work is
`.cache/runs/optimization-prepared-final-20260923/`; it can be reused after recovery
if the deployment and reference contract remain unchanged.

## Device validation, one command at a time

Before device work, confirm readiness and device permissions. An active recovery
marker blocks normal runs; do not clear it to force execution. The dated recovery
probe below records the user's explicit authorization and the archived incident.
After recovery, run the existing small startup check and inspect `selection.json`
for correctness, release and latency before proceeding:

```bash
python scripts/check_np101_acceptance.py --selection-only \
  --output .cache/runs/graph-startup
```

Then run only the one-column cache candidate:

```bash
python scripts/check_np101_graph_cache.py --mode stack \
  --output .cache/runs/graph-cache-column
```

The synthetic check produces K/V inside the graph, consumes them in attention,
checks attention against a host oracle, and checks all parent bytes. Its nonzero
query makes the result depend on K as well as V. A full prefix, shorter reset,
and another full prefix check causal masking and untouched suffixes.

Only if that passes, run the short whole-model decode check:

```bash
python scripts/check_np101_optimization.py --case decode \
  --prepared .cache/runs/graph-prepared \
  --cache-gate .cache/runs/graph-cache-column \
  --output .cache/runs/graph-decode
```

Next validate shared storage between block and single-token graphs, then the
whole-model block-prefill path:

```bash
python scripts/check_np101_graph_cache.py --mode stack-block \
  --output .cache/runs/graph-cache-block
```

```bash
python scripts/check_np101_optimization.py --case prefill \
  --prepared .cache/runs/graph-prepared \
  --cache-gate .cache/runs/graph-cache-block \
  --previous .cache/runs/graph-decode \
  --output .cache/runs/graph-prefill
```

Each inference gate requires the appropriate small gate, matching boot/SDK/build,
and the preceding inference result where applicable. Never chain stages after a
failure or increase a timeout without inspecting the current phase. No runner
starts the next stage automatically.

| Case | Prompt lengths | Block | Capacity | Predictions per request | Timeout ceiling |
|---|---|---:|---:|---:|---:|
| `decode` | 5, 3, 5 (A/B/A) | 1 | 16 | up to 4 | 360 s |
| `prefill` | 9, 5, 9 (A/B/A; includes tails) | 4 | 16 | up to 4 | 600 s |
| `capacity64` | 32 | 4 | 64 | up to 4 | 600 s |
| `capacity128` | identical 32 | 4 | 128 | identical outputs | 600 s |

The current ceilings include room for teardown. The first complete-graph trial
spent approximately 95 seconds initializing (SetupGraph 33.72 s, VerifyGraph
40.64 s), about 47 seconds generating, then more than 36 seconds in release.
Its former 180-second total timeout interrupted cleanup. Decode now allows 360
seconds and two-graph cases 600 seconds; these are ceilings, not expected waits.
Runtime defaults come from the current runner, including when reusing an older
prepared fixture. Explicit `--timeout` overrides remain available.

The native experimental runner now receives independent expected token rows.
After a mismatch it saves that request's KV diagnostics, skips later requests,
and enters normal release. `initialization.json` is saved before inference, so
initialization time and loaded payload survive a later failure. These runner
changes have only been built/host-checked; hardware validation remains pending.
They do not fix the all-zero model result.

The small cache timeout remains 120 seconds; both new candidates completed below
one second. Progress appears every 10 seconds. A timeout/signal still sets the
recovery marker. Do not clear it or repeatedly kill/relaunch a device process.

## Capacity comparison

After the prefill gate passes, run these separately:

```bash
python scripts/check_np101_optimization.py --case capacity64 \
  --prepared .cache/runs/graph-prepared \
  --cache-gate .cache/runs/graph-cache-block \
  --previous .cache/runs/graph-prefill \
  --output .cache/runs/graph-capacity64
```

```bash
python scripts/check_np101_optimization.py --case capacity128 \
  --prepared .cache/runs/graph-prepared \
  --cache-gate .cache/runs/graph-cache-block \
  --previous .cache/runs/graph-capacity64 \
  --output .cache/runs/graph-capacity128
```

The comparison rejects different input/output tokens, checkpoint policy,
manifest, boot, binary, SDK or runtime options. Both cases have exactly two
execution graphs; capacity changes do not create per-position graphs. Start with
one cold request and no warmup. These bounded observations guide further work;
they cannot establish stable benchmark performance. Do not directly claim speedup
against the old 16-output, warmed-up baseline using these four-output samples.

## Reading results

- `summary.json` and `results.md`: checks, prompt/continuation, decode tokens/s,
  prefill tokens/s, TTFT, initialization, RSS and capacity comparison. Metrics are
  withheld on validation failure. Decode throughput excludes prefill/first token
  and uses only the intervals between generated tokens; first-token EOS has none.
- `device/measured.N.json`: raw token IDs, consumed count, stop reason, times and
  per-request launches/transfer counters. Expected launches are
  `floor(prompt/B) + prompt%B + predictions - 1`, with one upload per launch.
- Decode/prefill gates read every layer's KV after each request, outside generation
  timing/counters. Valid prefixes use the existing CPU tolerance; repeated A
  prefixes must also be byte-identical. Capacity timing gates skip these diagnostic
  readbacks after the earlier KV gates, and still compare all output tokens.
- `device/sdk-timing.tsv`: summary-only public API durations. Request scopes must
  contain the expected launches/uploads/scalar reads, no revalidation and no SDK
  exceptions. Extra KV diagnostics use a separate scope. SDK totals must fit
  within measured host wall time. Strace/target printing/profilers are disabled
  for whole-model measurements.
- `device/host-resources.jsonl`: RSS before/after initialization, after each
  generation and after release. Peak RSS is in `device/run.json`. Physical device
  peak, SDK-private layouts/copies and execution engine remain unproven.
- `sources.json`, executable snapshot and `device/execution-evidence.json`:
  source fingerprints, binary/SDK identity, process exit and recovery state.

Shared graph construction also affects the existing OPT decoder and common
operators. After recovery, rerun a short existing OPT slice and retained Qwen
component regression before calling the refactor hardware-regression-safe. The
host suite and build alone do not replace those checks.

## Recorded prerequisite failures (2026-09-23)

| Attempt | Observed result | Local evidence under `.cache/runs/` |
|---|---|---|
| Scatter to separate output buffers | Ten append/reset steps and release pass, 1.69 s; full-cache output/copy is not a production solution | `o3-cache-control-20260923` |
| Scatter output aliases input via a full view | `VerifyGraph` returns -18, normal release, 0.83 s | `o3-cache-view-20260923` |
| Scatter output retains the identical input tensor | `VerifyGraph` returns -18, normal release, 0.83 s | `o3-cache-shared-20260923` |
| Rank-three, multi-head `TENSORSTACKCONCAT` | `SetupGraph` logs reshape 16→32 failure, then SIGSEGV, 1.78 s; no `RunGraph` | `o3-cache-stack-20260923` |

The installed `vx_types.h` names -18 `VX_ERROR_INVALID_GRAPH`; aliasing an output
with its graph input is consistent with a dependency cycle, but the SDK did not
supply a precise cause. The reshape failure does not establish whether the shape
is unsupported or the implementation is defective. The process group exited;
this signal alone does not prove a kernel/driver hang.

The historical `stack` binary is preserved with its hash and logs in the last
run directory. **Current `--mode stack` uses the new per-head two-dimensional
candidate and passed the small gate on 2026-09-24.** It is not a replay of that binary.
`separate`, `view`, and `shared` remain explicit diagnostics; there is no reason
to rerun the known rejections without a relevant SDK change.

Following the 2026-09-23 failure, the user requested offline work only. Device
checks resumed after the user confirmed a cold start on 2026-09-24, as recorded below.

## Cold-start validation and system logs (2026-09-24)

Boot ID: `e9e10f53-b6d8-450f-ae02-233f9c2b5702`. The permissions service succeeded;
`/dev/galcore` was mode 666. No sudo was used for any test. The user separately
authorized sudo journalctl reads after the failure; no driver/system changes followed.

| Check | Result | Device-process wall time | Evidence under `.cache/runs/` |
|---|---|---:|---|
| Existing selection | Numerical/lifecycle pass; no latency anomaly | 0.67 s | `o3-recovery-selection-20260924` |
| Existing OPT layer 0, four-step repeated sequences | 249 checks pass; normal release | 4.58 s | `o4-opt-regression-20260924` |
| Existing Qwen synthetic decoder | 353 checks pass; normal release | 4.53 s | `o4-qwen-regression-20260924/device` |
| Existing alternate KV layout | Pass; normal release | 0.27 s | `o4-qwen-regression-20260924/kv` |
| New 2-D single-column cache | 21 launches/tokens; history, reset, attention pass; zero revalidations | 0.82 s | `o3-cache-column-20260924` |
| New block/single-column shared cache | 6 launches for 21 tokens; causal mask and tails pass; zero revalidations | 0.87 s | `o3-cache-block-20260924` |
| Complete 24-layer graph | All three requests return `[0,0,0,0]`; all 144 K/V snapshots are zero; timeout during release | 180.22 s (terminated) | `o4-graph-decode-20260924` |

The original crashing three-dimensional case was not repeated. The passing small
cases do not establish correctness at model dimensions or complete-graph scale.
Full-model prefill and capacity comparisons were not started after decode failed.
Its 22 launches/uploads and 12 scalar reads match the request schedule, with no
per-step VerifyGraph, but successful SDK calls and correct counts do not imply
correct computation. All timing metrics are withheld because numerical and
lifecycle acceptance failed. CPU reference files were reused and verified.

The complete-graph runner reached release around process second 144 and was
terminated by its 180-second deadline with SIGTERM (`-15`), not SIGSEGV. Its process
group fully exited. Normal SDK teardown was not confirmed, so recovery protection
was retained until the separately authorized probe below. Do not interpret this
runner-induced timeout as proof of a driver hang.

Authorized system-log captures are in `graph-system-log-20260924/`:

- `journal-current-boot.log`: startup through 09:32:47, 18,754 lines. The driver
  reports `.ko probe success!`; module signature/taint and disabled-power-saving
  notices do not indicate a failed load. No NP101-related Oops, allocation failure,
  OOM, GPU timeout or segfault was found in the captured test window.
- The complete-graph PID 6633 has 13,053 `x86/PAT ... req write-back ... got
  write-combining` entries, principally during initialization. The six successful
  test processes also have 1,264 such entries in total. These record memory-mapping
  cache attributes; they are not tensor comparisons. Their relationship to the
  zero result, mapping overhead or earlier memory issues has not been established.
  See the [Linux 5.15 PAT documentation](https://www.kernel.org/doc/html/v5.15/x86/pat.html).
- `release galcore` appears at 09:29:23.341803, following timeout termination. This
  records device-file release; it does not prove every SDK/device resource is clean.
- `journal-prior-crash.log` covers the previous boot's 17:21–17:24 window. It shows
  device open/release but no additional kernel stack trace explaining the old SDK
  SIGSEGV. The old SDK reshape error and signal remain the more specific evidence.

Next validation should first isolate shared weight binding/readback, packed INT32
control slicing and blocked embedding lookup, then one decoder layer's projections
and cache writes. Use exact small fixtures and a single prediction before retrying
all 24 layers. No root cause is assigned to the SDK, driver or application yet.

After this failure, the runner was changed to stop subsequent requests after the
first token mismatch, save initialization timing before inference, and use current
timeout budgets even with older prepared fixtures. The native build, all nine
optimization host tests, and the C++/Python formatting checks pass. These safeguards
have not been rerun on hardware and do not fix the zero-result failure.

## Authorized minimal recovery probe (2026-09-24, same boot)

The user requested one minimal test to decide whether another cold start was
necessary. No reboot, driver reload or system change preceded this probe. Under
the diagnostic lock, the old marker was archived and the existing synthetic
selection fixture ran with a 60-second ceiling; a failure would retain the guard.

- Seven input cases and all 23 numerical/lifecycle checks pass; normal exit and
  release take 0.772 seconds in total, with no remaining process-group members.
- Eight `vsi_nn_ReleaseGraph` calls total 0.033 seconds; `vsi_nn_ReleaseContext`
  takes 0.053 seconds. No exception is recorded.
- Submission-thread galcore ioctls have a maximum of 0.0115 seconds and no wait
  at least one second. No latency anomaly is flagged.
- Evidence, the archived marker, and the one-off probe are under
  `.cache/runs/recovery-probe-20260924/`. The active marker is absent after success.
  This supports resuming small diagnostics without another cold start. It does
  not prove that the complete graph releases normally or computes correct values.

The failed complete graph's nine decode intervals total 19.0149 seconds, or
0.4733 intervals/s (2.1128 s/interval). The numerically correct O1 baseline was
0.3812 tokens/s (2.6230 s/token). Their arithmetic difference is about 24%, **not
a validated speedup**: the new output/KV is zero, prompt lengths are 5/3/5 rather
than 32, capacity is 16 rather than 64, and it has four predictions without warmup
rather than 16 after warmup. Raw observations are saved separately in
`raw-timing-observation.json`, not promoted into accepted performance metrics.
The new path does issue one graph launch per consumed token with no per-token
reverification; `vsi_nn_RunGraph` accounts for 99.89% of these failed decode
intervals. Correctness and a matched comparison remain prerequisites for claiming
an inference throughput improvement.
