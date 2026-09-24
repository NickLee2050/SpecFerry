# Engineering follow-up

Updated: 2026-09-24. Active target: complete resident OPT-350M text inference on NP101.
Numerical SDK validation and formal hardware acceptance are tracked separately.

Next implementation steps and closure criteria:
[remaining S11 acceptance checklist](docs/remaining-acceptance-plan.md).
SDK observation and optimization work is tracked as
[O0–O6](docs/optimization-plan.md). Cold-start tests on 2026-09-24 passed the two cache gates and
existing short regressions; the complete graph failed numerically and timed out
during teardown. A subsequently authorized small recovery probe and three pipeline
checks passed. After a further cold boot, the segmented-memory synthetic lookup
triggered kernel hang/recovery reports during verification and after process exit.
Its new recovery marker blocks device work. Whole-graph acceptance remains open.

2026-09-24 follow-up: const and non-const application tensor payloads are now
limited separately to 1 GiB, including shared-reference/temporary-backing accounting.
Capacity requests above 1024 MiB and oversized weight/state fixtures are rejected
before device submission. Pipeline checks audit all shared weights across setup,
verification and execution. This is an application policy, not a driver rollback
or bad-address exclusion; hidden SDK memory remains unknown.

- [x] Implement segment budgets and complete shared-weight readback diagnostics.
- [x] Confirm the subsequent cold boot and attempt the first segmented lookup.
  Weight readback passed in three phases; VerifyGraph returned after 22.372 s,
  but the first RunGraph was interrupted by the 30 s total deadline.
- [x] Read one explicitly authorized kernel-log window: NP101 hang/automatic
  recovery at 11:00:51 and again at 11:01:21, after test exit. Record the new guard.
- [ ] After recovery, compare the previous passing tiny path and added full-weight
  readback schedule, then resume [pipeline diagnostics](tests/np101-graph-pipeline.md).
  The lookup deadline is now 120 s, but no longer-budget retry was attempted after
  the hang evidence. All subsequent device gates remain unrun in this boot.

## NP101-OP-006: Validate graph-integrated cache append

- [x] Record scatter control success, alias rejection and the rank-three
  TENSORSTACKCONCAT setup crash with executable/SDK/source fingerprints.
- [x] Reject the known crashing geometry before SDK calls; implement an experimental
  per-head 2-D column writer and shared storage reshapes for block/decode graphs.
- [x] Complete offline fixed-graph OPT construction, block prefill, CPU references,
  host tests and bounded staged runners. Keep the existing generator as default.
- [x] Confirm cold start on 2026-09-24 and the working device permission service.
- [x] Run the user-authorized minimal health probe after the teardown timeout; archive
  the old marker only after numerical, normal-release and latency checks pass.
- [x] Validate 2-D append/attention, history, causal masks, resets and release in the small fixture.
- [x] Validate block/decode cache sharing in the small fixture without per-step revalidation.
- [ ] Fix complete-graph all-zero token/KV results before full prefill/capacity tests.
- [ ] Revalidate teardown with the revised budget and stop-on-token-mismatch runner.
- [x] Recheck existing OPT (249 checks), Qwen (353 checks) and alternate KV layout on 2026-09-24.
- [x] Check packed inputs, shared weights and blocked lookup (38 synthetic / 13 real checks)
  and one real joint decoder layer (30 checks, nonzero KV and normal release).
- [ ] Check device readiness after the new combined-prefix VerifyGraph timeout;
  preserve its separate incident and recovery marker.

The separate-output scatter control passed in 1.69 s. Both full-view and identical
retained-reference scatter aliases returned `VX_ERROR_INVALID_GRAPH` (-18) at
VerifyGraph and released normally. The rank-three TENSORSTACKCONCAT case logged
`Cannot calculate the reshape tensor 16 to 32`, then SIGSEGV during SetupGraph;
no RunGraph occurred. Its process group exited, which does not prove device health
or a kernel hang. Evidence is under `.cache/runs/o3-cache-*-20260923/`.

Both current 2-D small gates passed on 2026-09-24 (0.82/0.87 seconds). The full
decode graph completed three requests but returned zero token IDs and zero KV in
all 24 layers. It reached teardown around second 144; the 180-second timeout sent
SIGTERM during release. All processes exited, but normal SDK teardown was not
confirmed. O4 correctness/teardown, O5 full-model prefill and O6 performance remain
blocked; successful small graphs do not establish full-model support.

The runner now stops after the first token mismatch, retains its KV evidence and
releases normally; ceilings are 360 seconds for decode and 600 for two-graph cases.
These fixes need device validation. Authorized journal reads found 13,053 PAT
mapping-attribute notices for the failing PID, with the same message class in
passing tests, and a device-file release entry after termination. No corresponding
NP101 kernel crash/OOM/timeout was found; no root cause is established.
The subsequent same-boot selection probe passed 23 checks in 0.77 seconds, including
normal graph/context release, with no one-second driver waits. The old marker and
probe evidence are in `.cache/runs/recovery-probe-20260924/`. This permits further
small diagnostics at that point; it did not validate complete-graph teardown or repair its zero outputs.
Three subsequent diagnostics passed, but the one-layer production graph prefix
(embedding/head included, capacity 64) spent over 110 seconds in VerifyGraph and
was terminated at the 120-second process deadline. It never reached RunGraph and
its process group exited. The new marker is retained; do not infer that the prior
health probe covers this later event. The capacity differs from the original
failing complete graph (16), so this is not an isolated layer-count comparison.
Next isolate tiny lookup/head multiple consumers and compare at the same capacity.
See [pipeline diagnostics](tests/np101-graph-pipeline.md) for evidence and header options.
Detailed staged commands and acceptance criteria:
[fixed-graph experiments](tests/np101-graph-optimization.md).

## NP101-OP-005: Repeated long SDK waits after startup

- [x] Preserve the startup selection with 36 galcore ioctl calls lasting at least
  one second (many approximately 30.7 seconds); completed numerical checks still pass.
- [x] Add a checkpoint-free selection entry, progress heartbeat, separate short-gate
  timeout/latency warning, and optional SDK begin/end records.
- [x] Provide SDK summary timing by request, phase, component and public API without
  introducing tensor readbacks or interpreting SDK-internal behavior.
- [ ] Determine the cause with the chip team if cold-start checks reproduce it.

The historical run `.cache/runs/opt-32-20260923-145020/` has about 1,097 seconds of
completed galcore ioctl time in selection. Teacher also has five calls above one
second; its later boundary and generation stages recover. This is separate from
the normal low token throughput and the three memory observations. The cold-start
comparison under `.cache/runs/o0-cold-32-20260923/` passes all correctness gates and
two complete 32-token model lifetimes. The traced submitting thread no longer has
calls over one second; SDK worker waits are reported separately. Decode remains
slow (0.3695/0.3728 tokens/s; TTFT medians 82.350/82.135 seconds), despite identical
native binaries and SDK hashes to the earlier baseline. Commands and stopping rules are in
[SDK latency diagnostics](tests/np101-sdk-timing.md). Do not claim root-cause repair
from a successful cold-start run. O2's further reduction is conditional on recurrence.

O1's same-binary off/summary comparison passes token, transfer, API-count and
lifecycle checks (0.3789/0.3812 tokens/s; TTFT 81.003/80.349 seconds). Of measured
request time, `RunGraph` accounts for 70.91%, KV `VerifyGraph` for 13.56%, scalar
uploads for 9.38%, and KV `ProcessGraph` for 5.99%. This locates the time at public
API boundaries, without identifying SDK-internal launch/compute costs. O3–O6 now
have offline implementations; NP101-OP-006 blocks device acceptance. Host growth
and physical residency remain open.

## Current work and dependencies

| Work | Current state | Next action / dependency |
|---|---|---|
| OPT input/output and generation (S9/S10) | Implemented; full 24-layer numerical/lifecycle regression also passes on package 1.0.6 | Retain the regression; the independent convolution fault remains open |
| Short-context DLM acceptance (S11.4/S11.5) | KV/reset/boundary checks pass; 32/128/512/2033-token observations complete | Cold-start 32-token repeat complete; 2048-token case remains cancelled; memory/hardware-proof issues remain separate |
| Execution/residency evidence (S11.3) | Deferred at the user's request; hardware proof remains open | Resume profiler and physical-memory accounting later; not a prerequisite for bounded functional checks or host-wall timings |
| Long context and compression | Deferred to a later joint work item | Design separately; not part of the current short-context acceptance |
| Full Qwen weight loading | Allocation succeeds on package 1.0.6, retained bytes fail validation | NP101-MEM-001; also account for deferred DeltaNet duplication before any full-model integration |
| Qwen component regression | Existing implementation and evidence retained | Recheck affected components after relevant changes; no new full-Qwen integration in the active OPT scope |
| TLM/protocol and speculative inference | Deferred | Complete the DLM acceptance stage before joint inference and optimization measurements |

Host implementation, references, export verification and unit tests may continue
while device issues are unresolved. Earlier numerical passes are not acceptance
of a changed driver/SDK package. Do not repeat known failing capacity/compute runs
without a relevant change or a concrete diagnostic hypothesis.

## NP101-MEM-001: Validate retained tensor data above 1 GiB

- [x] Recheck constant and mutable storage using equal-byte FP16/FP32 diagnostics.
- [x] Capture failed bytes and explicit cleanup results for synthetic and actual weights.
- [x] Separately measure the accepted allocation bound and scan all retained blocks,
  continuing past byte mismatches as explicitly requested on 2026-09-23.
- [ ] Obtain a vendor-supported correction or explanation of the corruption.
- [ ] Revalidate retained bytes and normal release before increasing the deployment budget.
- [ ] Revalidate all Qwen weights and the historical allocation-state fixture together.

Package 1.0.6 accepts all 502 Qwen weight/state tensors (1,550,863,040 bytes),
but the first readback mismatch is in `model.layers.17.mlp.gate_proj.weight`.
The strengthened checks locate it at byte 3,400,312 within the first 7 MiB chunk;
all 46,071,808 zero-initialized state bytes still compare correctly.

All four 64 MiB controls pass. All four 1,152 MiB runs (FP16/FP32 × constant/mutable)
fail at the same three eight-byte regions of block 109, starting at byte 1,274,680.
The 2026-09-23 extension retains 2,840 MiB in all four combinations, then rejects
the next 8 MiB block (constant: AddTensor; mutable: upload status -5,
`VX_ERROR_NOT_ALLOCATED`). Complete scans of all 355 retained blocks find only
the same 24 differing bytes, all read back as zero. All remaining bytes compare
correctly and cleanup completes. This is an initialized-payload boundary at
8 MiB granularity, not proof of physical exhaustion or usable model capacity.
The largest earlier fully verified payload is 864 MiB in constant mode, a tested
lower bound rather than a physical limit. The old 1 GiB allocation ceiling is
historical; the current failure is retained-data integrity. Physical pool selection
and usable capacity near 4 GiB remain unverified.

Evidence and reproduction:
[capacity diagnostics](tests/np101-capacity.md),
[old constant/mutable comparison](tests/np101-mutable-allocation.md),
`.cache/runs/allocation-review-20260922/`.
The extended allocation and corruption-map evidence is in
`.cache/runs/capacity-map-20260923/`.

Closure checks after a relevant vendor change:

1. Record the matching driver/SDK versions, supported allocation path, ownership
   rules and any pool configuration. System changes require specific authorization.
2. Start with small FP16/FP32 constant/mutable controls. Increase only after full
   byte readback, explicit graph/context release and clean process exit.
3. Repeat the full weight/state fixture with a fresh output directory. Verify every
   retained weight and state byte, not just allocation/upload status.
4. Repeat initialization/readback/release with supported memory observations and
   investigate any growing footprint or failed cleanup.

Close this allocation item after a supported path passes those checks. Full-model
workspace and exclusive-NPU residency belong to S11 and NP101-OBS-001; they are
not prerequisites for closing the allocation diagnostic itself. OPT's smaller
payload needs its own full-graph memory checks and is not automatically blocked
or validated by the Qwen result. Per-token host weight streaming does not satisfy
resident-inference acceptance.

## NP101-OP-004: Diagnose the new-package convolution SIGFPE

- [x] Locate the fault in `libNNArchPerf.so:NNTransposeCycleCount_V9` at an integer
  division instruction and verify the installed files against the supplied package.
- [ ] Determine the failing operand/configuration with the chip team and correct
  the caller or SDK path as indicated by evidence.
- [ ] Revalidate changing-input convolution/ReLU/pooling, cleanup and process exit,
  on a corrected path.
- [x] Independently recheck exact OPT selection and the complete 24-layer numerical
  path on package 1.0.6; both passed on 2026-09-22 without changing the convolution test.

Caller/SDK root cause remains unresolved; matching package files do not establish
correct computation. This is separate from the allocation-only readback failure.
See the [preflight record](tests/np101-capacity.md).
This item gates convolution acceptance. The passing OPT regression shows that it
does not currently block the tested OPT graph path; neither result resolves the
independent allocation corruption.

## NP101-OBS-001: Establish execution and device-memory evidence (S11)

Priority update, 2026-09-23: focus now on short-context KV correctness, complete
responses and host-wall performance. Do not enable a profiler during inference measurements. The isolated one-tensor
accounting diagnostic explicitly enables SDK memory counters.
Per-kernel backend, physical residency and device accounting remain deferred proof
obligations; they do not block those functional checks or preliminary measurements.
Explore natural-text prefixes of 32, 128, 512 and 2048 tokens now; full long-context
quality, sustained resource stability and context compression remain separate work.

- [x] Implement staged correctness gates, checkpoint-default CPU token comparisons,
  same-instance warmups/repeats and fresh-process model lifetimes.
- [x] Separate initialization, prefill, first-token and decode wall times; retain
  counts, distributions, explicit transfers and host RSS outside timed requests.
- [x] Keep failed/incomplete runs out of timing summaries and retain missing
  hardware evidence as an explicit blocking result.
- [ ] Obtain a supported trace/profiler that identifies each kernel's execution
  backend and completion, plus physical allocation/residency observations.
- [ ] Account for weights, KV, activations, SDK layout copies and workspace with
  the complete OPT model resident at the tested capacity.
- [ ] Verify release/recreation against device memory observations.
- [x] Complete S11.4 short-context KV/prefix checks and same-instance A/B/A reset
  validation against independent CPU results, preserving complete-model execution.
- [x] Measure short-context initialization, prefill, model-ready TTFT and decode
  after functional correctness passes; report RSS growth and exclude tracing overhead.
- [x] Separate functional/timing results from deferred hardware proof and long-context
  work in the runner's reports; preserve strict hardware-pending status and flags.
- [x] Complete natural-text 128/512/2033-token input observations and near-limit
  continuation, recording actual output length, TTFT, throughput and readable text.
- [x] Repeat the 32-token timing without overlapping CPU reference preparation,
  after device recovery is confirmed; retain the separate cold-start O0 baseline.

The user cancelled the separate 2048-token input during prefill on 2026-09-23;
it has no completed timing or prediction and is not automatically scheduled again.
Its process group exited; the signal-triggered recovery marker remained until the
user's subsequent cold start. O0 then passed under a new boot ID. The cancelled
2048-token case still has no completed result and will not be restarted automatically.

Implementation, run commands and current observations:
[complete-model acceptance](tests/np101-acceptance.md). The runner can collect
untraced host observations now; they are not formal NPU performance measurements.
The expanded short suite passes 2,027 checks each for the natural teacher and
capacity-eight boundary, A/B/A reset/reuse, and two fresh 32-token model lifetimes
with one warmup plus three measured requests. Decode is 0.6633–0.6648 tokens/s;
TTFT medians are 45.12–45.16 seconds. KV writing is about 2.4% of request time.
This does not explain the remaining slow model execution. Host memory growth is tracked separately as NP101-MEM-003.
The independently prepared 128/512/2033-token cases each return 16 CPU-identical
tokens; decode rates are 0.6520/0.5974/0.5003 tokens/s. See the acceptance record
for TTFT, output text, sample counts and the cancelled case's evidence.

Generic driver IO, SDK success, numeric agreement and `argmax.execute_on_sw=false`
do not establish exclusive NPU execution. Host RSS/FD stability is not a board
memory measurement. Existing reports retain this distinction.

Known OPT weight/KV payload at capacity 512 is 712,724,480 bytes (679.71 MiB),
excluding SDK overhead, activations and workspace. Use the
[complete-model checks](tests/np101-generation.md) as the numerical baseline.
This item gates claims of proven execution backends, physical residency and NPU-only
performance. It does not gate reporting measured application/SDK wall times.

## NP101-STATE-001: Establish physical residency of reused state

- [x] Implement numerical state reuse/reset without per-token application state copies.
- [x] Validate DeltaNet recurrence/convolution state, Attention/KV append, untouched
  rows, valid-prefix bounds, reset, recreation and final-only readback.
- [x] Integrate fixed activation sharing into Qwen slices and full OPT generation.
- [ ] Establish that SDK-internal transfers do not shuttle state through the host
  during normal steps, using NP101-OBS-001 evidence.

Retain the existing DeltaNet two-bank implementation and single-buffer KV design.
Do not restore handle swapping or the retired SDK RNN/delay/flush test matrix.
Dynamic KV allocation remains deferred. KV truncation does not roll back recurrent
state; recreate a model after a partial SDK failure.

Numerical/lifecycle records: [DeltaNet](tests/np101-delta-net.md),
[Attention/KV](tests/np101-attention.md), [Qwen decoder](tests/np101-decoder.md),
[OPT generation](tests/np101-generation.md).
The [state investigation archive](tests/state-feedback-investigation.md) preserves
retired experiments. The remaining residency check gates formal device-resident
sequential inference acceptance, not implementation of already validated state routing.

## NP101-MEM-003: Explain per-token host allocation growth

- [x] Observe repeatable RSS growth during full OPT generation with persistent models.
- [x] Independently validate the exported HAL memory-profile counter with an 8 MiB
  allocation/readback control, then sample a separate full-model run.
- [x] Reproduce growth without a model: one copy node, preallocated views; fixed,
  same-destination and advancing-destination controls. RSS increases across reverify.
- [x] Provide integrated readable commands for host growth, one-tensor GPU accounting
  and existing large-allocation integrity: [memory diagnostics](tests/np101-memory.md).
- [ ] Locate the allocation source and determine whether it is retained workspace,
  missing release or another SDK/application ownership issue.
- [ ] Verify bounded steady-state host allocation and release after the correction.

The unprofiled suite grows approximately 3 MiB of host RSS per consumed token.
With `VIV_MEMORY_PROFILE=1`, three nine-token requests increase SDK
`system_memory.currentSize` by 28,313,280 bytes per repeated request. The SDK's
exit dump still reports 84,939,840 bytes and 1,296 allocations not freed after
27 consumed tokens: 3,145,920 bytes and 48 allocations per token. This is stronger
evidence than RSS alone, but does not yet identify the allocating call or root cause.
Application transfer counts and tokens remain correct. Ask the chip team to
interpret these counters and investigate allocation stacks/lifetimes, including
the per-slot KV graph verification path. The new single-node controls reproduce
growth across re-verification, but the responsible allocation and release contract
still need explanation; temporal localization alone is not a root-cause diagnosis.

SDK `gpu_memory` accounting stays constant during the three requests and returns
to zero in the exit dump. Its roughly 1.36 GiB peak is not accepted as physical
NP101 memory: an 8 MiB control produces roughly 16 MiB of counted allocations.
These counters remain diagnostic only, outside the default inference/benchmark path.
The source, binary and logs are retained under
`.cache/runs/acceptance-20260922-memory-profile/`.
This issue gates long-running resource stability. The revised S11 plan permits
bounded short-context KV/generation checks and preliminary timing while recording
the growth. Short repeated runs remain bounded; the requested longer exploratory
prefixes use one request per fresh process and check available host headroom using
4 MiB per consumed token plus a 4 GiB reserve. This estimate is not a proven bound. Limited localization
uses existing HAL evidence and non-profiled RSS/call observations first. Unknown
ownership remains open if vendor allocation details are needed. A normal process
exit is not a fix. See the linked plan for closure and stopping criteria.

## NP101-MEM-004: Explain SDK device-memory accounting overhead

- [x] Measure one 1/8/32 MiB FP16 tensor in both constant and mutable storage, with
  complete readback and release, independently of any model or compute graph.
- [x] Provide `check_np101_memory.py accounting` and its native test in the existing
  test tree; sentinel-check whether the SDK actually populates its output.
- [ ] Obtain the meaning of `gpu_memory` counters and explain the observed
  `2 * payload + 4,672` byte delta; distinguish physical allocation, layouts and accounting.

This is separate from host allocation growth (MEM-003) and corrupt data (MEM-001).
No common root cause has been established. It gates interpreting counters as
physical board occupancy, not bounded functional or host-wall performance checks.
See [commands and observations](tests/np101-memory.md). The isolated diagnostic
uses memory profiling; inference measurements do not.

## NP101-MEM-002: Share DeltaNet weights across alternating graphs

- [ ] Eliminate duplicate weights after the user resumes this deferred optimization.

Explicitly deferred on 2026-09-21. Do not fuse graphs or change weight ownership
as part of routine cleanup. The first four Qwen layers load 218.68 MiB of weight
payload versus 158.35 MiB unique weights: 60.33 MiB is duplicated by three DeltaNet
mixers. Separate allocation causes duplication; graph fusion alone does not prove
sharing or physical savings.

When resumed, validate shared read-only storage in both graph directions,
references/ownership, numerical trajectories, reset and release. Measure actual
SDK copies/layouts. Before future full-Qwen integration, either account for every
duplicated layer or remove duplication through a validated path. This does not
block OPT, and its resolution would not fix NP101-MEM-001. Baseline:
[decoder memory record](tests/np101-decoder.md).

## NP101-OP-003: Correct direct FP16 categorical sampling

- [x] Reproduce incorrect indices with direct FP16 RANDOM_MULTINOMIAL.
- [x] Validate FP32 logits and exact FP16-to-FP32 promotion for optional OPT sampling.
- [ ] Obtain a vendor correction and revalidate before enabling direct FP16 sampling.

The validated promoted path remains in use; model weights stay FP16. Default
text generation follows the checkpoint policy. This vendor issue does not block
that path, but its execution backend still needs NP101-OBS-001 evidence.
See [sampling reproduction and results](tests/np101-sampling.md).

## Completed implementation and retained evidence

| Item | Delivered behavior | Record |
|---|---|---|
| NP101-MODEL-001 | Model-specific configuration/reference/composition separated from reusable data, operators and storage | [Completed decoupling checklist](docs/model-decoupling-plan.md), [component regression](tests/np101-components.md) |
| NP101-MODEL-002 | Original FP16 OPT import/export and decoder slices; documented MatMul/Add avoids the biased-FCL optimizer crash | [OPT validation](tests/np101-opt.md) |
| NP101-MODEL-003 (S9/S10) | Shared embedding/head, all 24 layers, persistent KV, checkpoint-default generation and optional sampling | [Complete-model validation](tests/np101-generation.md) |

Qwen checkpoints, exported weights, references and historical validation artifacts
remain available as regression baselines. Future Qwen embedding/head or full-model
integration is separate work, subject to its memory and state gates; it is not the
next task ahead of active OPT acceptance. The old attachment-symbol probe is
retired; actual `retain_tensor`/`bind_tensor` sharing remains covered by mixer,
KV and decoder checks. Its historical result is retained in the
[operator record](tests/np101-operator-acceptance.md).
