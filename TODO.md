# Remaining work

Updated: 2026-11-14. Validated behavior and performance belong in
[status](docs/np101-status.md); execution gates and incidents belong in the
[recovery ledger](docs/np101-recovery-incidents.md). This file records deliverables,
dependencies and acceptance boundaries. Task IDs are planning references, not code or artifact names.

✅ Complete within stated scope · 🟡 Partial · 🔜 Ready · 🧩 Dependency · 🌐 External · ⏸ Deferred

## Code organization

The [runtime plan](docs/inference-runtime-plan.md) records the C++ configuration,
loading, inference and supervision work. Directory migration and C++ loading/runtime extraction
are accepted for the scoped SoC regressions. Native process supervision is accepted for the same requests; the Python
adapter retains deployment and recovery policy. EP and SoC remain under
`src/devices/np101/api/`; future vendor intermediate
representation (IR) support starts independently, without a speculative common platform layer.

| Order | Status | Deliverable and acceptance |
|---|---|---|
| Directory migration | ✅ | Sources, tools, tests, imports, build and documentation paths migrated. Both targets compile; 142 Python tests, four C++ CPU tests, 18 command entries and one SoC full-model request per model passed. |
| EP hardware regression | ⏸ | The user accepted compilation-only for the 2026-11-11 migration. Reuse accepted OPT/Qwen requests when hardware becomes available; do not reopen performance exploration. |
| 1. Configuration and contracts | ✅ Offline scope | OPT named execution/artifact objects, shared Qwen preparation validation, optional JSON inputs and documented ownership/default/override rules. Native formats, artifact checks and execution policy unchanged; standalone IO extraction stays with loader/supervisor work. |
| 2. C++ loading and thin entry points | ✅ | Shared native request/report modules and typed model request loaders; thin commands retain existing arguments. Five CPU tests, 48 Python tests and both builds pass. |
| 3. Shared inference runtime | ✅ Scoped SoC acceptance | Shared scheduling/reporting and model execution sessions; graph construction separated from state/runtime. OPT 17→16 and Qwen 96→16 passed with unchanged token/state outputs and NPU pool peaks. |
| 4. Independent supervisor | ✅ Scoped SoC acceptance | Native process groups/readiness/deadlines/termination and verified companion deployment; 54 Python tests, five C++ tests and both builds pass. OPT/Qwen tokens/state and pool peaks match the prior run. Deployment, health sampling and recovery policy remain Python. |

PR review: tracked-source-only EP/SoC builds (including explicit device-check binaries), 154 Python
tests, five C++ CPU tests and 18 command help entries passed. Fixed malformed final-token parsing
and buffered output-close error reporting; no model/operator changes or new hardware runs.
Before public release, the owner must choose a repository license and confirm third-party material
redistribution rights. EP hardware regression remains deferred as above. Do not treat a targeted
credential scan as a complete security or licensing audit.

Review follow-up: EP now uses the independent native guard, and SoC launch binds the
selected deployment identity before execution. Local offline regression and both device builds
are covered. Newly packaged OPT 17→16 and Qwen 96→16 passed SoC revalidation on 11-12, including
numerics, release and kernel checks. EP hardware revalidation remains deferred.

## Recommended execution order

| Order | Work | Acceptance and boundary |
|---|---|---|
| 1 🔜 | Measure sustained and retained-history behavior when needed | Reuse the accepted full/reset collection; do not repeat it without a specific gap. The all-18 path has not inherited the original model's natural two-turn retained-history acceptance. Wider capacity and persistent reliability remain unvalidated. |
| 2 🔜 | Minimal rejection/correction protocol and measured acceptance | Start offline or reuse existing target-model calls. Use accepted draft length and end-to-end cost to decide whether RPC/hardware integration is worthwhile. |
| 3 🌐 | Target-model integration and vendor IR | Wait for interfaces. Do not prebuild a universal compiler framework or tree-serving system. |

Keep split/shared DeltaNet projections retired from the formal path: independent state, mapped
handoffs, separate commits and M1-only recurrence did not eliminate hangs. Preserve evidence and
the original model; reopening requires a discriminating hypothesis or new SDK/IR support.

The migrated SoC binaries passed Qwen 96-input/16-output and OPT 17-input/16-output requests,
including numerics, release and kernel checks. This does not validate other configurations.
Try at most three evidence-based candidates per optimization step; stop if none is useful.
Pause small head-grouping, CPU state-copy and fixed-row-copy experiments. Complete execution
comes before performance.

### Qwen prefill follow-up

| Priority | Work | Decision gate |
|---|---|---|
| Deferred | Reduce duplicated M1/M8 embedded projection weights | Preserve complete-block numerics and decode speed. Previous split sharing failed; do not repeat it unchanged. Await better SDK/IR support or a new source-bound representation. |
| Deferred | Remove measured hidden movement or redundant submissions | Explicit IO was about 0.56% of profiled prefill. Require collection-level gains; fewer Map/Unmap calls alone are insufficient. Keep fixed bindings and avoid runtime Verify. |
| External | Lower-level performance attribution | Completion waits contain outstanding execution; they do not prove an IO bottleneck. Obtain actionable counters or IR before another broad optimization sweep. |

## S6: Qwen deployment, sessions and capacity

| ID | Status | Next step and acceptance |
|---|---|---|
| S6.4.1 | 🟡 Text comparisons available | Align draft/target tokenizers, templates, end-of-sequence (EOS) behavior and effective generation policy. Similarity of four independent answers is not speculative acceptance rate. |
| S6.4.2 | ✅ Current model fits | Bounded M1 and batched-FFN runs passed with 768 MiB Linux / 2048 MiB NPU. Account separately for weights, derived copies, attention key/value (KV) cache, recurrent state and actual SDK peaks; KV is not currently dominant. |
| ↳ S6.4.2.1–2 | ✅ Weight reuse | Do not load raw weights already embedded in compiled graphs. Embedding and output head share one vocabulary matrix. Keep each application const/nonconst segment within 1 GiB; do not change `is_const` to bypass limits. |
| ↳ S6.4.2.3 | ⏸ Alternatives if needed | If deployment or performance remains inadequate, evaluate other small models, mixture-of-experts models or existing quantized checkpoints against weights, operators, precision and target-model/tokenizer compatibility. No automatic downloads or general quantization research. |
| S6.4.3–4 | ✅ Bounded model acceptance | Shared fully connected layers (FCL), mapped KV, joint DeltaNet state reads, M8/M16 feed-forward networks (FFN), normalization fusion and FFN state commit have evidence. Do not repeat local acceptance without a new gap. |
| ↳ Batched DeltaNet capacity | ✅ Bounded complete-block coverage | All 18 layers pass C128/C512, long/full/reset requests with explicit 1408 MiB const and VIRTUAL spill. Long-term stability remains unmeasured; split projections remain rejected. |
| ↳ Whole-model attribution | ✅ Main costs identified | Preserve low-overhead off/on/off evidence. FFN wrapper time includes state commit; completion waits do not establish an IO bottleneck. |
| ↳ Natural conversation | ✅ Two-turn replay/retained comparison | Latest C512 M8 combination passed 28→20 and 72→36 tokens, both EOS, in reset/reset/reset/retain order. Retained versus replay snapshots matched bitwise; the final unconsumed token was included correctly. Preserve the exact template/token prefix. Broader conversation quality is unmeasured. |
| S6.1 | 🟡 Context coverage differs by model | OPT original-layout C128/C256 boundaries, full-capacity rejection and reset passed; bounded C512 short/long passed, four-turn replay did not. Qwen C128 and C512 96→16 / 241→16 passed. On 11-12, two C512 241→16 requests reproduced tokens/reset snapshots bitwise; 512 inputs filled all KV positions and stopped correctly at capacity. C1024 remains untested. |
| S6.2 | 🟡 Bounded repetition passed | Two long requests in one initialized Qwen instance passed numerics, release and kernel checks. Longer long/short alternation and hours of persistent stability remain deferred; do not equate this acceptance with long-duration reliability. |
| S6.3 | ⏸ | Lightweight command-line service only when integration or user access needs it. |
| S6.5 | 🟡 C++ continuation accepted for two turns | Model-owned history and fail-closed exact-prefix checks are integrated; default generation still resets. Preserve attention KV plus recurrent/convolution state. A persistent CLI/protocol, longer conversations and rollback remain separate work; cross-request batching and concurrency stay deferred. |

## S7: Correction, collaboration and tree candidates

| ID | Status | Next step and acceptance |
|---|---|---|
| S7.1 / S7.1.1 | 🔜 Protocol first | Define session/step, accepted prefix, candidates, accepted count, correction and extra token after full acceptance, EOS/capacity/reset/failure semantics. Reuse existing calls; RPC is not required. |
| S7.2 | 🔜 Minimal single-branch loop | Cover zero/partial/full acceptance, EOS and capacity against clear-and-replay of the same accepted prefix. Distinguish accepted and committed positions. |
| ↳ S7.2.1 | 🔜 OPT reference | Fixed short traces for KV length, uncommitted slots, correction-token writes and reset; natural-language quality is not the acceptance criterion. |
| ↳ S7.2.2 | 🧩 Qwen rollback | Handle FP32 recurrent matrices, convolution history and attention KV separately. Compare checkpoint, recompute and replay memory/latency; truncating KV alone is insufficient. |
| S7.3.1 | 🔜 Real target-model baseline | Verify model, quantization, execution backend, concurrency, input/output lengths and throughput. Remeasure acceptance after quantization changes rather than carrying over BF16 assumptions. |
| S7.3.2 / S7.3.2.1 | 🧩 Correction loop first | Begin with four draft candidates, then eight only if useful. Measure average accepted length, zero-acceptance rate and draft/verification/communication/rollback costs; derive break-even and speedup thresholds from the matching NP101 model. |
| S7.3.3 | 🌐 Target verification ready | Compare target-only and collaborative execution under the same request/policy, including TTFT, decode and end-to-end time; include a DGX-local draft baseline. |
| S7.4.1 | 🔜 Offline design | Define token/parent/depth/score and bounded node budgets for a single-model tree; check top-k capability/cost beyond the greedy head. |
| S7.4.2 | 🧩 State rollback first | Share one set of weights and reuse prefixes path by path; budget recurrent/convolution/KV snapshots and temporary peaks. Do not assume aliasing or parallelism is supported. |
| S7.4.3–4 | 🌐 Target tree verification ready | Define tree masks, positions, accepted paths and mixed recurrent state. Compare chains and small trees at a fixed candidate budget; sequential path verification is not parallel tree verification. |

## S3–S4: Memory and lower-level interfaces

| ID | Status | Reopening or progression condition |
|---|---|---|
| S3.1 | 🟡 Accounting available; SDK storage opaque | Record application weights/copies/state and actual SYSTEM peaks for each new candidate; file size alone is insufficient. |
| S3.2.1–3 | ⏸ External OPT FFN weights | Public filter ports did not preserve dynamic-weight semantics; the MatMul workaround was correct but slow. Reopen only with a new sharing contract, IR or different-model evidence. |
| S3.3 | ✅ OPT shared output / FP16 validated | Qwen recurrent state needs separate treatment; do not simply reduce it to FP16. |
| S3.4 | 🟡 Extend context on demand | OPT C1024 remains untested. Measure state budgets and workload value first; do not assume KV dominates. |
| S4.1–2 | 🌐 Useful observation interfaces | Separate submission, execution, completion wait and internal movement. Establish counter units, clocks, coverage and perturbation. Old SoC VPD profiling heavily distorts execution; CPU perf is not an NPU pipeline view. |
| S4.3 | Ongoing | Record each hang, direct-memory-access timeout and storage fault independently. Do not repeat lockup experiments without a new hypothesis or infer root cause from a static Delay branch. |
| S4.4 / S4.4.1 | 🌐 Vendor IR | Establish IR level, import/export, compilation/execution, memory and observability contracts. First round-trip one unchanged layer, then one local edit; choose infrastructure after seeing the interface. |

## S8: Environment and recovery

| ID | Status | Boundary |
|---|---|---|
| S8.1 | ✅ Controls passed after 11-10 power cycle | Six-tensor IO and three-input decoder controls passed numerics, release, full kernel and 10 s observation. Keep old incidents; storage root cause is unresolved. |
| S8.2.1–6 | 🟡 Layout comparison complete; long-request fault unresolved | With 512 MiB Linux, long requests failed using both 2304 and 1792 MiB pools. Prioritize Qwen on 768 MiB / 2048 MiB; do not repeat comparisons that cannot distinguish hypotheses. |
| S8.3–4 | ⏸ | Full-pool/DLA coverage and internal v106 audit require a new need or matching-SDK evidence. |
| S8.5.1 | 🔜 Offline confirmation | Establish ownership, CPU/NPU reachability and ordinary-tensor allocation contracts for a second DDR region. Exclude memory-mapped IO, firmware and reserved ranges. |
| S8.5.2–4 | 🧩 Region identified and main pool stable | Leave the main pool unchanged. Test 4 MiB allocation/read/write/release, then small cross-pool computation, then 16/64 MiB real weights. Do not force allocation by filling the main pool, cross reserved holes or bypass budgets. |
| S8.5.5 | 🌐 More DDR or a new configuration | Within the visible 3 GiB window, even 512 MiB Linux plus all unknown 248 MiB tail space gives only about 2.49 GiB. Reaching 2.7 GiB requires additional evidence. No tensor spanning discontiguous regions in the first implementation. |

Reuse agreed recovery controls; never restore deleted convolution checks. Follow the ledger's
bounded isolated-SoC-hang retry rule, stopping after three consecutive failures of the same
experiment. SDHCI/EXT4/Oops or unreleased resources stop board work. Preparation and inference
budgets are separate and cannot be extended while running.

## Closed or deferred investigations

| IDs / direction | Decision |
|---|---|
| S1.1–3 | Batched convolution selection helped; FFN Verify caching is retired. Do not keep increasing convolution coverage. |
| S1.4 | Fused-graph initialization failed; wait for a new SDK/IR. |
| S2.1–4 | KV output optimized. Attention layout and selective transpose–GEMM–transpose (TTGT) gave no robust gain; do not proactively investigate historical performance drift. |
| S5.1–2 | Embedding/read organization, deployment checksums and startup metadata checks are in use. Prefetch/CPU overlap gave no robust gain. |
| S5.3 | Parallel Verify lacks thread-safety and shared-budget validation; it is not a default solution. |
| Multi-request batching, general quantization, training | Deferred; retain one endpoint/one request. Scope alternatives or IR work separately. |
