# NP101 implementation and validation status

Updated: 2026-11-14. This page contains execution contracts, representative measurements and
conclusions that affect engineering decisions. See [Build and run](getting-started.md),
[code boundaries](development.md), [remaining work](../TODO.md) and the mandatory
[recovery ledger](np101-recovery-incidents.md).

Evidence paths below are relative to ignored `.cache/runs/`. Logs, weights and binaries are not
distributed with the source; a local path alone does not make an experiment reproducible elsewhere.
Earlier technical detail is preserved in commit `23b7530b0c66b95203feb97424cf4930aa14a71b` and local
`.cache/maintenance/pr-cleanup-20261110/`. The original incident timeline is archived without edits.

## Measurement definitions

- Actual output throughput = generated token count / generation wall time, including prompt
  processing and required token IO. A token is one unit of model text input/output.
- Time to first token (TTFT) runs from request start to first output. Post-first-token decode rate
  uses the remaining output count and its corresponding elapsed time.
- Preparation, SDK Setup/Verify, reset, coarse diagnostic snapshots and logging are reported
  separately; they are not operator execution or generation improvements.
- `M` is the number of token rows processed by one matrix operation; `C` is context capacity.
  M16 processes 16 known inputs, not 16 simultaneously generated unknown tokens.
- Completion waits include outstanding computation. A high ioctl share does not establish an IO
  bottleneck. Measure tracing perturbation with the same workload off/on/off. Identify historical,
  different-request and different-boot comparisons explicitly.

Terms: NBG means Network Binary Graph, compiled offline by vendor tools; FCL means fully connected
layer; FFN means feed-forward network; KV is the attention key/value cache; QKV are query/key/value
projections; LN is layer normalization. EP runs host control through PCIe; SoC runs control on the
board CPU. SDK refers to the vendor software development kit.

## Offline integration checks

The PR follow-up shares the existing native process guard with EP and verifies the selected SoC
manifest and runner files before launch. Blocking health sampling cannot extend the EP child
execution deadline. New SoC packages include the shared supervisor adapter in their deployment
inventory and require repackaging/redeployment; frozen packages remain unchanged. Offline validation passed:
160 Python tests (159 in the full suite plus the added standalone import check), three SDK-free
C++ tests, five EP CPU tests, EP/SoC compilation and formatting checks. The 11-12 SoC deployment
acceptance below covers the new adapter/identity contract; EP remains compile-only.

The SDK-free `host` build and local checks are described in the [test guide](../tests/README.md).

## Current delivery acceptance

The 11-12 freshly packaged OPT/Qwen entries passed one SoC whole-model request per model, reusing
the original NBGs, requests and CPU references. Both passed original numerical/state checks,
resource release, complete kernel checks and 10 s observation with no new fault. Historical markers
remain retained. This extends the earlier C++ runtime/supervision acceptance to the updated
deployment identity and standalone supervisor adapter.

| Configuration, input→output tokens | Preparation | TTFT | Actual output | Post-first-token decode | Sampled SYSTEM peak |
|---|---:|---:|---:|---:|---:|
| Qwen C128, M8 FFN + attention normalization fusion + FFN state commit, 96→16 | 147.327 s | 69.813 s | 0.185003 token/s | 0.899712 token/s | 1604.836 MiB |
| OPT C256, four uncached convolution layers in M16 prefill, 17→16 | 148.895 s | 1.720 s | 2.954269 token/s | 4.058464 token/s | 1447.515 MiB |

Relative to the preceding accepted supervisor run, actual output changed +0.36%/−0.35%; sampled
SYSTEM peaks were identical. Minimum available Linux memory was 291.797/445.574 MiB. These are
single regression samples, not speedup claims. Native rates agree with independent Python
aggregation. Evidence: `qwen-release-acceptance-20261112/acceptance.json`.

The independent C++ guard owns process groups, readiness, fixed deadlines and termination. The
Python policy adapter still owns deployment/SDK identity, health sampling and recovery/release
acceptance. Tests cover blocked policy sampling/publication, deadline phases, failed exec, stopped
children, descendants and lost control channels; no new tracing was inserted into model execution.
Evidence: `native-supervisor-20261111/acceptance.json`; prior comparison and runtime extraction:
`runtime-contracts-20261111/acceptance.json`. Offline evidence lives under
`.cache/maintenance/native-supervisor-20261111/` and `runtime-extraction-20261111/`.

The subsequent PR review fixed rejection of an overflowing final prompt token and propagation of
buffered binary-output close failures. Valid model requests, operators and tensor/state layouts are
unchanged; error-path fixes do not imply an inference speedup. Clean
tracked-source EP/SoC builds, 154 Python tests, five native CPU tests and 18 command help entries
passed; evidence is in `.cache/maintenance/pr-readiness-20261111/`. The checks use the installed
Conda environment and explicitly supplied SDK/toolchain, not a newly provisioned machine.

The earlier directory migration passed 142 Python tests, four C++ CPU tests, 18 tool entries and
both builds including explicit device-check binaries. The subsequent clean Ninja rebuild retained
compiler/SDK/flags and produced byte-identical tested SoC binaries; see
`ninja-switch-20261111/acceptance.json`. Cache-hit build times do not establish clean-build speedup.
EP acceptance remains compilation-only by user instruction; hardware regression is pending.

## Closed EP exploration

Accepted EP configurations remain frozen. Optional exploration reopens only for a new requirement
or evidence; historical implementations are regression/diagnostic references, not automatic retries.

| Model/path | Configuration | Accepted scope and performance |
|---|---|---|
| OPT-350m recommended NBG combination | NBG QKV/output projections, fused M1 FFN/final LN and one head; optional M16 prefill; hand-written attention; mapped token readback | C64 and bounded C256. C64 17→16: M1/M16 actual 1.21641/1.95893 token/s, decode 2.35418/2.34884 token/s. |
| OPT historical fastest combination | FCL attention projections, fused NBG FFN/final LN and NBG head; direct FCL validation subsequently stopped | Historical C64/C128 scope. C64 maximum actual 1.96582 token/s and decode 3.63403 token/s; do not combine with current-period results. |
| Cached Qwen3.5-0.8B text model | GATHER_ND, zero-based layer 9 MatMul FFN, 18 DeltaNet + 23 FFN NBGs, mapped readback; fastest explicit combination adds full shared FCL head, broadcast outer product, fused input RMSNorm and joint state commit | C64, two 23→16 requests: actual 0.287186 token/s, decode 0.610743 token/s. Conservative defaults remain available. |

Different models, lengths and request counts are not cross-model speed comparisons. Preparation,
reset and coarse snapshots are excluded from generation time. Packages bind graphs, source weights,
ports and five library identities; keep original contents and do not silently re-export them.

| Accepted local package | manifest.json SHA256 |
|---|---|
| `opt-entry-integration-20261014/ffn-head-package` | `ec99e136f8458c301ed4165a582ade84751c468b596fcd3ef5431d0d96cf64fd` |
| `opt-fast-prefill-20261016/nbg-projections-prefill-package` | `910f4918102fc6569280e126aac3dc4d93d7f14320fb9fc409dade8e5438c450` |
| `opt-qwen-integration-20261014/qwen-compiled-blocks-package` | `9ccdf15e4c8e783b5e875af8ae03960a918563326a9529c2a57e2cd0d2e7f0be` |

These packages bind v105 `srcversion=F53A3E6EFBB7D0FD7D4617C`. Checkpoints, NBGs and fixed accepted
binaries remain ignored; reject manifest/source mismatches. Qwen's conservative Python/C++ defaults
remain; shared FCL head and recurrent optimizations are explicit. OPT retains its component fallback.
Packages with complete QKV/output artifacts default to NBG; old head-only packages retain FCL.
CPU contracts cover this default selection; hardware evidence used the same explicit NBG settings.
The fastest EP Qwen also uses `qwen-deltanet-optimization-20261015/outer-normalized-package`, manifest
`4c9d1100efd57e751318f0e74a5dee53bd01fca6f678db058ec6b21abc6bbd35`.

## SoC deployment and memory

Current Qwen configuration reserves 768 MiB for Linux and 2048 MiB for SYSTEM. Linux System RAM is
`0x40200000–0x6fffffff`; driver `contiguousBase=0x70000000`, `contiguousSize=0x80000000`, ending at
`0xefffffff`. Actual Linux MemTotal is about 672.4 MiB, not the full configured window.
`/np101_driver/go.sh` is loaded explicitly, not automatically; reboot does not imply it ran.

| Pool/limit | Meaning |
|---|---|
| SYSTEM | Reserved contiguous NPU pool containing application tensors and SDK allocations. |
| VIRTUAL | Linux-backed memory; do not add it again to reserved SYSTEM capacity. |
| EXTERNAL / EXCLUSIVE | No validated configuration for the current model; names do not imply const/nonconst. |
| Application const/nonconst | Each capped at 1 GiB, counting shared backing once. Both consume physical pools; they are not separate physical regions. |
| Reserved/unregistered addresses | Require ownership evidence before allocation. Logical corruption offsets are not physical addresses. |

New Qwen bundles match this layout. OPT packaging still pins the historical 512 MiB Linux /
2304 MiB pool by default and cannot be deployed unchanged onto the Qwen layout. Keep accepted
bundles immutable; do not edit a manifest to pretend compatibility. C512 long requests failed with
512 MiB Linux and both 2304/1792 MiB pools; neither Linux capacity nor address layout is proven causal.

Payload hashes are checked at deployment. Launch checks receipt, metadata, ports, sources, budgets
and SDK identity. Preparation ≤240 s; OPT inference ≤300 s; Qwen defaults to 300 s and can explicitly
select 600 s before launch. Never extend a live deadline. SoC exit observation is 10 s; reread the
complete kernel before the next run. Successful output does not replace release/health checks.
Only isolated SoC NP hangs qualify for bounded retries, stopping after three consecutive failures
of the same experiment. Storage faults and Oops do not qualify.

## Maintained SoC model paths

| Path | Implementation and boundary |
|---|---|
| Conservative OPT | FP16 KV, no Verify cache; first four M16-prefill FFNs use locally compiled convolution, the rest shared FCL/compiled decoders. C256 long/short/long and original-layout bounded C512 passed historically, not current-layout long-request or persistent stability. |
| Qwen M1 | 18 DeltaNet NBGs, six hand-written attention layers, fixed-weight FFNs; shared FCL QKV/output and layer 10 FFN, full shared vocabulary head; MatMul fallback retained. |
| Batched Qwen FFN | One M8 or M16 NBG per each of 24 layers, zero-padded reuse during decode; optional mapped KV append, attention input-normalization fusion and FFN state commit; no extra M1 weight copy. |
| Two-layer M8 DeltaNet candidate | Shares M1 next-state output buffers; one request validated, not formal-entry or two-turn/reset acceptance. |

For Qwen C128 96→16, the earlier shared-FCL path had TTFT 100.293 s and actual 0.1328 token/s.
The matched mapped-KV comparison improved TTFT 100.201→90.281 s and actual 0.132680→0.146514 token/s,
with zero runtime KV reverifications.

| Qwen configuration | Request | TTFT | Actual output | Comparison scope |
|---|---|---:|---:|---|
| Joint DeltaNet state reads, M1 FFN | 96→16 | 82.749 s | 0.159380 token/s | Historical mapped-KV baseline, not same-boot A/B. |
| M8 FFN | 96→16 | 77.413 s | 0.168200 token/s | One weight set; decode essentially unchanged. |
| M16 FFN | 96→16 | 76.565 s | 0.169778 token/s | About 0.9% over M8, not a robust advantage. |
| M8 + attention normalization fusion | 96→16 | 71.993 s | 0.180081 token/s | Same-boot baseline 77.557 s / 0.167861 token/s. |
| Add FFN state commit | 96→16 | 70.069 s | 0.184747 token/s | Same-boot prior commit 73.081 s / 0.177784 token/s. |
| Above M8 combination, two clear-and-replay turns | 28→20, 66→49 | Mean 34.661 s | Aggregate 0.480163 token/s | 69 outputs / 143.701 s; both EOS; tails/reset covered. |
| Add two M8 DeltaNet layers, experimental | 96→16 | 66.340 s | 0.192707 token/s | Decode 0.898862 token/s; model multi-turn untested. |

C512 reuses all 24 M8 FFNs, attention normalization fusion and FFN state commit, without recompiling
NBGs or duplicating weights. Both same-boot bounded requests passed original tolerances, unwritten-KV
zero checks, release, full kernel and 10 s observation:

| Capacity, input→output | Preparation | TTFT | Actual output | Decode | SYSTEM peak |
|---|---:|---:|---:|---:|---:|
| C512, 96→16 | 148.127 s | 70.261 s | 0.184288 token/s | 0.905794 token/s | 1612.617 MiB |
| C512, 241→16 | 146.059 s | 175.641 s | 0.083262 token/s | 0.907773 token/s | 1612.617 MiB |

Compared with the 11-10 C128 96→16 result (69.745 s / 0.185252 token/s), TTFT increased about 0.74%
and actual output fell about 0.52%; a single sample does not establish regression. SYSTEM peak rose
7.781 MiB; KV itself grew from 1.5 to 6 MiB. Long-request minimum Linux availability was 282.012 MiB.
Increasing input from 96 to 241 tokens made TTFT about 2.50× longer, with nearly unchanged decode.
Token-by-token prefill dominates long requests. The 241→16 run consumes 256 positions, not full C512
capacity. Evidence: `qwen-context-512-20261110/comparison.json`; the long CPU reference ran alongside
the short board test.

On 11-12 the updated deployment passed two 241→16 requests in one process at C512. Mean TTFT
was 176.494 s; 32 outputs over 386.315 s gave 0.082834 token/s, with 0.900147 token/s post-first-token
decode. Both generations and all collected reset snapshots matched bitwise. A separate 512-input
request filled every KV position, returned the matching first token and stopped with `capacity`
before another consume; TTFT was 373.757 s. Hidden/recurrent state and full written KV matched
the existing CPU tolerances. This boundary probe is not an ordinary output-throughput benchmark.
Preparation took 145.653/146.470 s, sampled SYSTEM peaks were 1612.620/1612.617 MiB, and Linux
availability stayed above 291 MiB. Both released pools/owners/GFP and passed complete kernel and
10 s checks. This is bounded repetition and full-capacity coverage, not hours of stability testing.
Evidence: `qwen-release-acceptance-20261112/acceptance.json`.

In the earlier 11-09 two-turn test, output matched CPU but the second answer omitted the rice
question: numerical acceptance is not answer quality or speculative acceptance. Original pipeline
tolerances remain; strict FP32
state differences in M4/M8 DeltaNet are recorded separately. M8 model SYSTEM peak is about
1604.836 MiB; the two-layer DeltaNet candidate adds about 55.8 MiB during execution. Expanding to all
18 layers was initially estimated at 1347.632 MiB const and about 2043 MiB physical runtime use.
The later authorized full-coverage trial below passed with a temporary const-policy exception,
but only 6.265 MiB physical pool headroom. Shared state output is not shared weights.

**Graph-boundary lesson:** adding state-copy source/destination as external ports of the FFN wrapper
produced Inf hidden values and zero token selection. Keeping the original two inputs/one output with
fixed internal copy bindings passed. Ordering, lifetimes and port permutations are execution contracts,
not cleanup details. Evidence: `qwen-ffn-state-diagnosis-20261110/`,
`qwen-block-progress-20261110/`, `qwen-shared-state-output-20261110/`.

### Retained conversation state

The 11-12 C512 diagnostic used the formal M8 FFN combination and one model instance for
reset/replay turn 1, reset/replay turn 2, reset turn 1, then retained turn 2. Both turns matched the
CPU tokens and ended at EOS: 28→20 and 72→36. Each two-turn sequence generated 56 tokens.

| Mode | Mean TTFT | Total generation time | Actual output | Decode |
|---|---:|---:|---:|---:|
| Clear and replay | 36.320 s | 132.501 s | 0.422639 token/s | 0.902105 token/s |
| Retain exact history | 19.640 s | 98.943 s | 0.565982 token/s | 0.905067 token/s |

Second-turn TTFT fell 52.021→18.495 s (64.45%); aggregate output throughput improved 33.92%.
The gain avoids repeated prompt processing, not faster decode kernels. Preparation was 146.116 s,
SYSTEM peak 1612.617 MiB, minimum Linux availability 291.387 MiB, application const/nonconst peaks
930.435/623.907 MiB. No runtime KV Verify occurred. Collected layer-23 hidden/KV and layer-22
recurrent snapshots matched CPU tolerances; retained versus replay snapshots matched bitwise.
Release, full kernel and 10 s checks passed. This validates two turns, not arbitrary conversations
or speculative rollback. No new hang or storage fault occurred.

`Model::generate(..., HistoryMode::Retain)` reuses model-owned KV, recurrent and convolution state
and requires the unchanged full token prefix. The next prompt includes the previous final output,
which has not yet been consumed. Here 47 positions were committed after turn 1, so turn 2 processed
25 additional tokens rather than replaying all 72. Preserve raw generated/template tokens; rendering
the text history again can omit the empty non-thinking template prefix and invalidate reuse.
Default generation still resets; the persistent CLI/protocol remains deferred.

Attention reset is a **logical length reset**, not physical zeroing. The initial diagnostic oracle
incorrectly required a zero stale suffix after the long→short reset. Saved evidence instead proves
the written 47-position prefix matches the first turn, positions 47–106 retain exactly the previous
request's masked bytes, and the never-written suffix is zero. No tolerance or model code was changed
to resolve this oracle mismatch, and no board rerun was needed. The earlier missing-oracle-file
packaging failure and initial comparison are preserved in the recovery ledger and evidence.
Evidence: `qwen-release-acceptance-20261112/{acceptance.json,conversation-complete-test/}`; the
diagnostic entry changes request sequencing only and is retained under ignored evidence paths.

## Current bottlenecks and candidate results

On 11-10, the two-layer M8 DeltaNet + all-layer M8 FFN candidate was profiled on 96→16 using
in-memory timing off/on/off, excluding preparation/snapshots/logging. Off runs took 83.028/83.262 s;
on took 83.309 s, +0.198% over their mean, below the off-run span of 0.282%. Tokens and four snapshots
matched bitwise; execution/release/kernel/10 s checks passed. This measures SDK calls and explicit IO,
not device pipeline counters; RunGraph time cannot be assigned wholly to movement or waiting.

| Segment | Before first token: 66.465 s | Following 15 steps: 16.844 s |
|---|---:|---:|
| Remaining 16 token-by-token DeltaNet layers | 33.434 s (50.30%, 1536 calls) | All 18 layers: 6.023 s (35.76%, 270 calls) |
| Independent state copy | 14.873 s (22.38%, 1344 calls) | Included in FFN wrapper |
| Two M8 DeltaNet layers | 2.630 s (3.96%, 24 calls) | Not used |
| Attention input projections / rotary positions | 5.478 s (8.24%) | 0.910 s (5.40%) |
| Attention reads / output projection | 5.461 s (8.22%) | 0.837 s (4.97%) |
| FFN wrapper including state commit | 2.795 s (4.21%) | 3.760 s (22.32%) |
| Language-model output head | 0.320 s (0.48%) | 4.811 s (28.56%) |
| Explicit decoder IO + mapped KV | 0.370 s (0.56%) | 0.163 s (0.97%) |

The remainder includes embedding/final normalization; uncovered time is 0.155/0.037 s. FFN wrapper
time is not pure matrix multiplication. Each independent commit copies a logical 1.047 MiB in an
average 11.07 ms, about 1407 MiB cumulatively; actual internal traffic is unknown. Token-by-token
DeltaNet and independent commits account for 72.68% before the first token. Eliminating only commits
has a mathematical TTFT ceiling of 22.38%, not a several-fold end-to-end gain.

### Memory attribution

Initialization ownership snapshots show 976.283 MiB application const + 611.059 MiB nonconst =
1587.342 MiB resident. Weight/NBG accounting is about 1545.288 MiB:

| Component | Accounted size |
|---|---:|
| Shared embedding/head | 485 MiB |
| Original attention weights | 84.006 MiB |
| 18 M1 DeltaNet NBGs | 362.921 MiB |
| 24 M8 FFN NBGs | 566.753 MiB |
| Two extra M8 DeltaNet NBGs | 46.419 MiB |
| Other constants | 0.191 MiB |
| Double-buffered recurrent state | 36 MiB |
| Double-buffered convolution history | 1.688 MiB |
| C128 attention KV | 1.5 MiB |
| Other ordinary tensors | 2.865 MiB |

NBG accounting conservatively covers files/source weights, not bytewise internal allocations. SYSTEM
sampling constrains extra SDK workspace/copies. Duplicate DeltaNet weights, not KV, currently limit
batch coverage. Shared head does not allocate another 485 MiB vocabulary matrix.

### Shared projections and decode candidates

These serial tests followed recovery controls. Final accepted runs released resources and passed
full kernel/10 s checks. Keep the initial two Verify rejections and trailing head-control preparation
timeout as failures, even though revisions later passed. Formal defaults are unchanged.

| Single-layer candidate | Prefill collection | M1 continuation collection | SYSTEM resident | Decision |
|---|---:|---:|---:|---|
| Shared input/output projections; separate M1/M8 recurrence | 1.497→1.662 s | 0.524→0.712 s | 51.701→34.273 MiB | Saves memory, slower. |
| Also fuse output projection and state commit | 1.496→1.579 s | 0.523→0.594 s | 51.701→34.277 MiB | Recovers some submission cost; not integrated. |
| CPU mapped state copy | 3.136→4.399 s | 0.531→0.754 s | Both 51.576 MiB | Rejected. |

Collections use 96 prefill rows followed by 16 M1 steps. Projection baselines use M8 prefill; the
CPU-copy baseline uses M1. Do not mix baselines or call these output throughput. All passed original
pipeline tolerance; shared projections had two strict-state outliers after M1, maximum 0.0004578,
without relaxed thresholds. CPU copy also passed strict tolerance. Shared constants fell
43.355→23.392 MiB per layer; all-18-layer const of 987.993 MiB is an untested extrapolation.
Projection/commit fusion keeps original NBG ports and internal DATACONVERT bindings. Its M1 is
about 16.5% faster than separate submission, but 13.7% slower than original M1; this does not prove
hardware parallelism.

### Complete-block Qwen prefill

The standard JSON/C++ entry offers complete eight-token (M8) DeltaNet graphs for all 18
linear-attention layers; original M1 graphs remain resident for decode/tails and as the default
fallback. All 24 FFNs use M8. Shared FCL projections/head, mapped KV, fused normalization,
shared residual blocks and FFN state commits remain enabled; unnecessary intermediate outputs
are not republished. M1/M8 sequentially write shared next-state tensors, distinct from their
current-state inputs. FFN commits once per block; this is not in-place or parallel recurrence.

The complete graphs embed projection weights again. Their explicit SoC profile permits
1408 MiB const / 1024 MiB nonconst; other configurations retain separate 1024 MiB caps.
Neither const flags nor driver pools change. The original tokenwise model remains available.

EP/SoC builds and offline contracts pass. The 11-14 formal process passed two C512 241→16
requests: CPU acceptance, bitwise diagnostic/state and reset comparisons, release and complete
kernel/10 s observation. No runtime KV revalidation occurred. Historical fault markers remain
under the [explicit soft-restart exception](np101-recovery-incidents.md#11-14-explicit-soft-restart-diagnostic-authorization).

| Accepted complete-block workload | Preparation | Prefill | TTFT | Generation wall time | Actual output | Post-first-token decode |
|---|---:|---:|---:|---:|---:|---:|
| C128, 96→16 | 163.504 s | 39.274 s | 39.610 s | 56.326 s | 0.284059 token/s | 0.897342 token/s |
| C512, 96→16 | 164.716 s | 39.022 s | 39.355 s | 56.134 s | 0.285030 token/s | 0.893927 token/s |
| C512, 241→16 | 163.931 s | 98.611 s | 98.948 s | 115.656 s | 0.138342 token/s | 0.897772 token/s |
| Formal C512, 2 × 241→16, mean | 165.098 s, once | 98.468 s | 98.800 s | 115.530 s | 0.138492 token/s | 0.896593 token/s |
| C512, 496→16 | 164.296 s, shared process | 202.654 s | 202.982 s | 219.906 s | 0.072758 token/s | 0.886317 token/s |
| ↳ reset, 512→1 capacity stop | same process | 209.008 s | 209.328 s | 209.328 s | 0.004777 token/s | Not applicable |
| ↳ reset, 96→16 | same process | 39.294 s | 39.626 s | 56.454 s | 0.283417 token/s | 0.891378 token/s |

Each workload was accepted once. The formal row aggregates 32 outputs over two generations;
preparation/snapshots are excluded. The full post-ready phase including release was 248.549 s
within 300 s. Application const/nonconst peaks were 1348.403/623.548 MiB. Sampled SYSTEM/VIRTUAL
peaks were 2026.689/23.008 MiB (combined 2049.697 MiB); driver SYSTEM high-water was 2030.756 MiB.
Linux availability stayed above 251.547 MiB. Aggregate timing reproduced the diagnostic within 0.2%.

The last three rows form a separate resident boundary/reset collection: 503.561 s post-ready
within an explicitly selected 600 s limit. Earlier tokenwise DeltaNet 241→16 measured TTFT
176.494 s, output 0.082834 token/s and decode 0.900147 token/s. Complete-block prefill reduced
TTFT by about 44%, increased actual output by 67%, and left decode essentially unchanged.
This is a historical comparison, not an interleaved trial or long-duration reliability claim.

**Reset contract:** reset changes visible length, not physical KV contents. Compare the new
valid prefix to the CPU reference, preserved masked slots to their previous snapshot and
never-written slots to zero. The original boundary oracle wrongly required physical clearing;
corrected acceptance of saved evidence required no rerun or tolerance change. Retained-history
conversation and C1024 remain unvalidated for this path.

Evidence under `.cache/runs/`: `qwen-full-block-coverage-20261113/const-trial/`,
`qwen-full-block-context-512-20261113/`, `qwen-full-block-long-20261113/`,
`qwen-full-block-boundary-20261113/` (including `reset-review.json`),
`qwen-release-acceptance-20261112/context-repeat-test/`, `qwen-block-prefill-formal-20261113/`
and `qwen-block-prefill-soft-restart-20261114/` (`comparison.json`, `native-run/`).

### Split-projection findings and limits

Sharing projection graphs between M1/M8 recurrence saves duplicated embedded weights but did not
provide a reliable whole-model path. The single-layer 96-row + 16-step collection improved
3.644→2.453 s (32.7%); prefill improved 46.0% while continuation regressed 46.4%. Existing pipeline
tolerances passed with recorded stricter-state outliers. Two/four/eight-layer model trials passed
once, but the identical eight-layer package subsequently failed; coverage is not a safe threshold.
All 18 layers initialized at 988.490 MiB const / 646.960 MiB nonconst and 1811.321 MiB SYSTEM,
then failed inference. Do not report failed elapsed time as throughput.

Private state/device copies, mapped handoffs, separate FFN commits and M1-only recurrence with
or without resident M8 all still hung. The last failed after accepted cold-boot recovery, while
complete-M8 graphs passed on the same boot. This rejects split projections as a reliable path,
without proving an internal SDK cause. RunGraph is blocking and static audit found no established
early free or missing wait. Some recurrence calls returned after about 24 s before a later hang;
logging changes did not consistently repair it. Do not add arbitrary sleeps/fences or restore
these variants without a discriminating hypothesis or new SDK/IR support.

The [recovery ledger](np101-recovery-incidents.md#split-projection-investigation-11-12-13) retains
each fault, exact SDK/kernel timeline, identities, evidence paths and retry boundaries.

### FFN marginal cost and head grouping

The same collection was repeated 32 times, with uploads/readback/logging outside graph timing.
Hidden output and final committed states matched bitwise. FFN-only omits state update and is diagnostic.

| Path | 384 prefill calls | 512 decode calls | SYSTEM resident |
|---|---:|---:|---:|
| FFN with internal commit, surrounding-control mean | 3.747 s | 5.068 s | 28.514 MiB |
| FFN only | 3.721 s | 4.924 s | 26.418 MiB |
| FFN + independent commit | 7.590 s | 10.603 s | 26.418 MiB |

Removing internal commit changed decode 2.85%, below the 4.26% control span. An isolated copy's
roughly 10 ms cannot simply be added to fused execution. Focus on still-independent prefill commits.

Grouping eight→16 vocabulary blocks per head graph, without weight copies, reduced a 16-selection
collection 5.103→4.787 s with bitwise logits. The following same-boot 96→16 model pair passed original
numerical/state/KV checks:

| Metric | Eight blocks/graph | 16 blocks/graph |
|---|---:|---:|
| TTFT / generation | 66.948 / 83.768 s | 66.582 / 82.882 s |
| Actual output / decode | 0.1910 / 0.8918 token/s | 0.1930 / 0.9202 token/s |
| Preparation | 149.744 s | 213.177 s |
| Whole-run / ready-state SYSTEM peak | 1604.836 / 1596.727 MiB | 1652.623 / 1596.719 MiB |

One pair gives about 1.07% actual-output improvement, not a robust gain, at about 63 s extra preparation
and 48 MiB extra initialization peak; not adopted. Evidence:
`qwen-shared-projection-resume-20261110/{comparison,ffn-cost-comparison,head-model-comparison}.json`
and archived `diagnostic-sources.tar.gz`; attribution/initial offline artifacts:
`qwen-performance-capacity-20261110/`. Temporary tools are retired, their measurement sources retained.

## Retired paths and interface limits

| Path | Conclusion / reopening condition |
|---|---|
| FFN Verify cache | Saves preparation only; alternating long/short requests and historical-success reruns were unstable. Retired; old bundles require original runner/binary. |
| Shared residual blocks / omitted intermediate publication | About 0.17% model change, no useful gain; SDHCI during release prevented health acceptance. Removed from source/CLI. |
| Transpose–GEMM–transpose (TTGT) | Shape/path dependent. Qwen hand-written matrix multiplication was much slower than shared FCL; the compiler can eliminate equivalent transposes. No broad shape sweep. |
| Dynamic external FFN weights | Public Conv/FCL filter ports do not establish dynamic-weight semantics. NaN occurred; MatMul workaround correct but slow. Wait for a new contract/IR. |
| CPU copy, fixed-row copy, more small graphs | No useful gain; some numerical differences. Evidence retained, no formal experimental modes. |
| Parallel Verify | Thread safety unvalidated; v105 Contexts share a process shader-slot table, so separate Contexts do not isolate its 1024-slot budget. |
| Private Verify instrumentation | Tied to x86 library offsets; retired. Public-API timing and independent numerical/lifecycle acceptance remain. |
| Vendor profiler / VPD output | Collection works on SoC, but old full-model overhead was approximately 127–148×. Pipeline timing and counter units/meaning remain untrustworthy. |

Public/internal NBG port permutations must agree; ONNX order alone is insufficient. MatMul and FCL
may select different hardware paths; mathematical equivalence does not imply equal speed or precision,
and EP/SoC optimal combinations differ. Flush/completion waits include execution; static disassembly
shows boundaries, not root causes. Preserve FP32 recurrent state. FP16 ADD flushes subnormal values,
so arithmetic is not a bit-copy substitute. Vendor IR is not ready; no preemptive MLIR, custom
instruction set or kernel framework.

## Evidence index and remaining boundaries

| Topic | Evidence directory |
|---|---|
| EP completion and throughput definitions | `ep-baseline-finalization-20261015/`, `output-throughput-review-20261009/` |
| OPT NBG/M16 | `opt-fast-prefill-20261016/`, `opt-entry-integration-20261014/` |
| Qwen joint state reads and conversation | `qwen-delta-joint-model-20261109/`, `qwen-prefill-resume-20261109/` |
| All-layer batched FFN | `qwen-ffn-block-model-20261109/` |
| Graph boundaries and state commit | `qwen-ffn-state-commit-20261110/`, `qwen-ffn-state-diagnosis-20261110/` |
| Batched state sharing | `qwen-block-progress-20261110/`, `qwen-shared-state-output-20261110/` |
| Dominant costs and memory | `qwen-performance-capacity-20261110/` |
| Projection/copy/head/FFN comparisons | `qwen-shared-projection-resume-20261110/` |

Remaining work includes retained-KV conversation acceptance for the complete-block Qwen path,
rejection/correction, real draft–target acceptance rates and long-term reliability. Bounded C512
full-capacity/reset acceptance is recorded above; it does not establish sustained reliability.
EP's accepted exploration remains closed; [TODO](../TODO.md) owns the implementation order.
