# NP101 implementation and validation status

Updated: 2026-11-11. This page contains execution contracts, representative measurements and
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

## Current delivery acceptance

After the 11-11 directory migration, both rebuilt programs passed one SoC whole-model regression,
reusing the original NBGs, requests and CPU references. Numerics, application/pool release, complete
kernel checks and 10 s observation passed with no new device fault. Historical markers remain retained.

| Configuration, input→output tokens | Preparation | TTFT | Actual output | Post-first-token decode | Sampled SYSTEM peak |
|---|---:|---:|---:|---:|---:|
| Qwen C128, M8 FFN + attention normalization fusion + FFN state commit, 96→16 | 145.756 s | 69.505 s | 0.185759 token/s | 0.902094 token/s | 1604.836 MiB |
| OPT C256, four uncached convolution layers in M16 prefill, 17→16 | 147.597 s | 1.712 s | 2.956322 token/s | 4.054121 token/s | 1447.515 MiB |

Relative to the same requests on 11-10, actual output changed +0.27%/−0.47%, with identical sampled
SYSTEM peaks. Minimum available Linux memory was 291.992/446.180 MiB. These are single regression
samples, not performance improvements. Both target builds, including explicit device-check binaries,
142 Python tests, four C++ CPU tests, 18 tool entries and formatting passed. Bodies of 83 C++ files
were unchanged except includes/formatting. NBG inventories and final deployed binaries matched bytewise.
Evidence: `directory-migration-20261111/acceptance.json`; prior comparison:
`pr-cleanup-validation-20261110/acceptance.json`.

EP migration acceptance is compilation-only by user instruction; hardware regression is pending.
Afterward the entire build tree was cleaned and rebuilt with Ninja, retaining compiler, SDK, flags
and ccache. Both default builds and four CPU tests passed; SoC binaries matched the already-tested
files bytewise, so no board run was repeated. See `ninja-switch-20261111/acceptance.json`.
Cache-hit build times do not establish clean-build speedup.

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

The two-turn output matched CPU, but the second answer omitted the rice question: numerical acceptance
is not answer quality or speculative acceptance. Original pipeline tolerances remain; strict FP32
state differences in M4/M8 DeltaNet are recorded separately. M8 model SYSTEM peak is about
1604.836 MiB; the two-layer DeltaNet candidate adds about 55.8 MiB during execution. Expanding to all
18 layers is estimated at 1347.632 MiB const, exceeding the 1 GiB limit, and about 2043 MiB physical
runtime use, too tight. No full-18-layer capacity test was run. Shared state output is not shared weights.

**Graph-boundary lesson:** adding state-copy source/destination as external ports of the FFN wrapper
produced Inf hidden values and zero token selection. Keeping the original two inputs/one output with
fixed internal copy bindings passed. Ordering, lifetimes and port permutations are execution contracts,
not cleanup details. Evidence: `qwen-ffn-state-diagnosis-20261110/`,
`qwen-block-progress-20261110/`, `qwen-shared-state-output-20261110/`.

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

Remaining work includes formal batched-DeltaNet integration, retained-KV conversation/full-capacity
Qwen acceptance, rejection/correction, real draft–target acceptance rates and long-term reliability.
EP's accepted exploration remains closed; [TODO](../TODO.md) owns the implementation order.
