# NP101 SDK and driver findings

This page records failed contracts, interpretation limits and evidence. See
[status](np101-status.md) for accepted models/performance, the [recovery ledger](np101-recovery-incidents.md)
for execution gates and [tests](../tests/README.md) for current commands. These are historical
observations, not an automatic reproduction checklist. A model passing does not close a driver issue.

`v105/v106` identify matching driver/SDK packages 1.0.5/1.0.6, not printed module version strings.
Historical libraries are under `/usr/lib/ljmicro`; galcore source versions are
`F53A3E6EFBB7D0FD7D4617C` / `8D46803809F15FA0FD98A92`. Evidence fixes boot, command, binary/library
hashes and kernel window. Retired convolution and biased-FCL controls must not be rebuilt merely
because their historical parameters appear here.

Terms: FCL is a fully connected layer; NBG is an offline-compiled Network Binary Graph;
EDMA is the logged direct-memory-access path; SG is scatter/gather; RSS is host resident memory.
Application logical offsets are not device physical addresses.

## Construction, verification and numerical failures

| Operation | Expected | Observed | Evidence alias / historical case |
|---|---|---|---|
| Convolution → ReLU → MaxPool Verify | Vendor one-pass error ≤0.1; rewritten control had separate two-pass evidence | v106 SIGFPE in `libNNArchPerf.so::NNTransposeCycleCount_V9` integer division before convolution execution. v105 vendor/repository versions passed historically; repository Verify 42.478 ms (cold boot 349.972 ms), maximum error 0.003234/0.003114. | Verification comparison, v105 regression: `convolution-control/`; v105 cold boot: `control-conv/`. |
| Biased FCL Verify | Input `[8,1]`, identity `[8,8]`, bias 0.25 → 1.25/−0.25 | Both versions SIGSEGV; equivalent MatMul+Add passed twice exactly. v105 null-offset `0xb0` read; v106 stack includes `vxoGraphOptimization_getKernelType`. Bias alone and universal FCL failure are not proven. | Verification comparison, v105 regression: `fcl-verification/`, `fcl-symbolication.txt`. |
| Single SELECT Verify / historical selection GATHER | Successful Verify without hang; preserve FP16 bits | Same SELECT binary: v106 cold-boot Verify 28.982 s, then correct 16 values after NP hang/recovery; v105 9.228 ms without hang. SELECT alone reproduces the issue, so GATHER index-write timing is insufficient explanation. | v105/v106 cold boot: `fp16-selection/`; historical `fp16-preservation --operator select` (30 s). |
| Rank-three TENSORSTACKCONCAT, axis=1 | Append `[8,1,2]` to preallocated `[8,8,2]` | v105 single node: Setup reports `Cannot calculate the reshape tensor 16 to 32`, then SIGSEGV, without an application RESHAPE. Rank-three append+attention crashed on both versions; rank-two passed. v106 isolated node untested. | Report review: `cache-append-{2d,3d}/` (30 s); rank-three support needs vendor confirmation. |
| Full-byte scan after chunked writes | Every byte of every retained tensor matches | v106 4096 MiB probe limit, 2840 MiB retained/scanned: F16/F32 × const/nonconst all corrupt 24 B in three ranges of block 109. v105 1024 MiB target, 1016 MiB retained/scanned: same four combinations and locations. Not exclusive to allocations over 1 GiB. | v105/v106 memory scans: `summary.json`, `readback-blocks.jsonl`; probe retired. |
| RANDOM_MULTINOMIAL FP16 | Eight-class indices 0–7; valid frequency/seed behavior | Both versions returned 8 for all seven groups ×4096 samples, including zero logits. F32 and F16→F32 passed. Normal process exit is not numerical acceptance. | Verification comparison, v105 regression, report review; historical F16/F32/F16_TO_F32 sampling cases. |
| FP16 ADD subnormals | Adding zero preserves input bits | All ten positive/negative subnormals became `0x0000` on both versions; other six values correct. SELECT preserved 16 values but v106 triggered Verify hang. Input bytes/CPU reference were correct; flush rules need confirmation. | v105 cold boot and v106 memory scan: `fp16-add/` (30 s). |
| SDK memory counters | Explain allocation/release accounting | An 8 MiB tensor increased `gpu_memory.currentSize` by `2×8,388,608+4,672` B: 524,288→17,306,176→524,288, with correct readback. This does not prove doubled physical allocation, leakage or overflow. | v105 regression: `memory-accounting/`; memory accounting alias. |
| Fixed/rebound copy-graph RSS | Explain fixed/rebind costs and release | Across 128 iterations on both versions, fixed RSS increment was zero, same/advancing binding about 16 MiB (iteration 127 minus iteration 0); readback/release passed. Host RSS does not establish device allocation, traffic or leaks. | v105 regression and rebinding-memory aliases; retired fixed/same/advance experiment. |

A full-window 1×1 convolution FFN candidate also crashed at first Verify: fixed OpenVX ELF
`0x2db0c2` (`vxoGraphOptimization_getKernelType+0x5d2`), the same instruction as historical FCL.
The FULLY_CONNECTED_RELU branch lacked required `enable_relu` scalar parameter[7], not bias.
Its origin is unknown; the entry was retired. Evidence:
`ffn-convolution-and-parallelism-20261004/ffn-linear-conv1x1/`. Convolution form alone does not
promise a safe or fast neural-network execution path.

Integrity probes used 8 MiB blocks and scanned all retained blocks after the next allocation was
rejected, continuing after mismatches before release. Block 109 offsets were 1,340,216 / 1,340,280 /
1,340,344, eight bytes each; historical v106 positions differed by 65,536 B. Cumulative logical offsets
were about 873.278/873.216 MiB, not physical addresses. v106 FP32 nonconst passed two 1024 MiB runs
but failed at 2840 MiB, so the format is not universally safe. Historical reproduction began with a
64 MiB control. Ordinary/model allocations remain capped at 1 GiB per segment. The separately
authorized v106 4096 MiB synthetic probe is retired and did not cover hidden SDK or unallocated memory.

## NBG public and internal port ordering

The 10-15 partial Qwen head failure was traced to the project's exporter, not established SDK
arithmetic or hardware failure. Generated `vnn_network.c` contained an original-kernel branch with
16 outputs reversed and an `inference_with_nbg` branch/public graph with normal order. The saved
`network_binary.nb` used the latter; the exporter read the former. Reference token 31759 (block 7,
offset 3087) became 35855 (block 8, offset 3087), matching `15−7=8`; feedback then diverged.

The fix used existing `compiled_ports(..., saved_public_graph=True)` and repackaged port metadata.
NBG SHA256 `f7b15450bc2c0a4f17425b0e1a1a92d8ec3ff2b99eb51cfbf1d22ccea4ed249b` and the native binary
were unchanged. Sixteen outputs/coarse state passed against CPU with before/after baseline controls;
other accepted artifacts did not receive a global parser-mode change. Regression covers both generated
C branches and semantic permutation checks. Initial release/budgets/kernel/45 s checks were clean;
an explicit numerical-repair exception retained the marker until full corrected acceptance. It did
not exempt kernel faults, incomplete execution, unreleased resources, excess budgets or compile-audit
failures. Evidence: `qwen-head-prefix-20261015/port-order-diagnosis.json` and original/fixed runs.
Wrong-output timing is not accepted performance.

The same day's full shared FCL head passed C64/v105 model comparison. Embedding/head retained references
to the same ordinary vocabulary tensor, read-only by application contract, without changing const
classification. Isolated simulator checks covered 4096-row and 2560-row tail blocks. No bias/activation
fusion was used; other FCL calls still require const weights. An ONNX candidate exposing weights as NBG
inputs generated MatMul, not evidence for FCL speed. See `qwen-shared-head-20261015/summary.json`.

## Missing compiler entries and shared slots

Per-head KV and const/mutable variants produced 1143 extensions: the first 1024 operations existed,
the remaining 119 were missing. Execution tables subsequently emptied, tiling/command generation was
skipped, Verify still succeeded, and decoders 23/24 produced zeros. Disassembly found 1024 slots in a
user-shader wrapper path, NULL on exhaustion and an ignored failure code above it. The first live
full-table/NULL event was not captured: this is not proven out-of-bounds writing. The budget is shared
wrapper state, not 1024 nodes per graph or a universal operator limit; it does not explain every EDMA
or chunk-corruption event.

For fixed OpenVX SHA256 `2f7223e504b15886c1562889668a80af48b19d605461beac175d336d396ff108`,
`vxCreateContext@0x50795` obtains process-level vxGlobalData; `0x507a5` writes Context+0xa8;
`vxnneGetUserShaderExecutable@0x114410` reaches the slot table through it. Graph release did not return
slots in the observed path; release of the final Context cleared the table. Concurrent Contexts share
it. Verify has a visible graph+0x20 lock; no lock was found inside the slot routine, but outer locking
and the vendor concurrency contract are incomplete. No race is proven; parallel Verify and
cross-process graph/resource transfer remain unvalidated.

Merged KV reduced usage to 423 extensions, two graph sets 846. Rebuilding within the same Context hit
`846+178=1024`, leaving 668 missing entries; compile checks rejected execution. Releasing the final old
Context before rebuilding produced two complete 846-entry runs with matching output. This lifecycle is
used; ordinary requests reuse the model. Detailed addresses/open questions:
`decode-static-audit-20261009/verify-and-incidents.md`.

## EDMA and interpretation limits

The 10-14 static audit found that v105's scatter/gather EDMA branch deliberately sets the destination
field used in timeout printing to all ones, while submitting an SG table. Therefore
`dst=0xffffffffffffffff` alone proves neither illegal access nor DMA mapping failure. Internal waits
include 4 ms / 30 s branches at HZ=250, distinct from a historical application 900 s limit; the actual
branch and root cause remain unknown. Exact four-byte EDMA timelines/disassembly belong in the ledger,
not permission to run a faulted device. `Waiting for HDMA Transfer...` alone supplies no duration.

## Evidence index

Paths are relative to `.cache/runs/`; historical aliases are in `evidence-directory-aliases.json`.
Read result, SDK log, execution identity/exit and kernel window together. Original evidence is unchanged.

| Alias | Directory/files |
|---|---|
| Verification comparison | `operator-verification-20260925/`, `demo-verify-20260925.log`, `driver-1.0.6-legacy-recheck-20260925/` |
| v105 regression | `driver-1.0.5-20260925/`: `summary.json`, `environment.json`, `kernel-review.json` |
| v105 cold boot | `v105-coldboot-20260928/`: summary/kernel review; 23 selection checks passed without hang. |
| v106 cold boot | `v106-coldboot-20260928/`: SELECT timing, `kernel-test-window.log`, `completion.json`; stopped after first hang. |
| Report review | `report-review-20260928/`: `review.json`, isolated append, sampling and rebinding checks. |
| v106 memory scan | `v106-memory-integrity-20260928/`: four combinations, ADD and kernel checks; no new hang. |
| v105 memory scan | `v105-memory-integrity-20260928/`; earlier large-payload evidence: `capacity-map-20260923/`. |
| Memory accounting | `memory-integrated-accounting/` |
| Rebinding memory | `memory-integrated-{fixed,same,advance}/` |
| Compile-table integrity | `incremental-model-checks-20260924/`, `incremental-model-warmboot-20260925/`, `incremental-cache-checks-v105-20260928/`, `weight-storage-and-cache-reshape-preparation-20260929/shader-limit/` |

Isolated rank-three append on v106 and the original v106 cold-boot selection case remain untested;
do not automatically schedule risky reproduction. Retired structures need their original independent
archives, not a current graph relabeled as historical. EDMA, control hangs and shutdown page faults
remain open despite recovery-control success. Retired source identities:
`.cache/runs/repository-compaction-20261016/`. Historical arguments/capacities are evidence, not current
commands or retry authorization.
