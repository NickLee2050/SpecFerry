# Remaining work

Before the next P3/selection hang reproduction, remind the user to cold-restart the
host and NP101; warm restarts have not eliminated the fault. Other explicitly authorized
diagnostics may proceed separately.
The September 28 v106 cold start also reproduced a hang in single-node SELECT;
cold restart is not evidence that the underlying problem is fixed.

Driver comparison: [NP101 1.0.6 / 1.0.5 report](docs/np101-sdk-driver-validation-report.md)
tracks the ten feedback issues P1–P10. After the September 25 rollback to
1.0.5, the convolution regression and unchanged vendor demo passed. The one-layer prefix
passed its 16 input/KV checks; Verify still took 140.917 s. Memory counters, host RSS growth,
and FP16 sampling errors matched the 1.0.6 observations.
MatMul/Add passed; the final FCL check again SIGSEGV in Verify. All test processes exited;
device access stopped until recovery. Authorized kernel-log
review found no NP hang/recovery, 15 device opens/releases, and the FCL userspace fault in
libOpenVX's getKernelType (offline instruction mapping, not a new full stack).
P5 initially skipped large allocations because of the old-driver capacity limit;
the within-limit readback comparison is now complete, as recorded below.
Full-model expansion was not run.

September 28 cold-start follow-up on v105:

- P1 convolution passed again; Verify took 349.972 ms and the process took 0.839 s.
- P3 selection passed all 7 cases / 23 checks. First GATHER Verify took 10.890 ms;
  the native process took 1.576 s. Kernel logs contained no NP hang/recovery.
- P7 isolated ADD flushed all 10 FP16 subnormals to zero; SELECT preserved all
  16 test values. Inputs remained unchanged and both processes released normally.
- P4 ran last using the same archived 3D executable as v106. SetupGraph reported
  a reshape from 16 to 32 elements, then SIGSEGV in libovxlib.so after 0.470 s.
  No Verify or RunGraph occurred; whether the shape configuration is valid remains unresolved.
- All five processes exited; kernel logs show five device opens/releases and no
  NP hang/recovery. The final 3D crash left a recovery marker. No further SDK calls followed.

September 28 cold-start follow-up on v106:

- Driver and five SDK library hashes match the earlier v106 installation. The SELECT
  executable and input bytes are identical to v105; no model or GATHER is involved.
- P7 SELECT preserved all 16 values and exited normally, but Verify took 28.982 s
  (v105: 9.228 ms). Kernel logs show one NP hang and automatic recovery. Total process
  time was 29.217 s; RunGraph took 0.253 ms.
- This supplies a one-node reproduction of P3 after a cold restart. v105 SELECT followed
  convolution; v106 SELECT was the first test, so preceding test order differed.
- SDK access stopped after reviewing the kernel log and left a recovery marker.
  P7 ADD and original P3 selection were not launched at that point.

Latest v106 P5/P7 follow-up, in the same boot under the authorized diagnostic exception:

- Archived the earlier marker; 64 MiB readback passed before larger probes.
- Each of FP16/FP32 × const/nonconst retained 2840 MiB and rejected the next 8 MiB.
  Each scan covered all 355 blocks; only 24 bytes in three ranges of block 109 differed.
  The local range starts are now 1,340,216 / 1,340,280 / 1,340,344, all 64 KiB later
  than historical observations. These are logical offsets, not identified physical addresses.
- All four 1024 MiB baselines scanned every byte. FP16 const/nonconst and FP32 const
  reproduced those 24 errors. FP32 nonconst passed twice at 1024 MiB despite failing
  at 2840 MiB. Do not infer a universal failing location across allocation configurations.
- P7 ADD uses the exact v105 executable, inputs and expected bytes. Output also matches
  v105: 10 subnormals become zero and the other six values remain correct. Verify took
  15.766 ms; process time was 0.168 s; resources released normally.
- Kernel comparison found no new NP hang/recovery or userspace crash. There are no
  remaining test processes or recovery marker. This does not resolve the earlier P3 hang.

P5 v105 comparison is complete after board power-off with host power retained; the
OS boot ID changed. Driver/SDK hashes match the earlier v105 installation and the
capacity executable is identical to the v106 baseline:

- The 64 MiB control passed. All four 1024 MiB targets retained 1016 MiB (127 blocks),
  then SDK allocation/upload rejected the next 8 MiB. No above-1-GiB probe was run.
- Every retained byte was scanned. All four combinations reproduced 24 bad bytes
  in block 109 at the same three offsets as the latest v106 scans; all other bytes matched.
  P5 therefore exists in both versions, including below 1 GiB.
- FP32 nonconst differs from the v106 1024 MiB baseline, which passed twice. Actual
  retained sizes and initial device state differ; do not claim either version always
  passes that combination. The v106 2840 MiB scan also failed.
- No NP hang/recovery was recorded during this comparison. All five test processes
  released and exited normally, with five device opens/releases; there was no
  recovery marker at the end of these P5 tests.

P5 no longer needs a driver switch merely to complete the comparison table.
P1/P2/P4/P6/P7/P8/P9/P10 also have both-version results.
Original P3 selection on cold-started v106 remains untested; the smaller SELECT case
already reproduced the hang.

Final report/code review on v105:

- P4 now has a current, single-node source test: `cache-append --rank 2|3`.
  The cache is preallocated; two indexed writes grow only its valid prefix.
  No application RESHAPE node or tensor resize is involved. The 2-D control passed;
  the 3-D case again reported the SDK's internal reshape error and SIGSEGV in SetupGraph.
  The 3-D shape contract still needs SDK confirmation. This new minimal test has
  not run on v106; the historical same-binary comparison remains separate evidence.
- Sampling inputs and seed are initialized before graph compilation. FP32 and
  FP16-to-FP32 controls passed; direct FP16 still returned invalid index 8.
- RSS logs now summarize the first-to-last copy delta in bytes and MiB. Fixed binding
  stayed stable; both rebind modes gained about 16 MiB over 128 rounds and read back correctly.
- The 3-D test ran last and exited with no remaining child processes. Its recovery
  marker is retained; no subsequent SDK test ran. No new kernel logs were read, so
  device health after this crash is unconfirmed.

## 完整模型输出与 KV 缓存错误

- [ ] 修复并验收 24 层 OPT 固定图的输出、KV 缓存和资源释放。
- 已知现象：`v106` 下曾输出全为 0 的 token ID，KV 数据与 CPU 参考不同，
  释放资源超时。`v105` 尚未复测完整图。
- 首个错误位置尚未找到，暂不能确定是模型拼图代码还是 SDK 的问题；作为仓库内部
  TODO 跟进，不列入 SDK 反馈报告。
- 先完成 F2.3 的单层 decoder/KV 检查，再做 F2.4 的输出投影、logits 和 token 选择检查；
  找到首个差异并修复后，再按 F2.7 扩到两层、24 层，并完成 F2.8 的释放检查。
- 通过条件：同一输入下，各层输出和有效 KV 数据满足 CPU 参考误差要求；完整模型能
  正确生成文本，重复请求不会混入上次缓存，程序正常退出。单层输入/KV 检查通过不算完成。
- 历史证据：[完整模型结果](.cache/runs/o4-graph-decode-20260924/summary.json)。

## Implementation sequence

| Item | Next action | Acceptance / dependency |
|---|---|---|
| F1: minimal device checks | v105 memory probes completed; final P4 single-node 3-D test subsequently crashed in SetupGraph | Recovery marker retained; restore/check device before further SDK tests. Cold-restart before the next intentional P3 reproduction |
| F2: fixed decode graph (O3/O4) | F2.1 and F2.2 passed; one-layer input and final KV prefixes agree with CPU | Continue decoder/head checks before expanding depth; slow Verify remains unresolved |
| F3: block prefill (O5) | Validate block 4, then token tails using the same cache/weights | Causal masking, prefix preservation, correct predictions and no per-token VerifyGraph |
| F4: full-model acceptance (S11) | Complete OPT prefill/decode, KV and reset acceptance | All 24 layers, checkpoint-default output, clean lifecycle; component path remains the comparison baseline |
| F5: performance (O6) | Measure only after graph correctness passes | TTFT and decode tokens/s; compare identical prompt/output lengths; exclude initialization, warmup and diagnostic readback |
| Deferred | Physical residency/backend profiling, long contexts/compression, Qwen duplicate DeltaNet weights | Separate work; do not claim these are solved by numerical agreement |

## Fixed-graph diagnosis

| Item | Status / next check |
|---|---|
| F2.1: reference contracts | Passed offline: one-layer/two-token OPT reference, cached versus fresh CPU logits/KV, file integrity, native configuration and byte counts. Full-model references cannot substitute for truncated logits. |
| F2.2: integrated input | Passed on the board: 14 input checks across two tokens, plus the existing two final KV-prefix checks. No additional SDK nodes/outputs; 612 nodes and 874 AddTensor calls as before. |
| F2.3: decoder/cache | Compare QKV, append preservation, attention and MLP at both steps; isolated-layer evidence already passes. |
| F2.4: output head | Compare projected hidden state, logits and selection against the matching truncated CPU model. |
| F2.5: first mismatch | Minimize and fix the first failing boundary from F2.2–F2.4, then recheck downstream. |
| F2.6: verification cost | Control graph construction and scale separately; distinguish long compilation from driver waits. Runs exceeding recovery limits stop further SDK access. |
| F2.7: depth expansion | After one layer passes, validate two-layer connections, then all layers and A/B/A request isolation. |
| F2.8: lifecycle | Selection and one-layer graph released and exited cleanly on September 25. Repeated initialization and larger-graph release remain pending. |

September 25 warm restart: selection passed all seven cases in 0.63 s; the one-layer
prefix passed 16 checks in 166.41 s. Verify took 157.887 s; two RunGraph calls totaled
0.346 s. Token/position/lookup checks were exact; decoder-input max error was
6.10e-5, K-prefix 9.77e-4 and V-prefix 1.22e-4. Both processes released normally,
with no new NP hang/recovery in the observed kernel logs. This does not establish
exclusive NPU execution or validate decoder outputs, logits or larger graphs.
Build, 38 Python tests, 4 CPU CTests, native fixture acceptance/rejection and
Python/C++ formatting checks passed during the preceding offline implementation.

## Unresolved SDK/driver observations

- **MEM-001 / P5 — byte corruption:** full scans of all 2,840 MiB find only three
  8-byte errors in block 109. Latest offsets shifted +64 KiB relative to historical
  records; see the comparison report. On v106, three of four 1024 MiB configurations
  also fail. All four v105 scans retained 1016 MiB and reproduced the same 24 bad bytes.
  This is a both-version fault, not evidence of corruption introduced only by v106.
  Ordinary const/nonconst payload caps remain 1 GiB each; only explicit synthetic
  capacity diagnostics may exceed them. The cap does not prevent this corruption.
- **MEM-003 — host growth:** repeated copy-graph rebinding/VerifyGraph increased RSS
  about 16 MiB over 128 rounds; fixed bindings stayed stable. A leak is unproven.
- **MEM-004 — SDK accounting:** FP16 counter delta was `2 * payload + 4,672` bytes;
  data matched and the counter returned after release. Physical doubling is unproven.
- **Driver wait:** on September 24, a tiny lookup using under 2 KiB application
  payload spent 22.37 s in VerifyGraph; the first RunGraph was interrupted by the
  30 s deadline. Kernel logs recorded two NP hangs/recoveries, including one before
  the deadline. Successful SDK return and process exit do not establish recovery.
  September 28 v106 cold start reproduced NP hang during a single SELECT Verify
  (112 bytes application payload, inputs initialized before compilation). Same executable
  on v105 passed without NP hang. The fault is not limited to GATHER or warm restarts.
- **Other retained workarounds:** biased FCL lowering crashed, so OPT uses MatMul+Add;
  direct FP16 categorical sampling was incorrect, so sampling promotes logits to FP32.
  Rank-three cache updates crashed; the experimental append uses validated 2D shapes.
  September 25 SDK 1.0.6 recheck: direct FP16 sampling again returned 8 for all 8-class
  draws; FP32 and promoted FP32 passed. The final rebuilt P2 test rejects unknown
  modes and checks actual nodes: MatMul/Add passed, FCL again SIGSEGV in Verify. Both were
  run with initialized input and the same executable, independently of the old-binary mistake.
- **F2 verification cost:** one-layer prefix Verify took 158.095 s before the restart
  and 157.887 s afterward. The latter run passed input/KV checks and released cleanly
  without kernel errors. Two layers previously exceeded the 300 s process deadline,
  with sampled CPU usage near 100%; no corresponding kernel NP hang was found.
  Do not infer that more nodes alone explain the cost, or retry two layers before
  completing the one-layer decoder/head diagnosis.

## Retained local evidence

- Latest warm restart: `.cache/runs/driver-1.0.6-hotboot2-20260925/` (numerical pass with kernel hang).
- P2/P6 recheck: `.cache/runs/driver-1.0.6-legacy-recheck-20260925/` (including P2 binary-mode correction).
- 1.0.5 rollback comparison: `.cache/runs/driver-1.0.5-20260925/` (15 native processes; FCL last).
- September 27 pre-SDK stop: `.cache/runs/v105-followup-20260927/` (same-boot recovery marker; no new board result).
- September 28 cold start: `.cache/runs/v105-coldboot-20260928/` (convolution, isolated SELECT/ADD, selection, then archived 3D cache; kernel logs included).
- September 28 v106 cold start: `.cache/runs/v106-coldboot-20260928/` (SELECT values passed after kernel hang/recovery; further tests stopped).
- P5/P7 follow-up: `.cache/runs/v106-p5-recheck-20260928/` (near-limit and 1024 MiB full scans, ADD comparison, before/after kernel logs and checked per-block summary).
- P5 v105 comparison: `.cache/runs/v105-p5-recheck-20260928/` (64 MiB control, four complete 1016 MiB scans, exact binary comparison and clean kernel/process checks).
- Report/code review: `.cache/runs/report-review-20260928/` (single-node cache append, sampling initialization, RSS summaries and offline fixture checks; final 3-D SetupGraph SIGSEGV left a recovery marker).
- Accepted P2 comparison: `.cache/runs/p8-verified-20260925/` (new build, exact control, FCL stack).
- F1 probe: `.cache/runs/f1-20260924/` (passing small tests and archived recovery marker).
- F2 probe: `.cache/runs/f2-20260924/` (lookup fix, passing layer, prefix timeout).
- Prefix reference: `.cache/runs/f2-prefix-reference-20260925/` (offline, one layer).
- Warm-restart validation: `.cache/runs/f2-hotboot-20260925/` (selection, prefix and kernel log).
- Earlier kernel fault: `.cache/runs/segmented-regression-20260924/` (kernel log and tiny lookup).
- Graph diagnosis: `.cache/runs/graph-pipeline-*-20260924/` and
  `.cache/runs/o4-graph-decode-20260924/`.
- OPT CPU baseline: `.cache/runs/opt-validation-cpu-20260921/`.
- Qwen CPU baseline: `.cache/runs/decoupling-20260921-cpu/`.

Commands are in [tests/README.md](tests/README.md). Detailed historical narratives
and retired diagnostics remain in Git before the cleanup; checkpoints and exports
remain in `.cache/models` and `.cache/np101`.
