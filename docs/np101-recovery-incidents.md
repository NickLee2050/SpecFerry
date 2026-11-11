# NP101 recovery gates and incident ledger

Updated: 2026-11-11 (documentation only; no device check performed).
Read this page before every board run. See [status](np101-status.md) for accepted workload scopes
and [SDK findings](np101-sdk-driver-validation-report.md) for interface failures.
Historical plans, permissions and successful windows do not override current gates or prove that
the device is healthy now.

## Before every run

1. Check board boot ID, `np101-recovery-required.json`, `/dev/galcore` owners and the complete
   boot kernel log. Detect NP hang, EDMA timeout, BUG/Oops/page fault, general protection fault,
   invalid opcode and soft lockup. With stuckDump, also detect `GPU DRV CONFIG` and
   `Stop driver to keep scene`. `Waiting for HDMA Transfer...` alone is not a timeout.
   Run serially with a fixed deadline and active monitoring. Any new fault terminates that child
   and retains a recovery marker; later retry eligibility is assessed separately below.
2. SDHCI hardware-interrupt timeouts, EXT4 errors and block IO errors stop board work even if
   no literal `I/O error` appears. They are not isolated-NP-hang exceptions. A stuck process,
   unreleased owners/pools or any new non-NP-hang kernel fault also stops work immediately.
3. Keep binary/NBG/SDK identities fixed for controls. Only exact previously recorded fault
   occurrences may be acknowledged under an established recovery/retry decision; never exempt
   an entire error class. Correct output, empty owners or `recovery done` do not clear a gate.
4. The repository convolution control and its cached binaries/entry points were deleted by user
   instruction on 10-13. Do not rebuild or restore them. Do not silently substitute vendor
   convolution or frozen old OPT: both also failed after confirmed cold boots. Replacement controls
   require agreement. Historical source paths and command names below are evidence, not instructions.
5. Clearing a recovery gate requires a user-confirmed NP101 board power cycle, agreed controls,
   numerical/execution/budget/release checks and delayed kernel observation. A host cold boot or
   changed host boot ID is not required. Archive markers only after all gates pass; retain original
   logs. Control success covers only its checked workload. Authorized vendor convolution or another
   diagnostic does not automatically clear the gate.
6. EP controls that fail again after a confirmed power cycle must not be retried unchanged without
   new analysis/authorization. Before intentional selection-hang reproduction, remind the user to
   power-cycle the board; host warm reboot alone did not establish recovery. Other explicitly
   authorized diagnostics retain their exact scope.
7. Read-only `sudo journalctl` has standing user permission. Other sudo commands, or journalctl
   operations that change state, require explicit approval. Do not infer permission from passwordless
   access. Private-offset Verify instrumentation is retired; use public-API timing when needed.

## Deadlines and bounded SoC retries

| Scope | Preparation | Execution | Post-exit observation |
|---|---:|---:|---:|
| Formal SoC OPT | ≤240 s | ≤300 s for the complete post-ready workload | ≥10 s |
| Formal SoC Qwen | ≤240 s | Default 300 s; explicit preselected limit up to 600 s | ≥10 s |
| EP / legacy unphased diagnostics | Combined child lifetime ≤300 s | Included in combined limit | EP ≥45 s; SoC ≥10 s |

A private readiness pipe changes phases once; requests do not restart a deadline. Do not extend a
running deadline. Terminate on expiry and retain evidence/markers before changing the next attempt.
Deployment checksums and offline compilation are recorded separately from these native phases.
Release checks remain mandatory. The shorter SoC observation can miss the historically observed
recovery Oops about 33 s after exit: reread the complete kernel before the next run. Ten clean seconds
do not excuse an existing Oops or an unreleased pool.

Standing SoC authorization permits retry of an isolated NP hang after the failed child exits,
owners/pools/GFP allocations return, and delayed observation finds no further fault. Record exact
old fault timestamps before retrying; another confirmation or power cycle is not required for this
bounded retry. It does not clear recovery markers or apply to EP, storage faults or Oops.

Stop after **three consecutive hang failures of the same experiment, including the first**.
Renaming a program/output directory or shortening repetitions does not reset the streak. Only
complete numerical/execution/release acceptance resets the consecutive count; history and markers
remain. Record every attempt separately. “Intermittent” is an operating assumption, not a demonstrated
root cause. After the limit, investigate offline instead of retrying indefinitely.

## Required incident record

Record each event independently: board/host boot identities as applicable; source/binary/SDK identity;
input and graph structure; expected/actual behavior; last returned and outstanding SDK calls; exact
kernel/call timestamps and clock domains; exit, owners, pools and marker state; evidence paths;
established facts versus hypotheses; next-run avoidance and recovery conditions. Overlapping a call
with a fault locates an interval, not the internal cause. Failed numeric/timing samples do not become
accepted performance just because a later revision passes.

## Latest recorded state and unresolved failures

The latest documented recovery followed the 11-10 storage incident, after a user-confirmed board
power cycle to boot `1e21c890-d53a-49e5-a531-3e28415f0e6c`. On the unchanged 768 MiB Linux /
2048 MiB NPU configuration, six-tensor IO and a three-input decoder passed numerics, release,
complete kernel and 10 s observation; old markers were archived by identity. Evidence:
`qwen-shared-projection-resume-20261110/recovery/` under `.cache/runs/`.
The 11-11 OPT/Qwen migration regressions also passed their scoped checks; see status.
These are recorded windows, not a live health check or a storage-root-cause fix.

Recent events that affect subsequent work:

- **Storage timeout before candidate launch (11-10).** Boot
  `ad5d4749-c7c0-4e6b-a6e2-79c4f96636ce`; deployment of bundle
  `f172885457e63284efa6e5a23309c86eb4a635a149101bbe8244a628b868dd24` succeeded, but the next full
  kernel read found `14803.479017 mmc0: Timeout waiting for hardware cmd interrupt.` No candidate
  child, Setup, Verify or RunGraph started. At 14814.914401, owners were empty and SYSTEM 122976 B,
  VIRTUAL 0 B and GFP 5808 pages matched idle. This does not excuse storage failure. Testing stopped;
  do not attribute the event to a candidate operator. Evidence:
  `qwen-performance-capacity-20261110/{storage-incident.json,storage-kernel.log,partition-test/}`.
- **Fixed-row copy mismatch (11-10).** Same boot, layer 0 with 96 inputs followed by M1: row 0 matched,
  differences began at row 1; FFN/recurrent/convolution-history maximum differences were
  0.0004883/0.0008545/0.015625. Before/after original controls matched bitwise. Release and health
  checks passed, no device fault. Retain the numerical marker; only 0.124% faster, not adopted.
  Evidence: `qwen-block-progress-20261110/fixed-rows-test/run-1/`.
- **FFN state-commit wrapper ports (11-10).** Original 96→16 baseline passed. Candidate request
  ran 800.430887→886.906632 (release entry), exited 0 and released cleanly, but all 16 output tokens
  were zero, hidden included Inf, and state/KV failed. Single-layer testing found committed state
  bitwise correct but FFN hidden maximum difference 0.918335. Removing added external state ports
  while retaining internal DATACONVERT copies fixed it. The exact SDK cause remains unknown.
  Repaired full-model run `c6bde7c27e674d9eb7dd5487fc5a5071`, 1959.386463→2046.029959, matched
  original snapshots bitwise and passed all gates. Initial fixture also omitted aggregate `run.json`;
  the exact software-record exception retained original failures and did not waive hardware gates.
  Evidence: `qwen-ffn-state-commit-20261110/`, `qwen-ffn-state-diagnosis-20261110/`.
- **SDHCI during shared-residual release (11-10).** Boot
  `a6c83b66-ccc8-49b5-a36b-7bf34926f46e`, PID 575. Start 5077.431325, ready 5223.338235,
  release 5312.077982, SDHCI 5325.330935, SIGTERM 5325.455405, galcore release 5325.765411.
  Child exited −15, no final `execution.json`; correct recovered outputs do not pass health
  acceptance. No per-call trace identifies a destructor as causal. Evidence:
  `qwen-block-layout-20261110/shared-test/run-1/`, `shared-fault-analysis.json`.
- **SDHCI after DeltaNet diagnostic exit (11-09).** Boot
  `5d436758-f7d8-4ed2-8cdb-ecac5ac8de9c`, PID 5018. Start 193713.844239, ready 193714.628628,
  normal exit after 12.100 s, galcore release 193725.779864, SDHCI 193735.959044 (about 10.179 s
  later), caught by 10.072 s post-exit observation. Pools/owners returned; no NP hang/Oops.
  Recovered numerics passed but health acceptance failed. Evidence:
  `qwen-delta-prefill-progress-20261109/{run-1/,faulted-run-analysis.json}`.

The three recent storage events report command 0x00000d1a, argument 0xaaaa0000, interrupt status
0x00000001 and ADMA error 0. This supports a shared command-timeout signature, not a proven medium,
controller, clock, interrupt or NPU cause. Do not restart large reads or change clocks/drivers merely
because numerical output exists. Confirm storage recovery and revalidate affected deployments first.
After the latest recovery, two shared-projection Verify returns of −1 and a trailing head-control
preparation timeout had no new kernel fault and released normally. Their exact software failures
remain recorded; later acceptance does not rewrite them.

## Established diagnostic boundaries

- Verify can allocate, optimize, initialize, generate commands and possibly interact with hardware;
  it is not purely CPU compilation. A first hang inside Verify cannot be caused by later application
  upload/Run/readback/termination.
- Whole-model long waits were observed after submission, inside WaitSignal. Existing evidence does
  not distinguish unfinished execution from a missing completion signal. The single-layer thirteenth
  RunGraph's approximately 24 s wait was not reliably reproduced.
- Binding every FFN cache to a unique fixed path still hung; cross-layer cache hits are not a necessary
  cause. Cold boots and uncached fallbacks also have failures, so cache is not an established universal
  explanation. Keep accepted uncached paths; do not restore retired caches as a default.
- Old convolution controls shared three nodes/six tensors and effective precision/const/virtual
  contracts. Creation order and unused bias dimensions differed, but no causal mechanism was proven.
  Ordinary tensor creation passes only declared dimensions; this weakens, but does not universally
  eliminate, a tail-initialization explanation. Both vendor and repository binaries had successes and
  failures with matching libraries; a faulted-boot retry is not an independent cold-boot sample.
- Shutdown Oops disassembly located an invalid event-list head `0x100000000`, dereferenced at head+8,
  with queue head at Event+0x5e0. This is a driver event pointer, not a tensor physical address.
  Event allocation is 0x6a8 B. List/queue locks exist, but complete caller synchronization is unproven.
  Reviewed Recovery bodies did not directly rebuild head/tail/freeList; indirect paths remain unknown.
  No first corrupting write or proven use-after-free/race was captured. The roughly 30.72 s watchdog
  rhythm is not proof of initial cause or Verify complexity.
- Static EDMA destination all-ones can be an SG logging convention; see SDK findings. Never infer an
  illegal physical tensor address from that log alone.

Do not reproduce shutdown, creation-order or bias-tail lockups without a new discriminating hypothesis
and authorization. Existing offline evidence includes `graph-construction-comparison-20261008/`,
`retired-convolution-control-20261008/` and `convolution-event-offline-20261009/`.
Vendor follow-up should distinguish the first Verify hang from the later event-queue close fault;
the causal link is still open. External messaging requires authorization.

## Immutable historical record and incident index

The [original incident record](np101-recovery-history-20261110.txt) is retained byte-for-byte in its
original language, including all independent events, boot IDs, exact SDK/kernel timelines, artifact
hashes, recovery windows and evidence paths. It is historical evidence, not a second maintained guide.
Its Markdown paths/headings reflect the original document location and names. Read the corresponding
record before repeating a historical experiment; the English index is not a substitute for its timeline.
New events belong in this active English ledger; do not append to or translate the immutable archive.

Archive SHA256: `70016fd8e54280bd160c6bd8e0268f22ef4033e30642efef2d39a60131d8053b`.
Earlier compressed 11-02/03 details remain in
`.cache/runs/opt-fp16-repository-cleanup-20261105/before/docs/np101-recovery-incidents.md`
and its `before-manifest.json`; earlier 10-10/10-14 snapshots remain referenced in the archive introduction.
Raw evidence under `.cache/runs/` is not distributed with the source. Line ranges below refer to the
archived text, not the current file. Every original heading is indexed, including recovery and analysis
sections; a row is not necessarily a single fault occurrence.

| Historical record | Original lines |
|---|---|
| Original ledger introduction | 1–10 |
| Historical operating gates; superseded by the active ledger | 11–60 |
| 11-10 SDHCI command timeout after candidate deployment, before launch | 61–85 |
| 11-10 Fixed-row view copy mismatch; no device fault | 86–95 |
| 11-10 FFN-integrated state commit numerical failure; no new device fault | 96–129 |
| 11-10 Shared-residual Qwen inference completed; SDHCI timeout during release | 130–156 |
| 11-09 SDHCI command timeout after DeltaNet reduction comparison exited | 157–185 |
| 11-08 Four-step DeltaNet comparison completed; fixture completion record missing | 186–204 |
| 11-07 SDK rejected blocked-prefill probes; no new device fault | 205–229 |
| 11-07 Qwen FFN M2 progression hung in the sixth group | 230–251 |
| 11-07 Shared FCL head preparation timeout; no new device fault | 252–275 |
| 11-07 Original Qwen request passed within 600 s; old timeout retained | 276–295 |
| 11-06 Qwen initialized with 768 MiB Linux; first request hit original execution deadline | 296–313 |
| 11-06 C512 long request still hung with 512 MiB Linux and restored 1792 MiB pool | 314–338 |
| 11-06 Qwen retried with authorized 32 MiB margin; Linux-memory guard still stopped it | 339–358 |
| 11-06 Qwen diagnostic stopped at head preparation by Linux-memory guard | 359–387 |
| 11-06 Third C512 long-request hang; same-experiment testing stopped | 388–416 |
| 11-06 Second C512 long-request hang on the same deployment | 417–431 |
| 11-06 C512 long-request hang after reducing Linux memory | 432–453 |
| 11-05 Third history-replay attempt: SDHCI timeout in preparation | 454–469 |
| 11-05 Four-convolution-layer C512 history replay hung in the second turn | 470–494 |
| 11-05 Six uncached convolution layers: second long request hung | 495–515 |
| 11-05 SDHCI interrupt timeout during C512 preparation | 516–533 |
| 11-05 Eight-layer FP16 uncached formal entry: second long request hung | 534–550 |
| 11-05 Minimal direct-run FP32 historical fast configuration: short-request hang | 551–564 |
| 11-05 FP32 historical fast configuration after cold boot: second long-request hang | 565–577 |
| 11-05 FP32 historical fast configuration: SDHCI preparation timeout | 578–591 |
| 11-04 Original four-layer baseline exited abnormally after power-cycle recovery | 592–615 |
| 11-04 Four-layer per-call diagnostic hung on the first long request | 616–634 |
| 11-04 Four-layer foreground direct run hung on the second long request | 635–650 |
| 11-04 SDK identity-check anomaly before six-layer rerun | 651–661 |
| 11-04 Six-layer uncached candidate hung after cold boot | 662–675 |
| 11-04 Serial six-layer uncached FP16/FP32 comparison | 676–694 |
| 11-02–03 FP16, FFN and device-incident source index | 695–701 |
| 11-03 Eight-layer uncached, local-Verify comparison | 702–706 |
| 11-03 Three-input recovery after board power cycle | 707–712 |
| 11-03 Whole-model submission and wait diagnostic | 713–718 |
| 11-03 Single-layer FFN submission and wait diagnostic | 719–722 |
| 11-03 FFN direct import from fixed paths | 723–726 |
| 11-03 All-convolution FFN comparison | 727–730 |
| 11-03 All-shared-FCL comparison | 731–734 |
| 11-03 Four-layer cached fallback | 735–738 |
| 11-03 Eight-layer cached fallback | 739–743 |
| 11-03 FP16 cached fast configuration reproduced hang after cold boot | 744–747 |
| 11-03 Nineteen convolution layers with local Verify | 748–752 |
| 11-03 Uncached M16 fallback | 753–757 |
| 11-03 Authorized segmented diagnosis and structural workarounds | 758–762 |
| 11-03 FP16 KV long-input/full-capacity acceptance hangs | 763–768 |
| 11-03 Recovery, progressive-state comparison and FP16 KV model; no new device fault | 769–773 |
| 11-02 Original FP32 baseline hung in FP16 KV single-layer comparison | 774–779 |
| 11-02 Embedded single-layer FFN / overlapped CPU checking model hangs | 780–785 |
| 11-02 Review found an overlooked 10-30 SDHCI command timeout | 786–847 |
| 10-30 Current-SDK C128/M16 numerics passed; health acceptance corrected | 848–861 |
| 10-30 Long-input comparison with per-call tracing disabled | 862–907 |
| 10-30 New-SDK C256 long-input prefill hang | 908–954 |
| 10-30 New-SDK compatibility test of nineteen FFN caches hung | 955–982 |
| 10-30 New-SDK repeated single-decoder RunGraph hang and post-exit recovery Oops | 983–1028 |
| 10-30 New-image memory-map review and preventive configuration | 1029–1047 |
| 10-29 Deployment reads after cold reboot exposed EXT4 corruption | 1048–1070 |
| 10-29 Offline filesystem repair and read-stability checks | 1071–1096 |
| 10-28 C256 first-request timeout and post-exit driver-recovery Oops | 1097–1137 |
| 10-28 Two SDHCI timeouts during C256 deployment | 1138–1174 |
| 10-28 Initialization-optimized SoC model hung on first long request | 1175–1195 |
| 10-28 Initialization-optimized SoC whole-model acceptance hang | 1196–1214 |
| 10-28 SoC model deployment rejected by file integrity check | 1215–1236 |
| 10-28 Public-filter NBG counterexample: coefficients remained frozen | 1237–1264 |
| 10-28 Binding diagnostic rejected by readback-rank guard | 1265–1277 |
| 10-28 Native four-dimensional external-weight SoC FFN produced NaN | 1278–1300 |
| 10-27 Shared-output worker diagnostic omitted peak report | 1301–1320 |
| 10-27 Shared-prefill-output comparison failed in-process release boundary | 1321–1334 |
| 10-27 Attention-layout control failed in RunGraph | 1335–1354 |
| 10-27 Publicly generated attention NBG import rejected; user-space SIGSEGV | 1355–1380 |
| 10-27 SD-controller timeout during model-file preparation | 1381–1393 |
| 10-27 Nineteen-FFN deployment digest rejection; no NPU submission | 1394–1424 |
| 10-27 Twelve-FFN controlled-cache model release failure; no new kernel fault | 1425–1477 |
| 10-27 Nineteen-FFN cached preparation timeout; no new kernel fault | 1478–1494 |
| 10-27 Same-layer complete FFN comparison failure | 1495–1516 |
| 10-27 Single-layer attention/batched-convolution FFN fused graph rejected | 1517–1544 |
| 10-26 Profiler-off control failed | 1545–1562 |
| 10-26 Current-slot KV-output decoder comparison failed | 1563–1613 |
| 10-26 Generation failed after initialization attribution | 1614–1659 |
| SoC boot, configuration and bounded acceptance | 1660–1682 |
| 10-25 SoC functional-segment diagnosis | 1683–1788 |
| 10-26 Fused-prefill graph rejection and C128 acceptance | 1789–1812 |
| 10-24 Eight-decoder SoC graph hung on fourth input | 1813–1842 |
| 10-24 Follow-up diagnosis on the same boot | 1843–1860 |
| 10-23 Single decoder on the new SoC pool hung during repeated RunGraph | 1861–1897 |
| 10-23 Follow-up diagnosis and file-integrity rejection | 1898–1915 |
| 10-21 Host unresponsiveness; NP101 causality unconfirmed | 1916–1931 |
| Historical successful recovery windows (not current device state) | 1932–2065 |
| 10-16 OPT C64/M1 comparison: four-byte EDMA timeout; separate recovery event | 2066–2105 |
| 10-16 OPT C256 conversation: four-byte EDMA timeout; separate recovery event | 2106–2163 |
| 10-16 OPT FCL CPU Verify timeout and one authorized restart | 2164–2195 |
| 10-14 Qwen tracing-off comparison: four-byte EDMA timeout | 2196–2241 |
| 10-14 Qwen compiled FFN execution hang | 2242–2290 |
| 10-14 Qwen first embedding Verify hung after cold boot | 2291–2338 |
| 10-14 Qwen prefill NP hang, recurring after release | 2339–2383 |
| 10-13 v106 vendor convolution integer divide exception and delayed hang | 2384–2413 |
| 10-13 Frozen old OPT initialization hang after convolution retirement | 2414–2449 |
| 10-13 Frozen convolution failed again after another cold boot | 2450–2482 |
| 10-13 Frozen repository convolution failed after cold boot; post-release NP hang | 2483–2524 |
| 10-13 Block16 hybrid OPT trailing plain-layout run hung | 2525–2568 |
| 10-11 Eight-graph pure GEMM stalled; NP hang after release | 2569–2608 |
| 10-12 Source review and avoidance changes | 2609–2629 |
| 10-11 Driver event-queue page fault during warm reboot | 2630–2644 |
| Historical failure signatures | 2645–2660 |
| 10-09 EVIS projection completion wait stalled; post-release NP hang | 2661–2698 |
| 10-08 17:35 Vendor convolution hung again after cold boot | 2699–2727 |
| 10-08 17:20 Authorized vendor-convolution retry on the same boot | 2728–2755 |
| 10-08 16:52 Unmodified vendor-convolution NP hang | 2756–2794 |
| 10-08 16:31 Layer 24 K diagnostic readback EDMA timeout | 2795–2846 |
| 10-08 11:44 Candidate numerical/lifecycle acceptance EDMA timeout; cause unknown | 2847–2869 |
| 10-08 11:59 Recovery-control failure on the earlier boot | 2870–2893 |
| 10-08 14:40 Recovery-control failure after confirmed cold boot; marker later archived | 2894–2923 |
| 10-08 Shutdown page faults on two faulted boots | 2924–2968 |
| Offline convolution/shutdown conclusions | 2969–2974 |
| Graph-construction differences and exclusion limits | 2975–2988 |
| Event queue and shutdown crash analysis | 2989–3010 |
