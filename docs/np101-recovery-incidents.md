# NP101 recovery gates and incident ledger

Updated: 2026-11-14. Scoped formal Qwen regression passed after an explicitly authorized soft restart;
historical fault markers remain. This does not establish a general SDHCI recovery rule.
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

Earlier documented recovery followed the 11-10 storage incident, after a user-confirmed board
power cycle to boot `1e21c890-d53a-49e5-a531-3e28415f0e6c`. On the unchanged 768 MiB Linux /
2048 MiB NPU configuration, six-tensor IO and a three-input decoder passed numerics, release,
complete kernel and 10 s observation; old markers were archived by identity. Evidence:
`qwen-shared-projection-resume-20261110/recovery/` under `.cache/runs/`.
The 11-11 OPT/Qwen migration, C++ runtime extraction and native-supervisor regressions also
passed their scoped checks on this boot; see status and `native-supervisor-20261111/acceptance.json`.
Those regressions used unchanged model binaries, with bitwise tokens/state, released owners/pools/GFP
and clean complete kernel/10 s observation. No new fault was recorded; exact historical
software-failure markers remain retained. This does not validate intentional hardware-fault recovery.
These are recorded windows, not a live health check or a storage-root-cause fix.

Recent events that affect subsequent work:

### 11-14 explicit soft-restart diagnostic authorization

The user authorized one formal C512 241→16 / reset 241→16 process after a soft restart,
acknowledging the old SDHCI incident only. Board boot became
`7e2e98f0-6f89-4f97-b3c9-93927aea6068`; no power cycle was confirmed. Initial complete kernel,
EXT4 and ownership checks were clean. Existing go.sh loaded the unchanged 768 MiB Linux /
2048 MiB NPU layout, with stuckDump=1 restored.

Full deployment verification caught four stale SDK identities in the packaging snapshot; all
132 payload hashes matched. Correcting five SDK-bound descriptions and their inventory left
weights, graphs, requests and board libraries unchanged. Verification then passed.
An exact-boot/marker/bundle-bound diagnostic launcher retained the prior-boot gate; shipped
supervision and all new-fault checks remained intact. The run passed numerical/reset acceptance,
exit 0, preparation 165.098 s and post-ready execution/release 248.549 s, within 240/300 s limits.
Owners cleared, SYSTEM/VIRTUAL returned to 122976/0 B and GFP to 5808 pages; complete kernel
and 10.071 s observation were clean. Both old markers were restored byte-for-byte.
This scoped success does not clear the gate or establish general SDHCI recovery.

Evidence: `qwen-block-prefill-soft-restart-20261114/` under `.cache/runs/`: `comparison.json`,
`native-run/`, `final-health.json`, deployment identities/audit and the exact authorization launcher.

The user then authorized a two-turn retained/replay comparison followed, on success, by a
long/short/long collection on the unchanged boot. Both all-18 M8 processes passed numerical,
execution and release checks with no new kernel or filesystem faults. Preparation limits stayed
240 s; execution limits were selected before launch as 300 s and 360 s, with actual post-ready
phases of 170.257 s and 303.630 s. Delayed observations were 10.084 s and 10.042 s.
Owners cleared and SYSTEM/VIRTUAL/GFP returned to 122976 B / 0 B / 5808 pages.
The diagnostic authorization bound the exact boot, old marker, bundle and preceding accepted
formal evidence. Both historical markers were preserved byte-for-byte; no gate was cleared and
no power cycle was claimed. Full deployment verification refreshed the source receipt after
hardlink creation changed inode metadata; source payloads were unchanged.
Evidence: `.cache/runs/qwen-complete-block-sessions-20261114/`, including both case directories,
full kernels, deployment identities and `conversation-final-health.json` / `stress-final-health.json`.

### 11-13 formal-entry deployment blocked by storage

Board boot `b68f12c0-f75c-425a-a7bd-8f2ac3687f2f`, host boot
`a57ec676-0192-462f-bd13-eb1fc801487c`. Preflight found only the recorded `[571.159097]` NP hang,
baseline resources and zero EXT4 errors. During staging, at kernel `19788.055039`, SDHCI reported
`Timeout waiting for hardware cmd interrupt`: command `0x00000d1a`, argument `0xaaaa0000`,
interrupt `0x00000001`, ADMA error 0. No model child or Setup/Verify/RunGraph had started.
The signature matches earlier storage failures; its cause is not established.

Staging was terminated and board copy PID 1455 killed; subsequent inspection confirmed exit,
empty owners, baseline pools/GFP and zero EXT4 errors. No repair or further board write followed.
The host marker retained this event and the previous marker; the board marker stayed unchanged.
A same-boot read-only recheck on 11-14 found no new fault, but did not establish recovery.
The later explicitly authorized soft-restart acceptance is recorded separately above.

Preserve incomplete-deployment evidence; require fresh full verification before execution and
stop on new storage faults. Evidence under `.cache/runs/`: `qwen-block-prefill-formal-20261113/`
(`storage-fault.json`, full kernels, `after-storage-stop.json`, identities and prior marker),
`qwen-block-prefill-recheck-20261114/`. The isolated-NP-hang retry exception does not cover SDHCI.

### Split-projection investigation, 11-12–13

Every row below is a separate NP hang, not a merged incident. All used 96 inputs / 16 requested
outputs. Every faulted child exited −15, with owners/pools/GFP returned and no additional fault
in the stated delayed observation. Complete logs, source/binary/NBG/SDK/bundle identities,
markers and supervisor records remain under `.cache/runs/<evidence>/`. Failed runs have no
accepted throughput. Times are seconds; kernel and application monotonic clocks are not
calibrated enough for sub-second causal ordering.

Boot identities:

- `1e21c890-d53a-49e5-a531-3e28415f0e6c`, host `fb760029-11c8-4c36-a3d9-bd88642a8c20`.
- `7bda02b4-5af7-4208-b2ca-8e2c05c30323`, host `7871ff3a-841a-458d-ab1b-f4405434dd4d`.
- `b68f12c0-f75c-425a-a7bd-8f2ac3687f2f`, host `a57ec676-0192-462f-bd13-eb1fc801487c`.

| Board boot prefix; experiment | SDK/application interval | Kernel hang / recovery / release | Delayed check | Evidence |
|---|---|---|---:|---|
| `1e21c890`; all 18, attempt 1 | Ready 128272.366339; request unfinished, call unknown | 128302.103113 / 128302.134953 / 128302.410183 | 10.042 s | `qwen-shared-delta-all-20261112/all-model-test/run-1/` |
| `1e21c890`; all 18, attempt 2 | Ready 128476.346302; request unfinished, call unknown | 128547.859113 / 128547.886869 / 128548.189672 | 10.066 s | Same root, `run-2/` |
| `1e21c890`; all 18, attempt 3 | Ready 128725.802430; request unfinished, call unknown | 128773.143094 / 128773.170865 / 128773.484419 | 10.079 s | Same root, `run-3/` |
| `1e21c890`; two layers, explicit one-shot exception after stop | Ready 137858.234358; request unfinished, call unknown | 137978.899131 / 137978.930925 / 137979.196989 | 10.046 s | `qwen-shared-delta-two-20261112/two-once-test/run-1/` |
| `1e21c890`; second layer only | Ready 139274.878142; 12 M8 and first M1 calls returned; last marker before public upload/FFN | 139371.539104 / 139371.566873 / 139371.856168 | 10.055 s | `qwen-shared-delta-second-only-20261112/native-run/` |
| `1e21c890`; unchanged two-layer replay | Ready 142871.214318; no per-call trace, request unfinished | 142976.019142 / 142976.046951 / 142976.289438 | 10.010 s | `qwen-shared-delta-two-replay-20261112/model/` |
| `1e21c890`; sixteen layers | Position 0, layer 14 M8 recurrence 145717.554918→145740.799178 (23.244 s); output projection starts 145740.800088, no end | 145740.819126 / 145740.846869 / 145741.147220 | 10.051 s | `qwen-shared-delta-full-coverage-20261112/sixteen/` |
| `1e21c890`; twelve layers | Prefill completed; decode position 102, layer 0 M1 recurrence starts 189412.958248, no end | 189445.139104 / 189445.170958 / 189445.442110 | 10.064 s | `qwen-shared-delta-twelve-20261113/twelve/` |
| `1e21c890`; exact accepted eight-layer replay | Position 56, layer 9 M8 recurrence starts 189767.211248, no end | 189793.299113 / 189793.330930 / 189793.588066 | 10.062 s | `qwen-shared-delta-eight-replay-20261113/eight/` |
| `7bda02b4`; independent state/device-copy handoffs | Prefill completed; decode position 97, layer 9 M1 recurrence starts 903.058341, no end | 937.491153 / 937.519003 / 937.767954 | 10.084 s | `qwen-independent-recurrence-20261113/bound-deployment/` |
| `7bda02b4`; independent state/mapped handoffs | Prefill completed; decode position 109, layer 4 M1 recurrence starts 2436.942134, no end | 2463.251164 / 2463.286970 / 2463.552722 | 10.033 s | `qwen-mapped-state-handoff-20261113/` |
| `7bda02b4`; separate FFN state commit | Position 56, layer 4 recurrence 3289.589363→3313.151034 (23.562 s); export interrupted before destination Unmap; no FFN/commit reached | 3313.171162 / 3313.198861 / 3313.491752 | 10.102 s | `qwen-separated-state-commit-20261113/` |
| `7bda02b4`; M1-only recurrence, M8 resident | Position 8, layer 4 M1 recurrence starts 4493.974105, no end | 4521.491166 / 4521.522885 / 4521.794450 | 10.061 s | `qwen-single-recurrence-resident-block-20261113/` |
| `b68f12c0`; M1-only, M8 removed, after cold recovery | Position 0, layer 6 recurrence 547.598879→571.139032 (23.540 s); output projection starts 571.139623, no end | 571.159097 / 571.186810 / 571.454809 | 10.103 s | `qwen-single-recurrence-minimal-20261113/` |

**Retry and recovery boundaries.** The all-18 sequence stopped after three failures. The next
two-layer run used explicit one-shot authorization, not a silently reset counter. A frozen
one-layer control then passed; second-only failed, followed by accepted localization and three
trace-ablation controls. Two-layer replay failed; bounded two/four/eight-layer controls passed.
Sixteen/twelve/eight then reached another three-failure stop. After the next confirmed power
cycle, agreed six-tensor IO and three-input decoder controls passed; independent-state failed,
the frozen original passed, then mapped handoff/separate commit/M1-only-resident reached three.
After the next confirmed power cycle and accepted controls, M1-only without M8 failed once.
Complete-block two-layer and then all-18 controls passed on that boot. Accepted controls reset
only the consecutive failure streak; all historical incidents and markers remain preserved.

**Established facts versus hypotheses.** Initialization completed in all inference failures.
All-18 split projections fit (SYSTEM 1811.321 MiB), while successful/failed eight-layer samples
shared the same artifacts and SYSTEM peak 1696.371 MiB. No stable failing-layer threshold was
found. Private state, mapped handoffs, removal of FFN copies and removal of resident M8 graphs
did not eliminate failure. The cold-boot M1-only failure rules out a requirement for earlier
same-boot hangs or M8 execution. Static audit found no established early free/missing blocking
wait; it does not establish closed-SDK alias correctness. Long recurrence returns preceding
later hang reports do not make the later outstanding API the initiating cause. No root cause
is proven. Keep split-projection variants out of the formal entry; do not repeat them unchanged.

**Non-device failures retained:** the combined projection/FFN wrapper first rejected mismatched
versions (`qwen-shared-delta-ffn-20261112/combined-test`, 5.500 s), then matching 1.1.34 wrappers
returned Verify −1 (`aligned-tail-test`, 5.741 s); release/kernel checks passed. Obsolete one-shot
launchers also rejected pre-child authorization, and the first independent-state package had a
frozen-path mismatch. Those were software gates, not allocation leaks or NP hangs. Their original
reports remain; the accepted-result retry reconciliation regression is kept in the test suite.

### Complete-block acceptance and current recovery boundary

On boot `b68f12c0-f75c-425a-a7bd-8f2ac3687f2f`, complete-M8 graphs without split projections
passed two-layer replay and all-18 C128/C512 short, long, full-capacity and reset requests.
Each exited 0 with accepted CPU tolerances, owners/pools/GFP released, zero EXT4 errors and clean
complete kernel/10 s observation. Only the explicitly authorized const cap changed to 1408 MiB;
nonconst remained 1024 MiB. Identities and exact per-run timing remain in the evidence directories
indexed by [status](np101-status.md#complete-block-qwen-prefill). This is bounded acceptance,
not persistent reliability or a general recovery guarantee.

The boundary run's initial host oracle incorrectly required physical clearing after logical KV
reset. Saved data proves correct new prefix and unchanged masked suffix; only the oracle was
corrected, with no rerun/tolerance relaxation. Initial reports and pending markers remain as
`initial-*`; `reset-review.json` explains acceptance. The previous exact board fault marker
`a9fb662da40f6c199c291579557849974005d5b4a64a277e442acb282af4093b` was restored. The host marker
`692bb5def0fc12f8cd59f43b3db0a80d4634446121625c58b9249aca830d9d70` also remains. The accepted
retry record points to the last complete acceptance; it does not erase `[571.159097]` or clear
the recovery gate. Next run must read the complete kernel and reconcile that exact result and
marker, with owners empty, SYSTEM/VIRTUAL 122976/0 B and GFP 5808 pages at baseline.

- **11-12 conversation fixture packaging error, no device fault.** On board boot
  `1e21c890-d53a-49e5-a531-3e28415f0e6c` / host boot
  `fb760029-11c8-4c36-a3d9-bd88642a8c20`, bundle
  `879a81929dc1941252be67acf487579a1ebd919fb7e4761b15b3c6ab23cd81ca` omitted the
  `expected-tokens.txt` files used by its diagnostic entry. After the first generation returned,
  reference parsing raised `invalid prompt token`, before measured output was saved. Child
  monotonic start/ready/exit were 124302.681/124448.690/124506.679 s; exit was 1, with no signal
  or outstanding SDK call established. Owners/pools/GFP returned, full kernel and 10.066 s
  observation were clean. Preserve the failed package and marker; do not count it as numerical
  acceptance or an NP hang. Retry only the corrected, fully verified package under the authorized
  conversation validation, retaining the exact software-failure marker and all health gates.
  Evidence: `qwen-release-acceptance-20261112/conversation-test/run-1/`. The corrected package
  includes both oracle files in its manifest; source/reference models and NBGs are unchanged.
  The corrected run completed all four generations and released normally with no device fault.
  Its first comparison incorrectly demanded physical zeroing after a logical long→short reset.
  Reanalysis of unchanged saved data verified the written prefix, exact retained stale suffix and
  zero never-written suffix, plus bitwise replay/continuation state; original tolerance checks
  were unchanged. The initial report remains at `conversation-complete-test/initial-acceptance.json`.
  Normal collection/acceptance then passed without another inference run; the exact earlier
  software-failure marker remains retained, not cleared.

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
