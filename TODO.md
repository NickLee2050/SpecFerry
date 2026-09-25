# Remaining work

| Item | Next action | Acceptance / dependency |
|---|---|---|
| F1: minimal device checks | Selection passed after the September 25 warm restart; driver load and test-window kernel logs were clean | Old recovery marker expired with the previous boot; no new marker or residual processes |
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

- **MEM-001 — byte corruption:** driver 1.0.6 accepted 2,840 MiB in 8 MiB blocks;
  the next block was rejected. Both FP16/FP32 and const/mutable reproduced corruption
  in block 109, with three 8-byte ranges starting at local offsets 1,274,680,
  1,274,744 and 1,274,808. These are logical offsets, not physical addresses.
  Current const/nonconst payload caps remain 1 GiB each; they do not blacklist those
  locations or establish that corruption is fixed. Preserve historical larger runs.
- **MEM-003 — host growth:** repeated copy-graph rebinding/VerifyGraph increased RSS
  about 128 KiB per verification; fixed bindings stayed stable. A leak is unproven.
- **MEM-004 — SDK accounting:** FP16 counter delta was `2 * payload + 4,672` bytes;
  data matched and the counter returned after release. Physical doubling is unproven.
- **Driver wait:** on September 24, a tiny lookup using under 2 KiB application
  payload spent 22.37 s in VerifyGraph; the first RunGraph was interrupted by the
  30 s deadline. Kernel logs recorded two NP hangs/recoveries, including one before
  the deadline. Successful SDK return and process exit do not establish recovery.
- **Other retained workarounds:** biased FCL lowering crashed, so OPT uses MatMul+Add;
  direct FP16 categorical sampling was incorrect, so sampling promotes logits to FP32.
  Rank-three cache updates crashed; the experimental append uses validated 2D shapes.
- **F2 verification cost:** one-layer prefix Verify took 158.095 s before the restart
  and 157.887 s afterward. The latter run passed input/KV checks and released cleanly
  without kernel errors. Two layers previously exceeded the 300 s process deadline,
  with sampled CPU usage near 100%; no corresponding kernel NP hang was found.
  Do not infer that more nodes alone explain the cost, or retry two layers before
  completing the one-layer decoder/head diagnosis.

## Retained local evidence

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
