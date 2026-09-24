# Remaining work

| Item | Next action | Acceptance / dependency |
|---|---|---|
| Device recovery | Recover the device after the September 24 kernel hangs; run selection/lookup first | Numerical match, normal release and no new driver hang; retain the current recovery marker until recovery |
| O3/O4: fixed decode graph | Locate the first incorrect intermediate in lookup → layer → prefix → full graph | CPU token/KV agreement, repeated A/B/A reset, normal release; graph output was previously all zero |
| O5: block prefill | Validate block 4, then token tails using the same cache/weights | Causal masking, prefix preservation, correct predictions and no per-token VerifyGraph |
| O6: performance | Measure only after graph correctness passes | TTFT and decode tokens/s; compare identical prompt/output lengths; exclude initialization, warmup and diagnostic readback |
| S11: full-model acceptance | Complete OPT prefill/decode, KV and reset acceptance | All 24 layers, checkpoint-default output, clean lifecycle; component path remains the comparison baseline |
| Deferred | Physical residency/backend profiling, long contexts/compression, Qwen duplicate DeltaNet weights | Separate work; do not claim these are solved by numerical agreement |

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

## Retained local evidence

- Latest fault: `.cache/runs/segmented-regression-20260924/` (kernel log and tiny lookup).
- Graph diagnosis: `.cache/runs/graph-pipeline-*-20260924/` and
  `.cache/runs/o4-graph-decode-20260924/`.
- OPT CPU baseline: `.cache/runs/opt-validation-cpu-20260921/`.
- Qwen CPU baseline: `.cache/runs/decoupling-20260921-cpu/`.

Commands are in [tests/README.md](tests/README.md). Detailed historical narratives
and retired diagnostics remain in Git before the cleanup; checkpoints and exports
remain in `.cache/models` and `.cache/np101`.
