# Engineering follow-up

Updated: 2026-09-22. Active target: complete resident OPT-350M text inference on NP101.
Numerical SDK validation and formal hardware acceptance are tracked separately.

## Current work and dependencies

| Work | Current state | Next action / dependency |
|---|---|---|
| OPT input/output and generation (S9/S10) | Implemented and numerically validated on the earlier SDK | Retain the regression; revalidate execution after the new-package computation fault is resolved |
| Complete DLM hardware acceptance (S11) | Open | Resolve the new-package computation fault; establish backend/residency evidence and measure full-model memory and performance |
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
- [ ] Obtain a vendor-supported correction or explanation of the corruption.
- [ ] Revalidate retained bytes and normal release before increasing the target size.
- [ ] Revalidate all Qwen weights and the historical allocation-state fixture together.

Package 1.0.6 accepts all 502 Qwen weight/state tensors (1,550,863,040 bytes),
but the first readback mismatch is in `model.layers.17.mlp.gate_proj.weight`.
The strengthened checks locate it at byte 3,400,312 within the first 7 MiB chunk;
all 46,071,808 zero-initialized state bytes still compare correctly.

All four 64 MiB controls pass. All four 1,152 MiB runs (FP16/FP32 × constant/mutable)
fail at the same three eight-byte regions of block 109, starting at byte 1,274,680.
The largest earlier fully verified payload is 864 MiB in constant mode, a tested
lower bound rather than a physical limit. The old 1 GiB allocation ceiling is
historical; the current failure is retained-data integrity. Physical pool selection
and usable capacity near 4 GiB remain unverified.

Evidence and reproduction:
[capacity diagnostics](tests/np101-capacity.md),
[old constant/mutable comparison](tests/np101-mutable-allocation.md),
`.cache/runs/allocation-review-20260922/`.

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
  then revalidate the affected OPT numerical path on the corrected package.

Caller/SDK root cause remains unresolved; matching package files do not establish
correct computation. This is separate from the allocation-only readback failure.
See the [preflight record](tests/np101-capacity.md).
This item gates new-package computation acceptance, not CPU work or the recorded
allocation-only controls.

## NP101-OBS-001: Establish execution and device-memory evidence (S11)

- [ ] Obtain a supported trace/profiler that identifies each kernel's execution
  backend and completion, plus physical allocation/residency observations.
- [ ] Account for weights, KV, activations, SDK layout copies and workspace with
  the complete OPT model resident at the tested capacity.
- [ ] Verify release/recreation against device memory observations.
- [ ] Measure initialization, prompt processing and decoding after correctness
  and execution/residency are established; separate diagnostic tracing overhead.

Generic driver IO, SDK success, numeric agreement and `argmax.execute_on_sw=false`
do not establish exclusive NPU execution. Host RSS/FD stability is not a board
memory measurement. Existing reports retain this distinction.

Known OPT weight/KV payload at capacity 512 is 712,724,480 bytes (679.71 MiB),
excluding SDK overhead, activations and workspace. Use the
[complete-model checks](tests/np101-generation.md) as the numerical baseline.
This item gates hardware acceptance and device-performance claims for all modules.

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
