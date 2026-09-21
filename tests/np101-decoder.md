# Complete decoder validation

`np101_decoder` uses explicit component parameters and an ordered layer list.
The default fixture runs the first four complete Qwen3.5-0.8B text decoder layers:
three DeltaNet layers followed by Attention. `--first-layer 3` selects the complete
Attention layer alone. This slice consumes captured hidden vectors; it does not
include embeddings, the remaining 20 layers, final norm, LM head or token selection.

## Computation and ownership

Each layer executes input RMS normalization, its mixer, residual addition,
post-mixer RMS normalization, and `down(SiLU(gate(x)) * up(x))` plus the second
residual. Hidden size is 1,024 and MLP intermediate size is 3,584. Both norms use
FP32 reduction and `(1 + weight)`, then return FP16. MLP and residual boundaries
follow the existing deployment policy.

The group owns its input storage. A layer owns its normalization graph, mixer
output storage, mixer and final residual/MLP graph. The next layer retains the
previous layer's output tensor. Consumers are destroyed before producers, including
partial initialization failures. Bound hidden tensors must be ordinary mutable,
materialized FP16 `[1024, 1]` tensors in the same context. Binding occurs before
compilation; no handles are swapped during execution.

Both DeltaNet state graphs share the same fixed public input and output. Only
recurrent/convolution banks alternate. Layers 0, 1 and 2 load their respective
weights and own separate state. The ownership mechanism reuses the already tested
`retain_tensor` exception: the SDK lacks an exported attachment API, while routing
these edges through host copies would introduce a transfer between every graph.
The expanded usage is validated by the complete-layer and group trajectories.
No additional undocumented arithmetic operator is introduced.

Attention retains one 1 MiB K/V allocation and appends only the current 2 KiB slot.
It still computes over the fixed 512-slot extent and reverifies its small copy
graph when the destination slot changes. Dynamic allocation and recurrent rollback
are not implemented.

## Precision correction

The initial composed test failed, despite the earlier isolated DeltaNet pass.
The SDK FP32-to-FP16 core conversion lost all 943 reference subnormals in the first
captured token. Replaying the SDK core through the CPU gated norm/projection
reproduced the mixer error, which subsequent decoder normalization amplified.

The sensitive core dot product now stays FP32 through RMS reduction. Its FP16
rounding boundary is deferred until after normalization. The official CPU model
continues to use its original arithmetic; acceptance thresholds are unchanged.
This is a deliberate precision difference, covered by the original error budgets,
not a claim of bitwise equivalence to the CPU model. Gate and gated-output snapshots
were added to the existing DeltaNet regression to expose this failure earlier.

## State and failure contract

Before any layer executes, the group checks input size, capacity and agreement
between all layer positions. Position advances only after every selected layer succeeds.
A mid-step failure may already have changed a subset of states, so the group
becomes unusable and requires recreation. No rollback by changing a counter is
claimed. Normal reset clears both DeltaNet banks, resets Attention's valid prefix
without clearing its stale suffix, and invalidates old output reads.

The suite checks invalid input and token 513 rejection before any layer counter or
KV write advances. SDK failure handling is conservative code-path protection;
the suite does not deliberately inject a driver fault or hang to exercise it.

## Run and acceptance

```bash
cmake --build build --target np101_decoder_check -j 4
python scripts/check_np101_decoder.py \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/decoder-two-step --steps 2 --diagnostic
python scripts/check_np101_decoder.py \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/decoder-capacity --steps 512 --timeout 900 --diagnostic
```

Use a fresh output directory. `--prepare-only` runs only the CPU reference;
`--layers 0 1 --steps 2` selects a shorter DeltaNet slice;
`--first-layer 3 --steps 2` isolates the outer decoder computation. The CPU reference
loads only the selected layers into the installed official Transformers modules.
Its inputs come from the saved layer boundary **before input normalization**.
The first eight available layer-0/3 outputs are checked against the earlier full
model trace. Layers 1 and 2 receive independently generated reference snapshots.

One suite runs the requested initial trajectory, an up-to-eight-step reset run,
a final-only run of up to 32 steps, and an up-to-eight-step fresh-instance run.
Selected checkpoints compare both norms, mixer, residual, MLP, layer output,
recurrent/convolution state and K/V against the CPU. Repeat comparisons are exact;
only inactive KV suffixes are excluded from repeat equality. Each normal step
must perform exactly three explicit helper uploads totaling 2,056 bytes and zero
helper reads: one input vector and two INT32 Attention controls. Diagnostic reads
are outside the step. These counters do not observe SDK-internal transfers.

The runner preserves fixtures, source/binary/SDK hashes, logs, process evidence,
comparison results and payload accounting. It serializes device access and stops
further work after a timeout requiring device recovery. Numerical/lifecycle pass
returns zero only with `--diagnostic`; hardware acceptance remains unproven.

## Memory and timing scope

The four layers have 158.35 MiB of unique weights. Retaining the current two-graph
DeltaNet implementation duplicates 60.33 MiB, giving 218.68 MiB of loaded weight
payload. Recurrent/convolution banks and the single K/V cache add 7.28 MiB.
Activations, auxiliary constants, transformed layouts and SDK workspace are extra;
these are payload counts, not measured board occupancy or evidence of extra capacity.
No full-model allocation probe is repeated.

The native report separates initialization, reset, release, overall step time,
per-layer norm/mixer/MLP time, KV write time and the revalidation portion of that
write time. Byte counts cover explicit application transfers and slot copies.
Use a separate `--no-trace` run for host timing observations. Times collected with
driver tracing are diagnostic only; even untraced times do not identify NPU kernels,
prove physical residency, or constitute full-model throughput.

## Results

Validation on 2026-09-18 uses the installed SDK, exported real weights and frozen
CPU tolerances. Evidence is under `.cache/runs/`:

| Run | Scope | Result |
|---|---|---|
| `decoder-20260918-attention-2-run` | Complete layer 3, two steps plus reset/final/fresh | Numerical and lifecycle pass |
| `decoder-20260918-group-2-fixed-reference` | Initial four-layer integration | Failed numerical checks; normal release; retained as FP16-core failure evidence |
| `decoder-20260918-delta-gated-diagnostic` | Isolated gate/core diagnosis before correction | Gated-output failures retained for diagnosis |
| `decoder-20260918-group-2-fp32-core` | Corrected four-layer two-step suite | 385 checks pass across eight group steps |
| `decoder-20260918-group-32` | Four-layer 32-step suite and repeats | 737 checks pass across 80 group steps |
| `decoder-20260918-group-512` | Capacity trajectory, token 513 rejection, reset/final/fresh | 801 checks pass across 560 group steps |
| `decoder-20260918-delta-regression-32` | Corrected standalone DeltaNet, including nonzero state | 1,940 checks pass |
| `decoder-20260918-attention-regression-2` | Standalone Attention regression | 107 checks pass |
| `decoder-20260918-timing-32` | Separate untraced 32-step suite and repeats | 737 checks pass; host timing only |

The 32-step suite's maximum complete-layer output error was 0.0009765625;
maximum recurrent-state error was 0.0030875206. Both graph directions and
final-only comparisons pass without a small-hidden-vector copy fallback.
All steps used the same fixed cross-graph tensor bindings.

The 512-token suite rejected the next token before any layer advanced; all 560
normal group steps performed exactly three uploads totaling 2,056 bytes and zero
explicit helper reads. The 560 KV writes each triggered a copy-graph revalidation. Across captured
512-step checkpoints, maximum complete-layer output error was 0.00390625 and
maximum recurrent-state error was 0.0036687851. Both instances released normally;
there was no timeout, residual process or recovery requirement.

Standalone regressions passed after the shared-binding and precision changes.
The untraced run also passed all 737 checks. Across 80 group steps, host-observed
step time averaged 447.4 ms. KV slot writing averaged 2.49 ms, of which 2.30 ms
was graph revalidation. These measurements include SDK dispatch/completion, use
the fixed 512-slot Attention graph, and are not a full-model or NPU-only benchmark.
Initialization, reset, diagnostic reads and release are excluded from step timing.
Duplicate DeltaNet weights, backend selection and MLP/mixer execution remain
visible costs; this observation alone does not justify optimizing KV revalidation
before identifying where the much larger compute time is spent. `NP101-OBS-001` and `NP101-STATE-001` remain
open: numerical success and explicit transfer counters do not prove every node's
hardware backend or SDK-internal device residency. `NP101-MEM-001` still blocks
full resident 24-layer inference. The known DeltaNet matrix-node creation warnings
remain in logs; the suite does not infer their backend or ignore numerical failure.

Host validation passed 47 Python tests and the CTest data-contract test. All eight
public NP101 headers compile independently, matching source/header include sets
are disjoint, and C++/Python formatting checks pass. Only two explicit standard
header includes were added after device runs; the final decoder binary's `.text`,
`.rodata`, `.data` and `.eh_frame` sections match the capacity-tested binary exactly.
This comparison is recorded in `.cache/runs/decoder-20260918-summary/`.

## Component separation (2026-09-21)

Model composition now lives in `native/models/qwen3_5/`; shared graph construction
and arithmetic live in `native/np101/ops/`. Dimensions, state sizes and layer kinds
come from the versioned `components.txt` generated by the Qwen fixture builder.
The values described above belong to the retained baseline, not shared-library defaults.
See [component contracts and regression](np101-components.md) for alternate dimensions,
explicit decoder slices, compatibility rules and the decoupling validation record.
