# Engineering follow-up

Updated: 2026-09-09. Target: resident text inference of Qwen3.5-0.8B on NP101.

## NP101-MEM-001: Enable full-model resident weight allocation

- [ ] Resolve the effective constant-tensor allocation limit and validate the
  vendor-supported allocation path.

Status: waiting for the NP101 development team. The user has sent the findings
to the team and will provide their driver update or alternative API usage.
Integration and verification belong to SpecFerry after that response arrives.

### Working constraint and evidence

Use **1 GiB (1,073,741,824 bytes)** as the provisional ceiling for the current
constant-tensor allocation path. It is a planning constraint, not a verified
description of all board memory or proof of which pool backs the tensors.
Allow room below it for SDK allocations, alignment, and any duplicated layouts.

- `vsi_nn_AddTensor` with `is_const=TRUE` retained 266 tensors containing
  1,070,874,144 bytes, then failed while creating a 7,340,032-byte FP16 tensor
  from `model.layers.20.mlp.down_proj.weight`.
- The same allocation boundary reproduced after a cold restart. The second
  run closed the device and exited with code 1 without timeout or residual
  children; the previous run remained in a driver mutex wait.
- Live driver parameters report `exclusiveSize=1,073,741,824` and
  `externalSize=2,008,023,040`. The selected constant-tensor pool is unconfirmed.
- The exported text weights contain 1,504,791,232 payload bytes. Weights plus
  planned state/auxiliary buffers have a known lower bound of 1,550,875,816 bytes,
  before unresolved SDK copies, activations, graph data, and workspace.
- The user additionally reports that a warm restart could not complete boot,
  while a cold restart restored the host. That observation has no independently
  captured boot-failure log here and must not be attributed to the memory limit
  without further evidence.

Local evidence is retained under
`.cache/runs/weight-allocation-after-reboot-20260909/`: `recheck-report.md`,
`comparison.json`, `driver-parameters.json`, `sdk.log`, and `driver.strace`.
The original run is under `.cache/runs/weight-allocation-first/`.

### Required follow-up and closure

1. Obtain the team's explanation of constant-tensor pool selection and a
   supported driver configuration, driver build, or alternative allocation API.
   Record affected versions, limits, ownership, and lifetime requirements.
2. Review and integrate that solution. Any sudo action or system configuration
   change still requires explicit authorization for that action.
3. Revalidate the affected tensor API and a previously supported small operation,
   including changing inputs, numerical comparison, cleanup, and device execution
   evidence. Full FP32 feedback capability remains a separate gate.
4. Re-run `scripts/check_np101_allocation.py` with a new output directory. All
   320 logical text-weight tensors, including their row blocks, and the planned
   state tensors must remain allocated together. Confirm shared embedding/head
   storage, successful release, and no remaining child processes. The number of
   SDK allocations can exceed 320 because weights are split into row blocks.
5. Repeat initialization/allocation/release with recorded memory observations;
   reject a growing allocation footprint, timeout, or failed cleanup. Document
   what supports device residency rather than counting SDK success alone.

Close this item after the supported resident allocation path passes these checks.
The full model's actual graph/workspace peak remains a later integration check;
it is not a prerequisite for closing this item, which would create a dependency
cycle. A vendor response alone does not close the item.

While waiting, do not repeat the known full-weight capacity probe without a
changed allocation path or new vendor guidance. Keep the model, precision policy,
and full-device target fixed. Per-token host weight streaming is outside the
resident-inference acceptance criteria.

## Dependencies and work that can continue

Split the existing weight-preparation prerequisite into three independently
tracked results: verified host export, validated layouts for the tensors used by
a subgraph, and full-model resident allocation. An isolated subgraph requires
the first two and its own memory check; it does not require all model weights to
be resident. The full-model allocation result remains blocked by NP101-MEM-001.

| Work item | Implementation while waiting | Device validation while waiting | Dependencies |
|---|---|---|---|
| Operator and state capability checks | Existing diagnostics can be completed and corrected. | Small and individual model-shape cases can run after device access/health checks. | Required FP32 math, feedback/reset, cache operations, and hardware evidence remain unresolved. |
| Weight export, integrity, layouts, and memory accounting | Host export and independent byte comparisons already pass; layout and accounting work can continue. | Validate selected weight tensors and their projections. Full-model simultaneous allocation remains blocked. | Selected-operator checks; NP101-MEM-001 for full resident allocation. |
| DeltaNet subgraph | Implement projections, short convolution, gates, FP32 recurrence, normalization, and reset. | Load one layer's weights and state; compare 1/2/4/8/32-token output and state traces, including nonzero initialization. | Its operators, state feedback, selected-weight layouts, and measured subgraph memory. |
| Attention subgraph | Implement Q/gate splitting, Q/K norm, partial RoPE, GQA, KV writes, masking, and output gate. | Load one layer; test positions 0/1/3/7/255/511, invalid-slot masking, and capacity rejection. | Its operators, persistent KV, layouts, and measured subgraph memory. |
| MLP and four-layer decoder group | Implement MLP, trunk normalization, residuals, and layers 0-3. | Test individual layers, then one complete four-layer group with state retained across tokens. | Validated DeltaNet and Attention; device-resident connections between operators/layers. |
| Embedding, LM head, and token selection | Implement lookup, fixed row blocks, valid tail rows, tie handling, and device-wide selection. | Test embedding and the full-vocabulary head in isolation using captured hidden states. Shared-table integration needs a validated storage/view strategy. | Gather, projection, device argmax/selection, supported sharing/layout, and measured memory. |
| Full 24-layer generation | Interfaces, sequencing, consumed-length semantics, EOS handling, and host-only contract tests can be prepared. | Full prompt/decode execution with all weights/state resident is blocked. | All subgraphs and head validated; NP101-MEM-001 resolved; final workspace allocation verified. |
| First complete DLM acceptance | Prepare fixed prompts, teacher-forcing cases, result schema, and benchmark collection. | Complete correctness, repeated reset/generation, full-model memory, and latency acceptance are blocked. | Working full 24-layer generation on NP101. |
| Spark TLM and speculative integration | Independent protocol/TLM preparation is technically possible, but remains deferred under the current single-DLM-first scope. | Real NP101-DLM/TLM integration and optimization measurements are blocked downstream. | Complete DLM acceptance, followed by the planned TLM/protocol stages. |

### Other capability gates remain independent

- FP32 state matrix operations, same-graph feedback/reset, and model-sized cases
  still need successful numerical and hardware validation. Host-buffer-backed
  state feedback does not meet device-residency acceptance.
- `vsi_nn_AttachTensorToGraph` is declared but not exported by the current SDK.
  An exported alternative or composition within one graph must be validated
  before relying on connections between subgraphs. Resolving NP101-MEM-001
  alone does not provide that capability.
- Device execution of argmax/selection and full-vocabulary gather/head remains
  unverified. CPU selection cannot satisfy the complete DLM requirement.

### Weight footprints for isolated tests

These are payload sums from the exported deployment manifest, not measured SDK
memory footprints. State, activation, layout copies, and workspace are additional.

| Isolated component | Weight payload |
|---|---:|
| DeltaNet mixer, layer 0, excluding its MLP | 20.11 MiB |
| Attention mixer, layer 3, excluding its MLP | 14.00 MiB |
| One MLP | 21.00 MiB |
| Complete layers 0-3, including their MLPs and norms | 158.35 MiB |
| Shared embedding/head table, counted once | 485.00 MiB |
| All text weights | 1,435.08 MiB |

These sizes support attempting isolated component tests under the provisional
limit, subject to actual allocation and execution checks. Load only the selected
component, keep its weights/state resident across tokens, then release it before
the next test. Do not combine all component fixtures into one resident test graph.
Head row blocking limits individual operations; it does not reduce total resident
weight bytes when all blocks remain allocated. Avoid assuming that manifest-level
weight aliases prove sharing inside the SDK.

## Recommended next implementation order

1. Complete the required small operator/state cases, then their individual real
   shapes and selected-weight layout checks. Stop on a failed capability gate.
2. Implement and validate one DeltaNet layer with persistent state and reset.
3. Implement and validate one Attention layer with persistent KV and boundaries.
4. Implement MLP/residuals and validate one four-layer decoder group.
5. Implement and validate standalone embedding, full-vocabulary head, and device
   token selection; validate their sharing strategy before combining them.
6. Stop before full resident 24-layer device integration until NP101-MEM-001 is
   resolved. Interfaces and host-only tests can be prepared without claiming
   successful device inference.

This is the maximum planned component-level progress allowed by the memory
constraint alone. Unresolved operator, state, or sharing capabilities can stop
the sequence earlier. Individual component passes never substitute for complete
DLM or speculative-inference acceptance.
