# Engineering follow-up

Updated: 2026-09-21. Active target: resident text inference of OPT-350M on NP101.

## NP101-MODEL-001: Decouple model composition and adapt OPT-350M

- [ ] Separate shared NP101 operators/storage from model-specific composition,
  then validate complete OPT-350M text generation.

The user selected `facebook/opt-350m` on 2026-09-21. The active download changes
to that checkpoint; existing Qwen3.5 caches, code and evidence remain available
for regression. The [implementation checklist](docs/model-decoupling-plan.md)
is pending review. Download and independent CPU FP16 import validation passed:
388 tensors, 331,196,416 unique parameters, 662,392,832 weight payload bytes;
three short generations and an eight-token cache comparison passed. Evidence
and a rerun script are in `.cache/runs/opt-350m-import-20260921/`.
The inference refactor, reusable OPT reference/export path and NP101 adaptation
have not started.

The Qwen weight, layer and allocation figures below describe the retained
baseline. OPT requires its own export, memory accounting, numerical validation
and execution/residency evidence. Its smaller weights do not establish that a
complete graph fits, but the Qwen weight-size blocker does not automatically
block OPT either. NP101-OBS-001 and the KV-related NP101-STATE-001 checks remain
applicable. Deferred DeltaNet weight duplication (NP101-MEM-002) is not an OPT
prerequisite.

## NP101-MEM-002: Share DeltaNet weights across alternating state graphs

- [ ] Eliminate duplicate weight storage between the A-to-B and B-to-A graphs.

Status: explicitly deferred by the user on 2026-09-21. Record the issue only;
do not implement graph fusion or change weight ownership in the current task.

Each `StepGraph` currently creates its own constant weight tensors. The first
four complete layers therefore load 218.68 MiB of weight payload instead of
158.35 MiB, including 60.33 MiB duplicated by the three DeltaNet mixers.
The duplication follows from separate allocation, not an inherent requirement
of using two graphs. Combining graphs alone does not establish weight sharing.

When resumed, validate a supported shared read-only allocation/packing path,
including both graph directions, numerical agreement, owner/consumer lifetimes,
reset and release. Measure SDK copies/layouts rather than inferring savings from
shared wrapper pointers. Evaluate graph consolidation only if necessary for a
validated storage/execution strategy.

Dependencies: existing two-graph execution and four-layer numerical validation
provide the baseline. This optimization does not block isolated component work
or the implemented decoder checks. Before full-model resident integration,
account for duplication across every DeltaNet layer or remove it through a
validated path. Resolving this item does not resolve NP101-MEM-001: unique text
weights alone still exceed the provisional 1 GiB allocation ceiling. Physical
memory savings and residency require NP101-OBS-001 evidence.

## NP101-MEM-001: Enable full-model resident weight allocation

- [ ] Resolve the effective tensor allocation/upload limit and validate the
  vendor-supported allocation path.

Status: waiting for the NP101 development team. The user has sent the findings
to the team and will provide their driver update or alternative API usage.
Integration and verification belong to SpecFerry after that response arrives.

### Working constraint and evidence

Use **1 GiB (1,073,741,824 bytes)** as the provisional ceiling for the current
tested constant and mutable allocation/upload paths. It is a planning constraint, not a verified
description of all board memory or proof of which pool backs the tensors.
Allow room below it for SDK allocations, alignment, and any duplicated layouts.

- `vsi_nn_AddTensor` with `is_const=TRUE` retained 266 tensors containing
  1,070,874,144 bytes, then failed while creating a 7,340,032-byte FP16 tensor
  from `model.layers.20.mlp.down_proj.weight`.
- The same allocation boundary reproduced after a cold restart. The second
  run closed the device and exited with code 1 without timeout or residual
  children; the previous run remained in a driver mutex wait.
- On 2026-09-17, all weights were created with `is_const=false` and explicitly
  uploaded. The first 266 chunks uploaded 1,070,874,144 bytes, exactly matching
  the constant-mode boundary. The next tensor object was created, but uploading
  its 7,340,032 bytes failed in `vsi_nn_CopyDataToTensor` with status -5
  (`VX_ERROR_NOT_ALLOCATED`). It was the same layer-20 MLP down-projection weight.
  Full weight readback and state allocation were not reached. The process exited
  normally with code 1, without a timeout or residual child. A bounded follow-up
  passed mutable FP16/FP32 weight readback and state allocation/release.
  Changing `is_const` alone therefore does not bypass the observed limit; the
  evidence no longer supports treating it as necessarily exclusive to constants.
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
The mutable experiment is under `.cache/runs/weight-allocation-mutable-20260917/`;
see [the allocation comparison](tests/np101-mutable-allocation.md).

### Required follow-up and closure

1. Obtain the team's explanation of constant/mutable tensor pool selection and a
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

The small constant, mutable, and mixed-weight graphs now pass numerical and
lifecycle checks with changing inputs/weights. This permits a separate, bounded
comparison of allocation paths before a full-model retry. It does not demonstrate
different physical pools, extra capacity, or device residency. See the
[operator acceptance record](tests/np101-operator-acceptance.md).
The all-mutable comparison has now been performed once and reproduced the same
upload boundary. Do not repeat either full-capacity path without another concrete
change or vendor guidance. Actual DeltaNet/Attention state reuse remains a
separate unresolved item below.

## NP101-STATE-001: Validate device-resident state reuse

- [ ] Implement and validate state reuse/reset without a per-token host state copy.

Status on 2026-09-17: a C++ layer-0 DeltaNet mixer now implements fixed A-to-B and
B-to-A graph execution with shared ordinary tensors. Two-step and 32-step
real-weight tests passed numerical/lifecycle checks, including nonzero initial
state, reset, recreation and final-only readback. There are no application state
uploads between steps. Layer-3 Attention now uses one preallocated FP16 K/V cache
with slot views; its 512-token suite passed 1,042 appends and 207 checks, including
truncate/overwrite, reset, fresh-instance comparison, final-only readback and
capacity rejection. KV payload is 1 MiB per layer, with only the new 2 KiB token
submitted to the slot-copy graph on each step. **SDK-internal transfers and
physical residency remain unverified. This item stays open.**
See the [DeltaNet validation record](tests/np101-delta-net.md) for the isolated
API exception, duplicate graph weights, SDK warnings and acceptance evidence.
The [Attention validation record](tests/np101-attention.md) documents its view/
ownership API exception, tested Softmax layout and the small copy graph's
per-append revalidation overhead. Dynamic KV allocation remains deferred.

Decoder composition now uses fixed shared inputs/outputs across all four complete
layers, including both DeltaNet graph directions. Initial integration exposed
FP16 core subnormal loss; the sensitive core dot product now stays FP32 through
gated RMS reduction. Independent CPU arithmetic and frozen tolerances are unchanged.
The group validates positions before execution and requires recreation after a
partial SDK failure. It does not claim recurrent-state rollback through KV truncation.
See the [decoder validation record](tests/np101-decoder.md) for current trajectories,
transfer counters, known memory payload and timing observations.

The 22 SDK RNN feedback cases and temporary buffer tests remain retired in the
[investigation archive](tests/state-feedback-investigation.md). Handle swapping
has not been restored. Their removal did not resolve this item.

Required implementation and acceptance:

1. Use the chip team's `demo/ref_op_api_guide.md` as the interface baseline for
   actual DeltaNet/Attention state. If it cannot express the required reuse or
   incurs a significant predictable performance cost, document that specific
   limitation before adopting and validating an alternative.
2. Verify a representative changing-input trajectory for recurrent state,
   convolution history, and KV storage against independent references. Preserve
   necessary FP32 state precision, check intermediate results, and ensure KV
   writes retain untouched rows and reject out-of-capacity positions.
3. Reset to a defined initial state and compare the subsequent outputs with
   both the reference and a fresh instance using the same input sequence.
4. Run continuously with final-only application readback and establish that the
   SDK does not transfer the state through the host between steps. Matching the
   final result or having no application copy alone is insufficient. Check normal
   release/recreation and retain separate hardware/residency evidence.

Implement these checks with the real modules rather than restoring a generic
matrix of delays, flushes, and SDK feedback variants. Use representative
configurations for trajectory, reset, and transfer-free execution.

This gates acceptance of device-resident sequential DeltaNet, Attention/KV,
four-layer decoder groups, and full generation. Isolated operator checks,
selected-weight projections, MLP, and bounded allocation experiments can continue;
DeltaNet/Attention implementation can proceed while establishing their state path.

## NP101-OBS-001: Obtain execution and device-memory evidence

- [ ] Correlate graph nodes with actual NP101 execution/completion and observe
  device allocation/release through a supported profiler or diagnostic API.

The installed target-query headers describe available targets and kernel support,
not the selected execution backend of every node. Driver IO, successful output,
`argmax.execute_on_sw=false`, and a swappable tensor flag do not establish full
hardware execution or residency. All current operator reports retain these gates.

Four representative cases each complete 20 graph lifetimes. File descriptors
stabilize after initialization and host RSS settles after early growth, but board
memory counters are unavailable. Do not infer leak-free device allocation or use
traced diagnostic timings as performance benchmarks.

This blocks hardware acceptance and trustworthy device memory/performance claims
for all modules; it does not block host implementation or numerical diagnostics.

## Qwen baseline dependencies and work that can continue

Split the existing weight-preparation prerequisite into three independently
tracked results: verified host export, validated layouts for the tensors used by
a subgraph, and full-model resident allocation. An isolated subgraph requires
the first two and its own memory check; it does not require all model weights to
be resident. The full-model allocation result remains blocked by NP101-MEM-001.

| Work item | Implementation while waiting | Device validation while waiting | Dependencies |
|---|---|---|---|
| Operator and state capability checks | Retain operator checks; implement state reuse/reset checks with the actual DeltaNet/Attention modules. | Targeted operator regressions can run after device health checks; state acceptance remains pending. | NP101-STATE-001 for state reuse/reset; NP101-OBS-001 for hardware evidence. |
| Weight export, integrity, layouts, and memory accounting | Host export and independent byte comparisons already pass; layout and accounting work can continue. | Validate selected weight tensors and their projections. Full-model simultaneous allocation remains blocked. | Selected-operator checks; NP101-MEM-001 for full resident allocation. |
| DeltaNet subgraph | Layer-0 C++ mixer and fixed state routing implemented. | Numerical trajectories through 32 steps, nonzero state, reset and recreation pass; actual backend, SDK state transfers and board memory remain unverified. | NP101-STATE-001 and NP101-OBS-001 for resident hardware acceptance; graph composition must revisit duplicated weights. |
| Attention subgraph | Layer-3 C++ mixer and single-buffer KV append/reset/truncate implemented. | Real-weight 512-token trajectories, invalid-slot masking, capacity rejection and lifecycle pass; actual backend, SDK transfers and board memory remain unverified. | NP101-STATE-001 and NP101-OBS-001 for resident hardware acceptance; copy-graph revalidation remains a performance concern. |
| MLP and four-layer decoder group | Complete layers 0-3 and fixed shared activation bindings implemented. | Complete Attention layer and four-layer 2/32/512-step numerical/lifecycle checks pass, including capacity rejection, reset/fresh/final-only equality and explicit-transfer limits. | NP101-STATE-001 and NP101-OBS-001 still gate resident hardware acceptance. |
| Embedding, LM head, and token selection | Implement lookup, fixed row blocks, valid tail rows, tie handling, and device-wide selection. | Test embedding and the full-vocabulary head in isolation using captured hidden states. Shared-table integration needs a validated storage/view strategy. | Gather, projection, device argmax/selection, supported sharing/layout, and measured memory. |
| Full 24-layer generation | Interfaces, sequencing, consumed-length semantics, EOS handling, and host-only contract tests can be prepared. | Full prompt/decode execution with all weights/state resident is blocked. | All subgraphs and head validated; NP101-MEM-001 resolved; final workspace allocation verified. |
| First complete DLM acceptance | Prepare fixed prompts, teacher-forcing cases, result schema, and benchmark collection. | Complete correctness, repeated reset/generation, full-model memory, and latency acceptance are blocked. | Working full 24-layer generation on NP101. |
| Spark TLM and speculative integration | Independent protocol/TLM preparation is technically possible, but remains deferred under the current single-DLM-first scope. | Real NP101-DLM/TLM integration and optimization measurements are blocked downstream. | Complete DLM acceptance, followed by the planned TLM/protocol stages. |

### Other capability gates remain independent

- Model-sized FP32 matrix operations and the four real-weight projections now
  pass numerical checks. Resident feedback/reset remains blocked by
  NP101-STATE-001, and hardware proof by NP101-OBS-001.
- `vsi_nn_AttachTensorToGraph` is declared but not exported by the current SDK.
  Retained ordinary tensors passed DeltaNet and Attention numerical sharing
  checks. New composed graphs still need their own integration and residency
  checks; resolving NP101-MEM-001 alone does not provide that evidence.
- Full-vocabulary gather, individual head blocks/tail, and block token selection
  pass their numerical probes. The complete blocked head is not yet integrated;
  device execution remains unverified. CPU selection cannot satisfy the complete
  DLM requirement.

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

## Retained Qwen follow-up order

The active OPT implementation order is in the model-decoupling checklist above.
The following sequence describes the previous Qwen scope and its allocation gate.

1. Reuse the validated operator results; check changed operators and selected-weight
   layouts as needed. Do not restore the retired SDK RNN diagnostic matrix.
2. Retain the implemented DeltaNet mixer and its numerical state/reset checks;
   obtain the missing hardware/residency evidence without repeating the retired
   diagnostic matrix.
3. Retain the implemented Attention layer and its single-buffer KV/boundary
   checks. Investigate copy-graph revalidation cost and obtain hardware/residency
   evidence during later integration; do not add dynamic allocation yet. The complete four-layer untraced
   observation averaged 447.4 ms per group step, including 2.30 ms of KV graph
   revalidation. These are host times with an unknown SDK backend; establish
   execution evidence before drawing device-performance conclusions.
4. Retain the complete MLP/residual/normalization and four-layer decoder group,
   with its numerical, state, boundary and explicit-transfer checks. See the
   [decoder validation record](tests/np101-decoder.md).
5. Implement and validate standalone embedding, full-vocabulary head, and device
   token selection; validate their sharing strategy before combining them.
6. Stop before full resident 24-layer device integration until NP101-MEM-001 is
   resolved. Interfaces and host-only tests can be prepared without claiming
   successful device inference.

This is the maximum planned component-level progress allowed by the memory
constraint alone. Unresolved operator, state, or sharing capabilities can stop
the sequence earlier. Individual component passes never substitute for complete
DLM or speculative-inference acceptance.
