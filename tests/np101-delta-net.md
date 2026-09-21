# Single-layer DeltaNet validation

The retained default fixture exercises Qwen3.5-0.8B linear-attention mixers for layers 0-2.
The standalone regression uses layer 0; normalization, residuals and MLP are
provided by the [decoder group](np101-decoder.md). This is not full-model inference.

## Computation and precision

`native/models/qwen3_5/delta_net.cpp` constructs the projections, four-tap depthwise
convolution, SiLU, Q/K L2 normalization, decay/beta gates, FP32 recurrent update,
gated RMS normalization and output projection. Convolution products and their
reduction use FP32, with one FP16 boundary before SiLU. Q/K normalize in FP32,
round to FP16, then enter the FP32 recurrence. The core dot product stays FP32
through gated RMS reduction because the installed SDK flushes FP16 subnormals.
After normalization, the original FP16 boundary is retained before FP32 weighting
and gating. The CPU reference remains unchanged; its core snapshot is only
serialized as FP32 for comparison with the revised device diagnostic. No tolerances are calibrated
from device results; the existing reference policy supplies the limits.

The independent reference executes the installed Transformers mixer using real
exported weights and previously captured single-token hidden vectors. Inputs
cycle after the saved sequence ends. These are numerical layer trajectories,
not generated text. Both zero and seeded nonzero histories are tested.

## State routing and API exception

All arithmetic uses operators listed in `demo/ref_op_api_guide.md`. Each graph
has fixed input/output bindings. Two SDK graphs alternate A-to-B and B-to-A;
the C++ step counter selects the graph. Recurrent state and convolution history
are never downloaded and reuploaded by `step`. Reset explicitly initializes both
banks; diagnostic reads are separate calls.

The guide's AddTensor interface creates graph-owned tensors and provides no
cross-graph shared-state ownership interface. The installed SDK declares but
does not export AttachTensorToGraph. A static feed-forward graph cannot express
the next execution as an ordinary acyclic edge; a per-token host state roundtrip
would violate the resident-inference requirement.

The common `retain_tensor` helper therefore retains an existing `vx_tensor` using
`vxRetainReference`, releases the receiver wrapper's unused tensor if present,
and assigns the retained object to that wrapper before graph setup. This goes
beyond the guide's interfaces. It uses public SDK tensor fields and OpenVX
reference ownership, not tensor-handle swapping or private node state. Each
wrapper releases one reference; the second graph is released first. This passed
the module's numerical/lifetime checks below but remains an experimental SDK
integration pending hardware and residency evidence. Pointer identity alone does
not prove physical storage placement or absence of SDK-internal transfers.

For now the two graphs load separate constant weights (about 40.22 MiB combined).
This avoids adding a second unvalidated weight-sharing mechanism. State, graph
temporaries and SDK workspace are additional; a measured full-model fit is not
implied. Later graph composition must revisit this duplication.

## Run

Use Conda `SpecFerry`, the exported pack and a deployment-FP16 layer-0/3 reference
trace. The runner verifies the pack and records source/weight provenance.

```bash
cmake --build build --target np101_delta_net_check -j 4
python scripts/check_np101_delta_net.py \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/delta-net --steps 32 --diagnostic
```

`--prepare-only` generates CPU fixtures without opening the device. Allowed
sequence lengths are 1, 2, 4, 8 and 32. One 32-step trajectory also checks all
shorter prefixes. The native process runs zero-initialized and nonzero-initialized
trajectories, zero reset after nonzero use, nonzero final-only readback, and a
fresh instance. Outputs and both states are compared at every observed step;
reset/fresh/final-only results must additionally match the corresponding device
trajectory exactly. Missing output or incomplete release fails acceptance.

The existing serialized device runner, recovery guard and bounded timeout remain
in force. Exit 0 with `--diagnostic` means numerical and lifecycle agreement only;
without it, numerical success exits 2 while hardware/residency evidence is
missing. JSON reports keep those gates separate. Full hardware acceptance still
requires a supported per-node execution and SDK state-transfer observation path.

## Results

Historical validation on 2026-09-17, before the FP32-core correction, with the installed SDK and exported Qwen3.5-0.8B weights:

| Check | Result |
|---|---|
| Official isolated CPU mixer versus the earlier full-model trace, first 8 inputs | Bitwise-identical FP16 outputs |
| Two-step trajectories and lifecycle | Pass, 112 array comparisons |
| 32-step zero/nonzero trajectories, including prefixes 1/2/4/8 | Pass; all 1,552 reference/repeat array comparisons pass |
| Reset, fresh instance and final-only versus the matching observed trajectory | Bitwise-identical outputs and states |
| Separate one-step run, checking reset after odd buffer parity | Pass, 64 array comparisons |
| Python host tests, native CPU CTest, full build, Ruff and clang-format | Pass |
| Per-node NPU execution and physical state residency | Unverified |

Across the 32-step zero/nonzero sequences, maximum output absolute error was
0.04541015625 and maximum FP32 recurrent-state error was 0.0028028488159179688.
Both passed the existing elementwise pipeline tolerances; these maxima were not
used to change thresholds. The 32-step process returned 0, released its graphs,
left no children and required no device recovery. Diagnostic timings are not
performance measurements.

SDK logs contain 12 `Call vxBatchGemmNode fail` warnings per complete suite,
despite subsequent successful verification/execution and correct results. These
indicate failed VX GEMM node creation attempts; the eventual backend is unknown.
Do not interpret the results as proof of full NPU execution or hide the warnings.

Local evidence (ignored by Git):

- `.cache/runs/delta-net-20260917-blocked-projection/`: two-step results.
- `.cache/runs/delta-net-20260917-trajectory-32/`: complete 32-step results and
  `captured-reference-comparison.json`.
- `.cache/runs/delta-net-20260917-odd-reset/`: one-step/reset-parity results.
- `.cache/runs/delta-net-20260917-summary/`: consolidated verification and source snapshot.

Each run retains its binary, exact fixtures, SDK logs, driver trace, SDK/source
hashes and numerical report. The first attempt under
`.cache/runs/delta-net-20260917-two-step/` failed normally during initialization
because the QKV read exceeded the existing 8 MiB chunk limit. The implementation
now splits the 6,144-row projection into 4,096- and 2,048-row blocks; the failure
is not included among the successful results above.

Attention/KV is now implemented and numerically validated separately; see the
[Attention record](np101-attention.md). Remaining DeltaNet acceptance work is
supported backend/state-transfer observation and later graph composition,
including weight sharing and actual device-memory accounting.

## Decoder integration correction

On 2026-09-18, complete decoder composition exposed an error hidden by the
isolated mixer's absolute tolerance. In the first captured core output, all 943
nonzero FP16 subnormal reference values became zero in the SDK output. Feeding
that SDK core into the CPU gated norm/output projection reproduced the approximately
0.03662 mixer-output error. Subsequent decoder normalization amplified it.

The core dot product now remains FP32 until normalization. Model weights, state
precision, independent CPU arithmetic and tolerance thresholds are unchanged.
The regression additionally records the projected gate and gated output to locate
this failure before accepting a composed decoder. Its `core` diagnostic is now
FP32; older FP16 fixtures must be regenerated. The bound interface shares one
fixed input/output between both state graphs and supports layer-specific weights.
See the current decoder validation record for the corrected integration results.

The corrected 32-step standalone suite passed all 1,940 reference/repeat checks
on 2026-09-18, including nonzero state and both graph directions. Maximum mixer output error
was 0.00048828125, compared with 0.04541015625 in the historical suite. Evidence:
`.cache/runs/decoder-20260918-delta-regression-32/`. The initial gated-output
failure is retained in `.cache/runs/decoder-20260918-delta-gated-diagnostic/`.
The 12 matrix-node creation warnings persisted in the corrected standalone suite;
numerical success does not close the hardware execution/residency gates.

## Component separation (2026-09-21)

Model composition now lives in `native/models/qwen3_5/`; shared graph construction
and arithmetic live in `native/np101/ops/`. Dimensions, state sizes and layer kinds
come from the versioned `components.txt` generated by the Qwen fixture builder.
The values described above belong to the retained baseline, not shared-library defaults.
See [component contracts and regression](np101-components.md) for alternate dimensions,
explicit decoder slices, compatibility rules and the decoupling validation record.
