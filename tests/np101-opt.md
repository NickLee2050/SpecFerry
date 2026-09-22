# OPT checkpoint and decoder validation

Status on 2026-09-22: **CPU, export, single-layer and four-layer numerical/lifecycle
checks passed after avoiding an SDK optimizer crash.** This covers the model adapter and existing component/slice scope.
Full embedding, all-layer execution, LM head and device text generation are not
implemented by this check.

## Checkpoint and precision

- Model: `facebook/opt-350m`, revision `08ab08cc4b72ff5593870b5d527cf4230323703c`.
- Source: `pytorch_model.bin`, SHA256
  `a5223ae6f3c26c6d90003f96a6bcd9a4aaaef0d36fca6469112efeeb985f2842`.
- 388 physical tensors, 331,196,416 parameters, 662,392,832 bytes; all original FP16.
- Hidden width 1024, FFN width 4096, 16 heads, 24 layers; embedding width 512.
- Post-norm LayerNorm with weight/bias and epsilon `1e-5`; biased Q/K/V/out and
  FFN projections, ReLU, learned positions, no RoPE or DeltaNet.

The loader verifies the pinned download manifest and source digest before
`torch.load(weights_only=True, mmap=True)`, then validates exact names, shapes,
precision and finite values. The official CPU model loads with strict keys and
shared embedding/head storage. Export uses the common bounded pack writer and
native index version 1; an independent comparison checks every packed byte
against original PyTorch storage. It does not convert BF16/FP32 weights to FP16.
FP32 intermediate normalization and Softmax arithmetic remain separate from
checkpoint precision.

## Implementation and acceptance

`python/specferry/models/opt/` owns checkpoint contracts, official references,
export and comparisons. `native/models/opt/` owns layer names, post-norm ordering,
ReLU selection and configuration. Shared `np101/ops/GraphBuilder` implements biased
projection with documented `MATRIXMUL` followed by `ADD`, plus LayerNorm arithmetic;
existing Attention and KV code is reused. Biased FCL is not used by the production
adapter because its graph verification crashes in the installed SDK.

MatMul stores an FP16 intermediate before bias addition, adding a rounding boundary
relative to a fused projection. This is an activation difference, not a checkpoint
conversion. Official CPU expectations and predeclared thresholds remain unchanged;
the checks include each projection and the complete downstream decoder trajectory.

The projection graph produces FP16 Q/K/V; Q is scaled before QK to match the
official OPT operation order. A slot-copy graph appends only the new K/V to one
preallocated pair. The consumer graph reads that pair, applies prefix masking,
Attention, residuals, LayerNorm and FFN. Adjacent layers share ordinary activation
tensors. No whole-cache copy or host state readback is part of a normal step.
The existing KV view/ownership API exception and per-append revalidation cost
remain as documented in [Attention validation](np101-attention.md).

The suite checks 14 boundaries per layer, including scaled Q, K/V, probabilities,
residuals, normalization, FFN and cache contents. Expected values come from official
`OPTDecoderLayer` execution with independent cache storage and exported weights,
not from rewritten production arithmetic. Initial outputs must also match the
full CPU model trace. Long runs cycle captured boundary inputs; they exercise
state trajectories and **are not generated text**.

Frozen tolerances: CPU `atol=0.02, rtol=0.01`; device `atol=0.08, rtol=0.02`, using
the existing decoder integration budget. Reset/fresh/final-only repeats must match
device outputs exactly. Masked probabilities must be zero; only the valid cache
prefix participates after reset. NaN, missing/short outputs, abnormal exit, failed
release, unintended explicit host reads or incorrect transfer counts fail.
Normal steps upload one 2048-byte hidden vector and four bytes per selected layer.
Capacity is fixed at 1–512; this suite selects one to four contiguous layers.

## Reproduction

From the repository root in the `SpecFerry` Conda environment:

```bash
python scripts/reference_opt.py --output .cache/runs/opt-cpu
python scripts/export_opt.py
python scripts/export_opt.py --verify-only
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target np101_opt_decoder_check -j 4

# Offline expectations; no device access.
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 1 2 3 --steps 512 --prepare-only --output .cache/runs/opt-prepared

# Device checks require a healthy device and new output directories.
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 --steps 8 --capacity 8 --diagnostic --output .cache/runs/opt-layer
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 23 --steps 8 --prompt 1 --diagnostic --output .cache/runs/opt-last-layer
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 1 2 3 --steps 32 --diagnostic --output .cache/runs/opt-group-32
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 1 2 3 --steps 512 --timeout 900 --diagnostic --output .cache/runs/opt-group-512
```

`--diagnostic` allows numerical/lifecycle success to return zero while hardware
execution/residency remain unproven. Without it, numerical-only success returns 2;
validation failure returns 1. SDK initialization failure is reported separately
from numerical disagreement. A passed software reference is never device acceptance.

## Evidence and remaining work

Paths below are relative to `.cache/`:

| Check | Result | Evidence |
|---|---|---|
| Official full CPU model | 159 comparisons passed; three short generations | `runs/opt-validation-cpu-20260921/reference.json` |
| Native FP16 export | All 388 tensors byte-identical; no precision conversion | `np101/opt-350m/source-verification.json` |
| Layer 0 / layer 23 expectations | Passed trace alignment | `runs/opt-layer-prepare-20260921/`, `runs/opt-last-layer-prepare-20260921/` |
| Four layers, 512-step expectations | Prepared; original prefix aligned | `runs/opt-group-512-prepare-20260921/fixture/reference.json` |
| Layer 0 device attempt | SIGSEGV in initialization; zero completed steps | `runs/opt-layer-device-20260921/device/` |
| Host regression and code checks | 61 Python tests, CTest, full native build, self-contained headers, Ruff and clang-format passed | Repository host test commands |

The original process returned `-11`, fault address `0xb0`, after driver activity was observed.
`execution.json` remained at `phase=initialize`, with no captured output.
All traced processes exited. The runner set `runs/np101-recovery-required.json`;
its existing recovery guard prevented automatic retries. The available Apport file
describes `strace`, not the decoder, so it does not locate the failing SDK call.
Current-user journal access did not expose kernel logs; no sudo was used.

## Diagnosis and correction on 2026-09-22

The user authorized basic health checks, then OPT diagnosis. On the same boot,
constant and mutable paths each loaded/read back 25 weights (118,156 bytes), and
two convolution/ReLU/pooling iterations reproduced the earlier outputs. All
processes exited normally. Evidence: `runs/device-health-20260922/summary.json`.

GDB located the fault during `vsi_nn_VerifyGraph` of the first Q/K/V projection:

```text
vxoGraphOptimization_getKernelType
vxoGraphOptimization_ConvertMaxPool2Conv
vxoGraphOptimization
vxoGraph_VerifyGraph
vxVerifyGraph
GraphBuilder::compile
Projection::Projection
```

The faulting instruction dereferenced `0xb0` from a zero register inside
`libOpenVX.so.1`. An isolated 8x8 biased FCL, without model weights, cache or
cross-graph bindings, reproduced the same stack. All four tensor handles were
non-null and passed `vxGetStatus` before verification; the FCL node handle was
also non-null. The failing graph contains no POOL node and does not replace
`nn_param` or `pool.local`, so this is distinct from the demo README's deinitialization
mistake. The SDK's internal reason for the null reference still needs vendor analysis.

Two alternatives were rejected before adoption: FP16-input/FP32-output MatMul
returned an explicit unsupported-dtype error, and a 1x1 convolution projection
hit the same optimizer fault. Ordinary FP16 MatMul plus FP16 Add passed a small
exact-output probe and the actual OPT regression. No internal SDK pointer or
undocumented optimizer switch was patched. Required node input/output arrays are
checked before dereference; SDK-owned parameter state remains intact.

Temporary probes, immutable binaries and GDB logs are under
`runs/opt-debug-20260922/`. They are diagnostic artifacts, not new permanent test
entry points. Existing OPT suites provide the regression for production `linear`.

| Corrected check | Result | Evidence under `runs/` |
|---|---|---|
| Layer 0, capacity 8 | 322 checks, 32 total steps; capacity rejection, reset/fresh/final-only agreement | `opt-fixed-layer-20260922/opt.json` |
| Layer 23, second prompt | 322 checks, 32 total steps | `opt-fixed-last-layer-20260922/opt.json` |
| Four layers, 32-step trajectory | 1,345 checks, 80 total steps, 320 KV appends | `opt-fixed-group-32-20260922/opt.json` |
| Four layers, 512-step trajectory | 1,465 checks, 560 total steps, 2,240 KV appends; token 513 rejected | `opt-fixed-group-512-20260922/opt.json` |
| Retained Qwen decoder and alternate KV layout | 353 decoder checks and KV suite passed | `opt-fix-qwen-regression-20260922/` |

All four corrected OPT runs passed the original `atol=0.08, rtol=0.02` budget.
The largest observed absolute difference in the four-layer capacity run was
0.0625. Exact reset/fresh/final-only repeats and masked-suffix checks passed. Normal
steps made zero explicit intermediate reads; the capacity suite uploaded 1,155,840
bytes across 560 steps (2,048 hidden bytes plus four 4-byte length controls per step).
The 61 Python tests, native CTest, complete build and both formatting checks passed.

The first corrected layer reused the original frozen fixture under
`runs/opt-layer-device-20260921/fixture/`. Subsequent runs snapshot their own fixtures,
binary and source hashes. The prior recovery marker was archived in
`runs/opt-fixed-layer-20260922/previous-recovery.json` after the corrected layer
passed reset, recreation, release and normal process exit; no driver change, sudo
or server restart was used. New abnormal exits must still stop automatic tests.

Exclusive NPU execution and physical residency remain unproven. The whole-model
memory report is a lower bound (weights plus KV); SDK layouts, activations and
workspace remain unknown. Slice success does not establish full-model allocation.
