# OPT checkpoint and decoder validation

Status on 2026-09-21: **CPU and export passed; device validation blocked during
initialization.** This covers the model adapter and existing component/slice scope.
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
ReLU selection and configuration. Shared `np101/ops/GraphBuilder` adds biased FCL
projection and LayerNorm arithmetic; existing Attention and KV code is reused.
FCL is documented in the chip team's guide; this configuration has **not yet
passed device validation**. Do not infer SDK support from its presence in the guide.

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

# Offline expectations; safe while device recovery is pending.
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 1 2 3 --steps 512 --prepare-only --output .cache/runs/opt-prepared

# Only after device recovery has been confirmed; use new output directories.
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

The process returned `-11`, fault address `0xb0`, after driver activity was observed.
`execution.json` remained at `phase=initialize`, with no captured output.
All traced processes exited. The runner set `runs/np101-recovery-required.json`;
its existing recovery guard prevents automatic retries. The available Apport file
describes `strace`, not the decoder, so it does not locate the failing SDK call.
Current-user journal access did not expose kernel logs; no sudo was used.
The cause is unresolved; do not label this as a memory-capacity or FCL defect.

After confirmed recovery, obtain a debugger backtrace, fix the demonstrated cause,
then pass the device commands in increasing scope. No device result or exclusive
NPU execution/residency claim is available yet. The whole-model memory report is
a lower bound (weights plus KV); SDK layouts, activations and workspace remain
unknown. Selected-slice success would not establish full-model allocation.
