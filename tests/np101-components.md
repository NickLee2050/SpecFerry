# Reusable component contracts and regression

This record covers the completed separation of checkpoint/reference tools,
export/storage, SDK graph helpers, state and decoder slices. The subsequent OPT
adapter and generation work is recorded in [OPT validation](np101-opt.md) and
[complete-model validation](np101-generation.md).

## Dependency boundaries

| Shared mechanism | Model-owned policy/composition |
|---|---|
| `data/checkpoint.py`: manifest and indexed safetensors integrity | Qwen identity, tensor classification and name mapping |
| `reference/tensors.py`, `validation/arrays.py`: snapshots and comparisons | Official Qwen modules, hooks, cache protocol, precision and tolerances |
| `export/weights.py`, `export/memory.py`: bounded payload IO, explicit dtype conversion, integrity and resource totals | Exact checkpoint schema, target dtypes, tied alias and historical memory resources |
| `native/np101/ops/`: graph construction, projection, normalization, Attention core, short convolution, DeltaNet recurrence and SwiGLU | Qwen projection paths, norm conventions, RoPE, gates and residual/layer order |
| `tensor.*`, `component_spec.*`, `kv_cache.*`: bindings, dimensions and one K/V allocation with slot views | Model dimensions, capacity and selected layer list |

Shared production libraries never import model modules. Scripts and tests import
Qwen-specific behavior directly from `specferry.models.qwen3_5`; the nine temporary
compatibility facades have been removed. `reference/tensors.py` remains a shared
utility. CLI names, checkpoint formats and Qwen regression entry points are retained.
`Context`, graph ownership, downloads, Conda and the raw SDK fixture protocol retain
their existing responsibilities. The SDK probes remain independent of production
arithmetic, and expected decoder values use official Transformers computation.

## Explicit contracts

`components.txt` starts with `specferry-qwen-components 1`. Its second line contains:

```text
hidden intermediate query_heads kv_heads head_dim capacity rotary_dim rope_theta epsilon delta_heads key_dim value_dim convolution_width
```

The third line lists the model's `delta`/`attention` layer kinds. `layers.txt` selects
an ordered decoder slice by zero-based layer number. Native tests require these
files and reject missing/unknown versions rather than guessing model dimensions.
Python prepares them from the verified checkpoint configuration; small synthetic
fixtures provide their own configuration. Weight shapes and dtypes are checked
before constructing component graphs.

The shared arithmetic retains FP16 projection/storage, FP32 sensitive reductions,
the corrected FP32 DeltaNet core, and FP32 recurrence. RMS epsilon comes from model
configuration; Qwen L2 uses the official function's independent `1e-6` epsilon.
Qwen requires equal DeltaNet key/value head counts, unbiased gated Attention,
default text RoPE and SiLU. Projection currently allows at most two 4096-row blocks
with bounded 8 MiB weight reads. KV capacity is 1–512. These are explicit implementation
limits; parameterized interfaces do not establish that every shape runs on this SDK.

KV SDK shape is `[head_dim, capacity, kv_heads]`; a slot is `[head_dim, 1, kv_heads]`.
Only the new slot is copied. Reader graphs keep referencing the parent buffers.
Reset/truncate only change the valid prefix. DeltaNet still uses two state banks
and two execution graphs; deferred weight duplication remains unchanged.

The physical `weights.index` version stays 1. `WeightStore` uses exact physical
names without a `model.` prefix requirement or implicit aliases. Qwen's export
manifest records the tied LM-head alias; the unused native alias resolver has been
removed. Generic manifest aliases must point directly to
physical records; dangling, chained, cyclic and colliding aliases are rejected.
Conversions are explicit: BF16→F16, F16→F16, or F32→F32. Other conversions fail.

Old deployment packages and saved comparison reports remain readable. New native
binaries require newly prepared component fixtures; rerun the existing CLI with
`--prepare-only` rather than applying defaults to an archived fixture. Historical
binaries/reports remain available for exact reproduction.

Allocation diagnostics accept an optional versioned `--state-spec FILE` listing
`DTYPE SDK_SHAPE COPIES`; omitting it checks weights only. The historical Qwen
fixture is stored explicitly in `tests/fixtures/qwen3_5_allocation_states.txt`,
with its original order and allocation-only KV shape `[256,2,512]`.
This is not the runtime KV layout. Small component checks use actual state shapes.

## Commands

Use the `SpecFerry` Conda environment and a fresh output directory:

```bash
cmake --build build -j 4
python -m unittest discover -s tests -t .
ctest --test-dir build --output-on-failure
python scripts/check_np101_components.py --output .cache/runs/components --diagnostic
python scripts/check_np101_decoder.py --layers 0 1 --steps 2 \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/decoder-short-slice --diagnostic
```

The synthetic test reuses the production decoder and existing native test binary.
It runs a 64-wide DeltaNet/Attention pair, MLP width 96, 4 query/2 KV heads,
head dimension 16, rotary dimension 8, DeltaNet dimensions 2×8×8, convolution width 3
and capacity 8. Seeded synthetic weights feed independent official CPU computation;
this test makes no claim about text quality or full-model trace alignment.
It then tests a separate KV layout with 3 heads, head dimension 16 and capacity 8.
`--prepare-only` requires no device. `--diagnostic` permits numerical success to
exit zero while execution-backend and physical-residency evidence remain unresolved.

## Decoupling acceptance record

All checks below passed on 2026-09-21. Paths are relative to `.cache/runs/`.

| Check | Evidence / result |
|---|---|
| Host contracts and quality | 55 Python tests; CTest; full build; all public headers self-contained; Ruff/clang-format and include checks |
| Both CPU precision modes | `decoupling-20260921-cpu/regression.json`: 1/2/4/8-token alignment and 718 historical trace comparisons per mode |
| Existing Qwen export | `decoupling-20260921-export.json`: all 320 tensors byte-identical to independent PyTorch conversion |
| Small component decoder + alternate KV | `decoupling-20260921-release-components/`: 353 decoder checks, capacity rejection, alternate KV exact writes and incompatible binding rejection |
| DeltaNet, Attention mixers | `decoupling-20260921-delta-4/`, `decoupling-20260921-attention-2/`: 260 / 107 comparisons |
| Complete Attention layer / two-layer DeltaNet slice | `decoupling-20260921-attention-layer-2/`, `decoupling-20260921-slice-0-1/`: 97 / 193 checks |
| Four-layer 32 / 512 steps | `decoupling-20260921-group-32/`, `decoupling-20260921-group-512/`: 737 / 801 checks |
| Bounded allocation and readback | `decoupling-20260921-allocation/{constant,mutable}/`: 118,156 weight bytes with explicit small states, both passed |
| Convolution baseline | `decoupling-20260921-release-conv_relu_pool/`: two repetitions, normal release |
| Raw SDK probes | `decoupling-20260921-release-operators/`: FP16 MatMul, FP32 RMS and L2 normalization passed |

The 512-step suite completed 560 total steps including reset/final-only/fresh,
rejected token 513, uploaded 1,151,360 bytes (2,056 per step), and performed zero
explicit intermediate reads. Physical weight duplication and KV copy/reverification
behavior remain unchanged. No full-capacity allocation pressure test was run.

These results establish numerical/lifecycle regression, not exclusive NPU execution
or physical state residency. The operator summary therefore still reports `blocked`
with `numeric_pass: true`; the existing evidence gates are intentionally unchanged.
