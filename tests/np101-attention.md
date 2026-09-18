# Single-layer Attention and KV cache validation

The C++ module implements Qwen3.5-0.8B layer 3's Attention mixer: Q/gate split,
Q/K RMS normalization, partial RoPE, grouped-query Attention, masking, output
gate and output projection. Decoder input normalization, residual and MLP are
outside this module. Prefill can feed tokens sequentially; batched prefill and
full-model generation are not implemented here.

## Storage and execution

One sequence owns one preallocated FP16 K tensor and one V tensor, each logically
`[2, 512, 256]`. Their combined payload is **1 MiB**. The current token's K/V
payload is **2 KiB**. Graph temporaries, views, weights and SDK workspace are
additional; these payload sizes are not measured board-memory consumption.

Each synchronous `Attention::step` does the following:

1. Run the fixed projection graph to produce Q, K, V and the output gate.
2. Copy only the new K/V into views of the current parent-cache slot.
3. Run the fixed reader graph against the same parent K/V tensors, masking all
   positions outside the valid prefix.
4. Advance the valid length only after successful execution.

The reader groups four query heads under each of the two KV heads. Its matrix
operations directly consume compact FP16 storage. There is no application
whole-cache copy, double-buffer KV, KV-head expansion or whole-cache conversion.
`step` uploads the hidden input and position/length metadata, never historical
KV. Diagnostic readback is an explicit, separate operation.

`reset` only sets the valid length to zero. `truncate(length)` can shorten the
prefix; subsequent steps overwrite the discarded slots. Neither operation clears
or copies storage. Reused slots are written before becoming valid, and stale
suffixes stay masked. Out-of-capacity appends and extending truncations are
rejected before device submission. An SDK execution failure invalidates the
instance; recreate it instead of reusing potentially partial results. Dynamic
allocation and multi-sequence serving remain deferred.

## API boundary and ownership

Arithmetic uses operators listed in `demo/ref_op_api_guide.md`. The guide does
not specify an in-place subrange update or shared cross-graph ownership path.
A full-output `SCATTER_ND_UPDATE` cache plus alternating destinations would retain
another full cache and process/copy historical KV on every append. That predictable
cost motivates the following narrowly scoped extension beyond the guide:

- `vsi_nn_CreateViewTensor` creates slot views; the installed
  `vxCreateTensorFromView` documentation specifies shared parent storage and
  parent-visible writes.
- Two `vxTensorCopyNode` nodes copy producer K/V to the selected slot views.
  Graph output parameters are selected using `vxSetGraphParameterByIndex`.
  This does not use `SwapTensorHandle` or exchange the parent cache's role.
- `retain_tensor` shares an ordinary mutable tensor across SDK graph wrappers
  using an explicit OpenVX reference. The declared `AttachTensorToGraph` symbol
  remains unavailable. Each wrapper owns one reference; the reader is released
  before the copy graph/views and their parent storage, then the producer.

The installed SDK invalidates the small copy graph when its destinations change.
The implementation checks verification state and reverifies that two-node graph;
the projection and Attention reader graphs keep fixed bindings. All 1,042 appends
in the capacity test required copy-graph revalidation. This is an unresolved
per-token overhead, not a claim that the append path is compiled only once.

The ownership/view path passed the tests below. This establishes application
semantics, not physical residency, zero SDK-internal transfers or selected NPU
backends. `NP101-STATE-001` and `NP101-OBS-001` remain open.

## Precision and SDK layout

Q/K normalization uses FP32 arithmetic and returns FP16, matching the official
reference. RoPE covers the first 64 head dimensions; the other 192 remain
unchanged. Precomputed sine/cosine tables are FP16. QK scores, scaling, normalized
probabilities and PV products preserve the reference's FP16 boundaries; masking
and Softmax operate in FP32. No device result changes the frozen tolerances.

The initial rank-three Softmax produced normalization across KV groups rather
than the requested token axis. The implemented path reshapes only the small
score tensor to SDK `[512, 8]`, applies the previously validated two-dimensional
mask/Softmax configuration, then restores the grouped layout. This does not
reshape or copy KV storage. The failed attempt remains in the evidence archive.

## Run

Use Conda `SpecFerry`, the exported pack and the deployment-FP16 reference trace:

```bash
cmake -S . -B build
cmake --build build --target np101_attention_check np101_kv_cache_check -j 4
python scripts/check_np101_attention.py --storage-only \
  --output .cache/runs/kv-storage --diagnostic
python scripts/check_np101_attention.py \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/attention --steps 512 --timeout 1200 --diagnostic
```

The storage-only mode uses synthetic producer inputs without model weights. It
checks exact parent bytes, untouched slots and a fixed graph's visibility of
writes at positions 0/1/3/7/255/511, including overwrites and final-only readback.
Its cache-sized ADD output is a test observation, not a second runtime KV bank.

The Attention runner uses the official Transformers mixer and rotary embedding
with exported real weights. Captured hidden vectors cycle after eight tokens;
this is a numerical trajectory, not meaningful generated text. Allowed lengths
are 2, 8, 32 and 512. `--prepare-only` prepares CPU fixtures without opening the
device. A full suite executes five sequences: initial, truncated branch, reset,
final-only, and a fresh instance. Checkpoints include 0/1/3/7/31/255/511 when
present. Reset/fresh/final-only outputs and valid cache prefixes must additionally
match the corresponding device trajectory exactly. Masked probabilities must be
exactly zero; reference cache snapshots retain the expected stale suffix.

The existing device lock, bounded runner and recovery guard apply to both modes.
Exit 0 with `--diagnostic` means numerical/lifecycle success only. Without that
flag, such success returns 2 because hardware and residency evidence is missing.
Do not use traced test timings as performance benchmarks.

## Results, 2026-09-17

| Check | Result |
|---|---|
| Official isolated CPU mixer versus the earlier full-model trace | First eight outputs bitwise identical |
| Slot views, untouched rows, fixed reader, final-only visibility and invalid index | Pass; 16 writes, one reader verification |
| Two-token Attention suite | Pass; 107 numerical/repeat/mask checks |
| 512-token suite, including reset/truncate/recreation/final-only | Pass; 1,042 appends, 207 checks |
| Append at position 512 and extending truncation | Rejected without a write or length change |
| Resource release/process exit | Normal; no timeout, residual child or recovery requirement |
| Host tests, CTest, complete build and formatting | Pass |
| Per-node NPU execution, SDK-internal transfers and physical residency | Unverified |

For the initial 512-token trajectory, checkpoint maxima were 0.0006103515625 for
the mixer output, 0.0078125 for Q, 0.00390625 for K, 0.00146484375 for V and
0.001953125 for probabilities. All passed the existing elementwise FP16 pipeline
tolerance. There were no `Call vxBatchGemmNode fail` warnings in these Attention
runs; that absence alone is not execution-backend proof.

Local evidence, ignored by Git:

- `.cache/runs/kv-cache-20260917-views/`: initial slot-view validation.
- `.cache/runs/kv-cache-20260917-runner/`: storage-only command, including runner
  evidence and normal release.
- `.cache/runs/attention-20260917-two-step/`: failed three-dimensional Softmax attempt.
- `.cache/runs/attention-20260917-token-axis-2/`: corrected two-token suite.
- `.cache/runs/attention-20260917-capacity-512/`: full capacity suite, reference
  cross-check, binary/source hashes, fixtures, logs and driver trace.

Next integration: MLP, decoder normalization/residuals and one four-layer group.
Full resident model integration still depends on the unresolved allocation limit.
