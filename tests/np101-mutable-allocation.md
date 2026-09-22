# All-mutable weight-loading experiment

Executed on 2026-09-17 on `fpga02` for the exported Qwen3.5-0.8B text weights.
Result: **setting all weights to `is_const=false` did not allow complete loading.**

## Method

The existing allocation diagnostic now accepts `--weight-storage constant|mutable`
(default: constant). Mutable mode uses `vsi_nn_AddTensor` with no initial data,
then `vsi_nn_CopyDataToTensor`, following the vendor guide's mutable input path.
Both modes keep the original order and chunking: at most 4,096 rows and 8 MiB
per weight tensor. The model contains 320 logical weights, split into 416 SDK
tensor chunks, with 1,504,791,232 payload bytes. The tied embedding/head is counted
once. All created weight tensors remain owned by one graph until cleanup.

After loading weights, the test attempts the existing 86 mutable state/RoPE
buffers (46,071,808 bytes), then reads every weight chunk back and compares it
byte for byte with the export while all tensors remain alive. Graph execution,
packing/workspace, activations, and smaller position/mask/head auxiliary buffers
are outside this allocation test. Passing it would not establish board residency
or working inference.

## Results

The constant baseline below is the recorded 2026-09-09 run; it was not rerun.

| Observation | Constant baseline | All-mutable experiment |
|---|---|---|
| Successfully initialized/uploaded chunks | 266 | 266 |
| Successfully initialized/uploaded payload | 1,070,874,144 bytes | 1,070,874,144 bytes |
| Next weight | `model.layers.20.mlp.down_proj.weight` | Same |
| Next chunk size | 7,340,032 bytes (7 MiB) | Same |
| Failure point | Tensor creation with initialization | Explicit upload after tensor creation |
| Exit | Code 1, normal process exit | Code 1, normal process exit |

Mutable mode returned `CopyDataToTensor failed: SDK status=-5`. The installed
`/usr/inc/VX/vx_types.h` names -5 `VX_ERROR_NOT_ALLOCATED`; it is not the separate
`VX_ERROR_NO_MEMORY` status (-8). The identical byte boundary is evidence of a
capacity-related limitation on this path, but does not identify the physical
pool or prove the driver's root cause.

The mutable report counts 267 created tensor objects and 1,078,214,176 bytes of
declared tensor sizes, including the last object whose upload failed. These are
not successfully loaded bytes or measured physical allocations. Only
`uploaded_weight_bytes=1,070,874,144` completed. Of 320 logical weights, 184
completed; state allocation and full retained-weight readback were not reached.
Peak host RSS was 39,096,320 bytes. No timeout, recovery marker, or residual SDK
process was observed.

A bounded follow-up used two actual weight vectors (32 bytes FP16 and 64 bytes
FP32), plus all 86 state/RoPE buffers. All 88 tensors coexisted, both weight
vectors matched their exported bytes after readback, and release completed.
This validates the new readback/cleanup path at small capacity; it does not
change the failed full-model result. The 39 Python unit tests, build, and Ruff/
clang-format checks also passed.

## Evidence and reproduction

Local artifacts (ignored by Git):

- `.cache/runs/weight-allocation-first/`: original constant allocation attempt.
- `.cache/runs/weight-allocation-after-reboot-20260909/`: cold-restart constant
  recheck, comparison, driver parameters, SDK output and trace.
- `.cache/runs/weight-allocation-mutable-20260917/`: allocation report, SDK log,
  driver trace, executable snapshot, source snapshot/hashes, and process check.
- `.cache/runs/weight-allocation-mutable-20260917-environment/`: SDK/driver facts.
- `.cache/runs/weight-allocation-mutable-small-20260917/`: bounded readback control.

For an authorized recheck after a relevant change, use a fresh output directory:

```bash
conda activate SpecFerry
cmake --build build --target np101_weight_allocation_check -j 4
python scripts/check_np101_allocation.py --weight-storage mutable --timeout 300 \
  --output .cache/runs/weight-allocation-mutable-recheck
```

`NP101-MEM-001` remains open. A concrete change or vendor guidance is needed before
another full-capacity retry; toggling `is_const` alone has now been tested.
