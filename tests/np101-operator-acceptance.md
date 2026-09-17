# NP101 operator acceptance record

Recorded on 2026-09-15 on `fpga02`, using branch `dev/DLM-impl`, SDK 1.1.37,
and the Python 3.12 `SpecFerry` Conda environment. This record covers operator
capabilities and their test infrastructure; complete DLM deployment is not accepted.

Maintenance update, 2026-09-16: the 22 SDK RNN feedback cases (11 handle and
11 ordinary) and the temporary buffer diagnostic were removed from the active
suite. The current catalog has 49 synthetic operator cases and four optional
captured-reference projections. The results below remain the historical record
of the original 75 cases; deletion does not turn failures into passes or close
state acceptance. See the [investigation archive](state-feedback-investigation.md)
for the source, commands, and evidence needed to reproduce retired cases.

## Results

All 71 synthetic cases and four captured-reference projections were executed.
Their native fixture validation and input/expected-file hashes also passed.
Of these 75 primary cases, **64 pass numerical comparison and normal cleanup;
11 handle-based state cases fail numerical comparison**. None establishes
individual-node hardware execution or device-resident state by itself.

| Group | Result |
|---|---|
| Convolution, ReLU, pooling baseline | 10 changing-input iterations pass; maximum absolute error below 0.00383, against the demo's 0.1 tolerance; normal release |
| Small operators, including weight-storage variants | 28/30 pass; FP16 and FP32 handle feedback fail from the second execution |
| Model-shaped stateless primitives | 23/23 pass, including FP32 DeltaNet update, attention matrix products, partial RoPE, weighted RMS, projections, head tail, and full-vocabulary gather |
| Real exported weights with captured CPU inputs | 4/4 pass: DeltaNet QKV/gate, Attention Q/gate, and MLP up projection |
| Real-shaped persistent trajectories and fresh-graph replay | 9 ordinary-tensor cases pass; all 9 handle-tensor cases fail |
| Final-only readback | All ordinary cases pass and exactly match the last output of the per-step runs; handle cases remain unsupported |
| Reset versus a fresh graph | All three ordinary state kinds match exactly across the 16 post-reset outputs; handle variants do not |
| Repeated graph lifetimes | Mixed weights, dynamic FP32 matmul, ordinary recurrent state, and ordinary KV each complete 20 lifetimes with correct checked outputs and normal release |
| Cross-graph attachment | `vsi_nn_AttachTensorToGraph` remains unavailable; the symbol check exits before device initialization |
| Same-graph composition | Mixed projections, DeltaNet update, RoPE, and block selection pass without application intermediate readbacks |

The state tests use `(16,128,128)` FP32 recurrence, `(6144,4)` FP16 convolution
history, and `(512,2,256)` FP16 KV storage. They cover zero initialization,
nonzero initialization, reset after 16 updates, and the full 32-step trajectory.
KV positions include 0/1/3/7/255/511. Invalid declared indices are rejected on
the host before creating an SDK context. Head selection tests include ties,
block offsets, and masked invalid tail rows; a full multi-block LM head remains
a later integration task.

## Blocking findings

**State feedback:** ordinary tensors use SDK host buffers (1 MiB recurrent,
48 KiB convolution, and 512 KiB KV per connection). Final-only application
readback does not eliminate those SDK copies. Handle tensors report swappable
storage and no RNN host buffer, but their observed output sequence is wrong.
For the tiny FP32/FP16 test, one element should read `0.375, 0.875, 0.375`;
the observed sequence is `0.375, 0.125, 0.625`. Every SDK run and release returns
normally. The cause is not yet established.

The installed headers mention flushing handle memory around swapping. A separate
diagnostic flushed both feedback handles before the next SDK-managed swap.
The two tiny feedback cases and real-shaped FP32 recurrent case still failed.
That diagnostic's source and executable are archived; it is not an accepted fix.
The archived probe preserves the original SDK RNN path for reproduction.

One handle convolution/reset case passes its final comparison despite failing
its intermediate outputs. Acceptance must require the whole trajectory, final-only
readback, and reset-versus-fresh checks together. Matching two wrong runs is also
insufficient: both must independently match the CPU reference.

**Hardware evidence:** successful `/dev/galcore` IO is captured, but does not map
every graph node to NPU execution. The inspected `vx_ext_target.h` queries target
capabilities and offers affinity assignment; it does not establish which backend
actually executed every node. `argmax.execute_on_sw=false` is recorded without
promoting it to hardware proof. A supported profiler/trace and memory-residency
interpretation are still needed from the SDK team.

**Resources:** file descriptors rise from 4 to 5 during initialization and remain
at 5 across all 20 lifetimes. RSS settles after early growth: last samples are
approximately 40.25 MiB for mixed weights, 35.20 MiB for FP32 matmul, 19.54 MiB
for ordinary recurrent state, and 36.23 MiB for ordinary KV. No timeout or residual
child was reported in these runs. These observations do not prove absence of
SDK/device memory leaks. Timings include diagnostic tracing and reporting overhead.

**Weight capacity:** constant, mutable, and mixed-weight graphs pass changing-input
and changing-weight checks. This satisfies the small functional prerequisite for
a bounded allocation-path experiment. Pool selection, capacity above 1 GiB,
full-model allocation, and board residency remain unproven. The known full-model
capacity failure was not repeated during this validation.

Track the independent remaining gates and current acceptance requirements in
`TODO.md`: `NP101-STATE-001`, `NP101-OBS-001`, and `NP101-MEM-001`. A resident
state path still requires a numerically correct supported implementation;
historical host-buffer diagnostics cannot close that gate.

## Reproduce retained operator checks

Activate `SpecFerry`, build with CMake, and use a fresh output directory for each
invocation. Run device tests serially. The normal exit code remains 2 while
hardware evidence is missing; `--diagnostic` permits exit 0 for numerical and
lifecycle passes only.

```bash
# Small prerequisite for later allocation-path experiments.
python scripts/check_np101_operators.py \
  --case matmul_fp16_small --case weights_constant_fp16 \
  --case weights_mutable_fp16 --case weights_mixed_fp16 \
  --diagnostic --output .cache/runs/weight-storage-recheck

python scripts/check_np101_operators.py \
  --case weights_mixed_fp16 --cycles 20 --readback final \
  --diagnostic --output .cache/runs/lifecycle-recheck
```

The runner now stages `/usr/inc/CL/cl_viv_vx_ext.h` in each execution directory.
Before that fix, the runtime compiler searched the working directory and failed
to compile some kernels, even though an alternate path returned correct outputs.
The repeated small suite with the staged header has no such compilation errors.
Some FP32 matmul logs still say `Call vxBatchGemmNode fail` before another path
produces correct outputs; its hardware backend remains unverified.

## Evidence and regression checks

Local artifacts are under `.cache/runs/operator-acceptance-20260915/` (ignored by
Git). `acceptance-summary.json` and `case-results.csv` index all 75 primary cases
and preserve failed results. `persistent-final/`, `persistent-restart/`,
`lifecycle/`, `reference-projections/`, `conv-baseline/`, and `graph-sharing/`
contain the additional checks. `handle-flush-check/` retains the unsuccessful
coherency diagnostic. Each device suite snapshots its executable; fixture,
reference, weight, SDK, and staged-header hashes are retained with the reports.
Earlier pre-fix runs remain separate from the primary result set.

`environment/` and `environment-final/` capture the environment and source hashes.
The final code differs from the initial test executable in phase-cycle bookkeeping;
each result retains its own binary identity. Fresh-graph replay fixtures were added
after the initial suite and executed separately. The final source snapshot and
patch are retained with the local artifacts.

Historical validation: **40 Python unit tests**, the native CPU CTest, native validation of
all 75 primary fixtures, and all recorded fixture hashes pass. Ruff and clang-format
checks pass with the repository's 100-column rules. These host checks complement
the device results above; they do not override the remaining acceptance gates.
