# Validation commands

Run from the repository root after [environment setup](../README.md). Current acceptance,
precision exceptions, performance and raw evidence live in [NP101 status](../docs/np101-status.md);
[TODO](../TODO.md) owns future work. Completed state/structural probes and historical report
bodies are recoverable from `.cache/runs/cleanup-20261001/before-source.tar.gz`.

## Device discipline

Each board command runs one bounded child process and uses a new output directory. Board
checks are excluded from ordinary CTest. Timeouts, signals and lingering children leave a
recovery marker: resolve the fault before another board command; do not delete it to force
retries. Stop on kernel hang/BUG or residual ownership. Before intentionally reproducing
selection hang, cold-restart both host and NP101. Sudo needs explicit authorization.
Use the [cold-restart incident ledger](../docs/np101-recovery-incidents.md) before each board run;
after recovery, a small control and at least 45 seconds of delayed kernel postflight must pass.
CPU references and `--prepare-only` do not open NP101; model wrappers accept `--cpu-threads N`.

## 设备恢复检查 recovery controls

以固定仓库自写卷积 + 冻结旧 OPT 的数值、正常释放与各至少 45 秒清洁延迟门禁
作为放行条件，两项通过后无需原厂 demo。自写控制直接复用归档 binary，不重编译；
它有历史故障，不能据一次数值正确跳过内核检查。详见
[恢复判定约定](../docs/np101-recovery-incidents.md#恢复判定约定10-08-用户更新)。

```bash
python scripts/check_np101.py conv --sudo-read-only --output .cache/runs/repository-convolution-control
python scripts/check_np101.py recovery --sudo-read-only --output .cache/runs/model-recovery-control
python scripts/check_np101.py recovery --prepare-only --output .cache/runs/recovery-control-prepared
```

`conv` runs the pinned repository CONV2D→RELU→MAXPOOL control, checking two input iterations
and explicit graph/context release. `recovery` runs that control,
then the frozen 2026-10-01 OPT component binary: all 24 layers, C16, prompt5, two output tokens
`[78,631]`, no warmup and one request. Each control requires numerical agreement, clean exit,
clear ownership and at least 45 seconds of delayed kernel checks. The lock covers both stages
and their postflight; any failure stops the sequence. Default process deadlines are 60/180 seconds.
Kernel health is polled every five seconds during a long child run; a fault or monitor failure
terminates the child and preserves a recovery marker. Checks also recognize page faults/Oops.
These are functional controls, not throughput measurements or proof that every NP unit is healthy.
`vendor-conv` runs the unchanged vendor demo once as optional diagnosis, with the same kernel gates.
The repository control binary/source stay at the pinned archive paths; preserve all fault evidence.

`--sudo-read-only` explicitly authorizes only journalctl/fuser reads; use it only with permission.
Without it, both reads must be accessible as the invoking user. If a recovery marker exists,
confirm both host and NP101 cold restart with `--cold-restarted`; the boot must have changed,
preflight must be clean, and the old marker is archived before the control opens the device.
Binary, fixture, deployment manifest and SDK identities are pinned in `fixtures/np101-recovery-controls.json`.
Required frozen files stay at their evidence paths under `.cache/runs/validated-opt-component-baseline-20261001/`;
missing/changed files fail before device access. Do not clean those paths or substitute a rebuild.
`--prepare-only` checks identities and prints commands without reading kernel logs or opening NP101.
SDK/Verify/ioctl instrumentation is rejected for these controls.

## Generation and performance

```bash
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --compare-cpu --output .cache/runs/opt-tokens
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --benchmark --warmups 1 --repeats 3 --sdk-summary \
  --output .cache/runs/opt-benchmark
```

Graph defaults to merged-kv/group1 and token-only readback. A/B/A or a JSON request matrix via
`--request-tokens-file FILE` shares a resident model; explicit `repeat_of` identifies repeats.
Block prefill consumes 4, 8 or 16 known consecutive tokens from one request, with block1 for
tails/decode; weight/KV storage and both execution sets remain resident. It requires a shared
merged-kv variant, group1 and capacity divisible by the block size. It accepts
`--prefill-head last-token` to skip unused prompt heads. The block4 last-token policy passed
matched C16 timing/token screening and prompt4/15/4 complete-logit A/B/A acceptance.
Every-token remains the default. Block8 model execution is an unvalidated candidate; its FFN
collection and host contracts have passed. Local-const/local-mutable graph candidates use block1.
Block16 passed split2/group1/last-token/C64: a matched 64-token prefill benchmark, final logits
for prompt17/63/64 and last-two-layer KV/reset/capacity/close for prompt64/3/64. Select it with
`--block 16 --variant merged-kv-fc1-split2 --prefill-head last-token --capacity 64`.
This improves known-prompt processing; dependent single-request decode still uses block1.
OPT LayerNorm affine weights are widened from checkpoint FP16 to FP32 `[width,1]` during
initialization and shared between fixed execution sets. Dynamic normalization retains its
FP32 calculation and FP16 output boundary. Host dtype/bounds checks run in `np101_data_contract`;
device acceptance and measured performance are recorded in the status page.
Checkpoint defaults govern generation; sampling is explicit and supported by the component
FP32-logit path. Token-only success does not establish full logits/KV correctness.

Use `--cache-readback logits` for complete final logits, or
`--cache-readback kv --cache-layer-range 1 4` for K/V of one-based inclusive layers 1–4.
The range changes diagnostics only; all model layers execute. These are separate scopes,
without automatic readback expansion on failure. Full all-layer generation readback was
retired; iterate explicit KV ranges only when the acceptance scope requires it.

`--readback-transport mapped` explicitly selects public `vxMapTensorPatch`/`vxUnmapTensorPatch`
for these diagnostics. It copies on the CPU while mapped, honors row strides, and rejects virtual,
handle-backed, quantized or higher-rank tensors. Use after synchronous graph completion; mapping
failure is an error with no automatic SDK-copy fallback. Token selection keeps its existing path.
The default remains `sdk`. Mapped readback has passed the finite hardware calibration and C16
acceptance scopes below. Header semantics and static dispatch analysis alone do not establish
numerical correctness or guarantee absence of hidden DMA.
The existing projection diagnostic has an explicit `mapped-readback` mode: synchronous ADD
outputs at KV/logit sizes and rank 1, four patterned writes/map/unmap cycles per shape, exact
FP16 CPU comparison and release checks. It never uses SDK conversion for the comparison.
Run only under the recovery ledger's clean preflight, live fault monitor and delayed gate.
The recorded 2026-10-08 run passed all 16 exact outputs, C16 final-two-layer KV/lifecycle,
and C16 block4 final logits with clean delayed kernel gates. This does not validate arbitrary
shapes, C64 split-fc1, hidden SDK transfer paths or long-term stability.

Benchmark repeats each original input on the same model. Warmups are checked but excluded
from statistics; `benchmark.json` maps names to input/repeat, and samples retain actual EOS,
capacity stop, output and consumed counts. Benchmark requires token-only readback and is
incompatible with lifecycle diagnostics.

`--sdk-summary` and `--sdk-timing` are mutually exclusive. Summary aggregates steady-clock
spans by request/phase/graph-role/API in memory and writes `sdk-summary.tsv` after release;
no per-call disk flush. `nested_count` marks overlap; `completed` is normal C++ return,
with SDK status still checked by its caller. Native mode is `SPECFERRY_SDK_TIMING=summary`.
Primary `graph_metrics` sums outer RunGraph plus independent ProcessGraph, including SDK
completion waits but excluding initialization, Setup/Verify (including legacy runtime
Verify), reset, explicit IO and release. First prediction belongs to prefill; decode steps
are `consumed - prompt_length`. Existing `metrics` reports host generation wall time.
Neither is pure NPU arithmetic time. Failed checks have no accepted performance metrics.

Graph request records also contain `cpu_phases` for reset, prefill and decode, with process
and main-thread user/system CPU seconds. Prefill includes the first prediction; decode begins
only when a predicted token is consumed. CPU spans include explicit token IO and SDK dispatch,
but exclude initialization, diagnostic readback and release. CPU time overlaps graph elapsed
time and must not be added to it; whole-child CPU counters also include preparation and teardown.

## Native generation contract

Both binaries take only three paths:

```text
build/bin/specferry_opt_generate DEPLOYMENT REQUEST OUTPUT
build/bin/specferry_opt_graph_generate DEPLOYMENT REQUEST OUTPUT
```

The Python launcher prepares REQUEST and `run-options.txt`; the graph CPU token oracle is
fixed at `REQUEST/expected.txt`. Options use unique `key value` lines; unknown/duplicate
keys and invalid combinations fail before SDK access. The first row below is common to both
binaries; the remaining rows are graph-only. Component also accepts `seed` for sampling;
omitting it selects greedy. Omitted options use these defaults:

| Keys | Default / choices |
|---|---|
| `max_new_tokens`, `warmups`, `repeats` | `8`, `0`, `1` |
| `block` | `1`; graph also accepts `4`, `8`, `16` under the restrictions above |
| `variant` | `merged-kv`; candidates `merged-kv-fc1-split2`, `merged-kv-local-const`, `merged-kv-local-mutable` |
| `decoder_group_size`, `prefill_head` | `1`, `every-token`; group2 requires block1; last-token accepts every block size |
| `readback` | `none`; `logits`, or `kv` with explicit `cache_first_layer` / `cache_last_layer` |
| `readback_transport` | `sdk`; explicit `mapped` requires `logits` or `kv`, validated on the stated C16 scopes |
| `benchmark`, `lifecycle` | `0`; graph accepts `1`, mutually exclusive |

Example `run-options.txt` for a benchmark:

```text
block 4
max_new_tokens 8
benchmark 1
warmups 1
repeats 3
```

Graph layout is fixed to partitioned. Public step/close/capacity guards and payload limits
remain active; LastToken skips intermediate prompt heads while public step and decode still
produce a complete prediction. Coarse diagnostics retain numerical assertions and explicit
SDK ownership/release, rather than inferring correctness from normal exit.

## Model and lifecycle diagnostics

```bash
python scripts/check_np101_model.py prefix-split --layer-count 2 --steps 2 --capacity 16 \
  --prepare-only --output .cache/runs/prefix-prepared
python scripts/check_np101_model.py prefix-split --layer-count 2 --readback outputs \
  --fixture .cache/runs/prefix-prepared/fixture --output .cache/runs/prefix
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --request-tokens-file /absolute/path/requests.json --max-new-tokens 2 \
  --lifecycle-check --cache-readback logits --output .cache/runs/rebuild
```

`prefix-split` defaults to complete logits/selection for two teacher steps. Explicit
`outputs-kv` adds current KV, bit-exact history and zero suffix; `boundaries` with
`--boundary-layers 8 16 24` adds selected decoder outputs. A boundary list may cover all
layers when fault investigation requires it; full-intermediate
readback modes were retired. `prefix-split-repeat` rebuilds Context/model twice. Teacher
checks do not replace generation reset/isolation, capacity acceptance or hardware evidence.
Two-decoder merged-kv graphs and merged-kv graphs with local const/mutable weights are
explicit candidates, not automatically accepted configurations.
Native prefix execution takes `np101_graph_pipeline_check DEPLOYMENT FIXTURE OUTPUT`;
`FIXTURE/run-options.txt` carries `layers`, `readback` (`outputs`, `outputs-kv`, `boundaries`),
comma-separated `boundary_layers`, `variant`, `decoder_group_size`, `repeat` and `check_fixture`.
The wrapper writes these options during preparation; use its `--prepare-only` path for offline
fixture validation instead of launching a native board command.

Lifecycle now checks two model lifetimes, each with a fresh Context, complete final logits,
close rejection and zero live storage. Same-Context rebuild and the two-layer state-only
entry have been retired after diagnosis; historical failures remain in raw evidence.
Lifecycle automatically enables the pinned compilation guard, rejecting missing/zero
operation entries before RunGraph. Ordinary requests keep one resident Context/model.

`--trace-verify` enables the explicit `np101_verify_trace` diagnostic (build its CMake
target first). It records Verify stages, nodes/operation entries and execution replay /
submit / completion observations. It gates fixed v105 OpenVX/GAL hashes, rejects inherited
preloads and does not modify SDK objects. These traces are neither throughput baselines
nor sufficient numerical/hardware proof by themselves. Only targeted fault investigation
should use them; no automatic detail tracing on failure.
For guarded timing, `SPECFERRY_VERIFY_EXECUTION_TRACE=0` disables only execution logging;
Verify/node snapshots and the compilation guard remain active. Guard overhead is not calibrated.
`SPECFERRY_VERIFY_EXECUTION_TRACE=summary` instead times Process/Submit/Replay/Wait and
HAL DeviceControl spans in fixed host storage, writing `-execution-summary.tsv` at normal exit.
The diagnostic holds at most 4096 graph calls; use a bounded collection and discard interrupted
runs. Inclusive spans overlap: Wait includes DeviceControl, Submit includes Replay, and
USER_SIGNAL is a DeviceControl subset. Never add these columns together or label waits as
transfer time. Compare the same collection with execution timing off/on/off; keep the pinned
compilation guard and all other conditions identical. No SDK state or scheduling is changed.

For a bounded pre-execution weight check, explicitly set
`SPECFERRY_VERIFY_WEIGHT_PRELOAD=1` when running `specferry_opt_graph_generate` with a
shared-weight variant. It checks every bank chunk has a live consumer input alias and maps
three 32-byte FP16 regions from each selected first/middle/last decoder fc1/fc2 chunk and
the last decoder Q matrix. `weight-preload.tsv` records exact checkpoint comparisons before
the first Run; no SDK-copy fallback or whole-weight readback is used. This checks SDK storage,
not physical device/on-chip residency or hidden runtime transfers. It touches weight caches,
so never use this diagnostic request as a throughput baseline. Unset it for normal execution.

FFN collection diagnosis uses `np101_projection_check OUTPUT MODE DEPLOYMENT FIXTURE`,
with `MODE` equal to `ffn-full`, `ffn-linear`, `ffn-linear-fc1-split2`, `ffn-linear-fc1-split4`,
`ffn-linear-shared-weights` or `ffn-linear-batched`. The batched mode takes one additional
argument after FIXTURE: `4`, `8` or `16`. The fixture supplies
OPT-350m `components.txt` and 24 validated `teacher.0.layer.N.attention_norm.bin` inputs.
All retained modes use shared mutable weights; split fc1 uses the same linear builder as the
model: two or four output-channel MatMul chunks, Concat, then the unchanged bias ADD/ReLU/fc2.
Channel splitting preserves each output's full reduction and FP16 boundary; it does not batch
future tokens. Only the two-part model candidate is retained as `merged-kv-fc1-split2`;
the default remains `merged-kv`. The four-part full-model diagnostic was retired after a
small measured benefit and reduced slot margin; its source/binary/results are archived.
Check combined graph-set extension slots and shared payload before expanding a candidate;
collection results alone are not acceptance.
Each collection runs four warmup sweeps and three measured batches of eight sweeps through
24 layers. Whole-batch timing and process/main-thread user/system CPU counters exclude
preparation, explicit IO and release. `execution.json` records budgets and release.
Only `last-layer.bin` is read back: compare with `teacher.0.layer.23.bin` (full) or
`teacher.0.layer.23.mlp.bin` (linear), and the matched baseline for split fc1.
Collection readback uses the validated mapped transport, without an SDK-copy fallback.
The shared-weight mode binds layer 23's fc1/fc2 matrices and biases in all 24 graphs through
one weight bank. It preserves the original 24 inputs, graph shapes/order and invocation count;
the last graph retains the same input and weights as `ffn-linear`, so its existing oracle applies.
Compare distinct/shared/distinct collections to bound drift. `weight_sets` and actual bank payload
are recorded in `execution.json`. This changes earlier graphs' mathematical outputs for diagnosis;
it is not a model optimization, an all-layer numerical check or proof of physical weight residency.
The batched mode uses the same linear graph with input/output shapes `[hidden,M]` and
unchanged weights. Its fixture stores M contiguous FP16 rows per input file and an M-row
last-layer MLP reference. Use matched baseline/candidate/baseline collections and divide each
sweep's time by its input count to report per-input cost.
Inputs are already prepared independent FFN columns; this cannot batch dependent future decode
tokens or establish whole-model throughput. Only the final graph's M outputs are read back.
The native command does not itself check numerical agreement. Use the guarded device runner
with execution logging and SDK call timing off. Input replay omits attention/KV gaps and
cannot establish full-model throughput, pure device arithmetic time or core utilization.
Pretransposed and full-window convolution modes are retired; results remain in the status
and SDK issue documents, and their source is archived under
`.cache/runs/diagnostic-cleanup-20261008/tests/native/np101/projection_check.cpp`.

Retained Qwen3.5 regression uses the cached checkpoint:

```bash
python scripts/export_np101_dlm.py --verify-only
python scripts/reference_dlm.py --output .cache/runs/qwen-cpu
python scripts/check_np101_model.py qwen --trace .cache/runs/qwen-cpu/layer-0-3-sequential.npz \
  --layers 0 1 2 3 --steps 2 --prepare-only --output .cache/runs/qwen-prepared
python scripts/check_np101_model.py qwen --fixture .cache/runs/qwen-prepared/fixture \
  --output .cache/runs/qwen-slice
```

## Operator, cache and memory diagnostics

Use one selected case per process; [SDK issues](../docs/np101-sdk-driver-validation-report.md)
contains precise expected/actual results, crash/hang restrictions and reproduction arguments.

| Launcher | Retained cases / responsibility |
|---|---|
| `check_np101.py` | lookup, projection, sampling, fp16-preservation: atomic synthetic checks; conv/recovery: frozen recovery controls |
| `check_np101.py` | kv, cache-append, cache-packed, cache-packed64, cache-packed-block4: fixed writes, independent readers and causal attention |
| `check_np101_memory.py` | capacity (allocation only), integrity (all retained bytes), accounting (SDK counters), growth (fixed/rebound RSS), weights (exported pack readback) |
| `check_np101_model.py` | OPT/Qwen decoder regression, teacher component reference, selection/io composition and prefix integration |

Memory payload remains separately capped at 1 GiB const/nonconst. Only explicitly authorized
v106 synthetic capacity/integrity probes use `--probe-limit` up to 4096 MiB; after refusal
scan all successfully written blocks, including those after mismatches. Full retained-data
coverage is not physical board coverage. RSS/counters cannot prove leakage or transfer cost.
Packed block4 uses a calibrated fixed writable `vxReshapeTensor` alias because ordinary
RESHAPE does not guarantee writeback; see the current status API exception and evidence.

## Offline checks and outputs

```bash
python -m unittest discover -s tests -t .
ctest --test-dir build --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

Compute reports use `result.json` and `device/`; memory reports use the output directory.
`execution-evidence.json` records command, boot/binary/SDK identity, return state and resource
usage; `sdk.log` retains SDK messages. Optional calls TSV and driver ioctl tracing are
fault diagnostics. Raw historical files keep their original names/IDs; current wrappers
must not be used to claim reproduction of retired graph structures.
