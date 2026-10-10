# Validation commands

Run from the repository root in Conda `SpecFerry` after [environment setup](../README.md).
[Implementation and conclusions](../docs/np101-status.md) owns acceptance, precision exceptions,
performance and evidence; [TODO](../TODO.md) owns unfinished work. This file describes interfaces.

## Device discipline and recovery

Read the [incident ledger](../docs/np101-recovery-incidents.md) before every board run. Serialize
bounded child processes, use a new output directory, check boot/recovery marker/kernel/ownership,
monitor kernel health during execution and inspect at least 45 seconds after release. Preserve
markers and stop on faults; numerical success or automatic recovery does not reopen the gate.
CPU references, `--prepare-only` and ordinary CTest do not open NP101.

```bash
python scripts/check_np101.py conv --sudo-read-only --output .cache/runs/repository-convolution-control
python scripts/check_np101.py recovery --sudo-read-only --output .cache/runs/model-recovery-control
python scripts/check_np101.py recovery --prepare-only --output .cache/runs/recovery-control-prepared
```

- `conv` uses the pinned archived repository CONV2D→RELU→MAXPOOL control; `recovery` runs it
  then frozen old OPT component C16/prompt5/two outputs `[78,631]`. `vendor-conv` is optional.
- Controls check numerical output, release, budget, ownership and individual delayed kernel gates;
  they use no SDK/Verify/ioctl instrumentation. Default process deadlines are 60/180 seconds.
  These are recovery checks, not throughput measurements. Do not rebuild or replace controls.
- With a marker, use `--cold-restarted` only after confirmed host and NP101 cold restart. Boot must
  change; preflight must be clean; the old marker is archived before controls open the device.
  Both controls must pass before further work. Unchanged-condition failure stops repeated attempts.
- `--sudo-read-only` authorizes journalctl/fuser reads only when the user granted permission.
  Otherwise both must be accessible as the invoking user. Long runs monitor faults every five seconds.
- Identities are pinned in [the control manifest](fixtures/np101-recovery-controls.json), including
  frozen OPT artifacts under `.cache/runs/validated-opt-component-baseline-20261001/`. Missing or
  changed identities fail before device access. `--prepare-only` verifies files without kernel/device access.

## Generation and timing

```bash
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --compare-cpu --output .cache/runs/opt-tokens
python scripts/generate_opt.py --backend graph --block 16 --capacity 64 \
  --variant merged-kv-fc1-split2 --prefill-head last-token \
  --max-new-tokens 16 --benchmark --warmups 1 --repeats 3 --sdk-summary \
  --output .cache/runs/opt-benchmark
```

Graph defaults to merged-kv, one decoder per graph, every-token head and token-only readback. Block prefill uses
4/8/16 known consecutive tokens from one request, shared resident weights/KV and block1 tails/decode.
It requires capacity divisible by block size. Block8 model execution is unvalidated.
`--prefill-head last-token` skips nonfinal prompt heads; decode/public step still predicts every step.
Checkpoint policy defaults to greedy; explicit `--sample --seed 42` uses the component FP32-logit path.
`--request-tokens-file FILE` supplies a JSON list of token-ID lists sharing one resident model.

Benchmark repeats each original input. Warmups are validated and excluded from statistics;
`benchmark.json` maps output names to input/repeat, preserving EOS/capacity stops and actual counts.
It requires token-only readback and excludes lifecycle mode. Report actual output count/generation
wall time and post-first-token decode separately, using the [defined scope](../docs/np101-status.md#性能口径与代表性结果).

`--sdk-summary` and `--sdk-timing` are mutually exclusive. Summary stores steady-clock spans by
request/phase/graph-role/API in memory, writing `sdk-summary.tsv` after release; native mode is
`SPECFERRY_SDK_TIMING=summary`. `nested_count` denotes overlap; `completed` denotes normal C++
return with SDK status still checked separately. Do not sum nested Process/Replay/Wait intervals.
`graph_metrics` excludes preparation, independent Verify, reset and explicit IO; it includes
completion waits. Generation wall includes required IO/dispatch and any runtime compilation.
`cpu_phases` records process/main-thread user/system time for reset, prefill and decode; CPU time
overlaps wall time. Failed checks provide no accepted performance metrics.

## Coarse readback and native contract

`--cache-readback logits` reads complete final logits; `kv` requires
`--cache-layer-range FIRST LAST` (inclusive, one-based). All layers still execute. Only expand
readback for a concrete failure or required acceptance scope; token agreement alone is not KV/logit validation.

`--readback-transport mapped` selects `vxMapTensorPatch`/`vxUnmapTensorPatch` after synchronous
completion, copying with returned strides. It supports ordinary, non-handle, nonquantized rank1/2
tensors and never falls back to SDK copy. `sdk` remains default; selected-token IO is unchanged.
Use only the calibrated contracts and acceptance scopes in the status page; mapping is not proof of
no hidden DMA. The projection diagnostic's `mapped-readback` mode calibrates ADD output, repeated
mapping and KV/logit-sized buffers without SDK conversion.

```text
build/bin/specferry_opt_generate DEPLOYMENT REQUEST OUTPUT
build/bin/specferry_opt_graph_generate DEPLOYMENT REQUEST OUTPUT
```

The Python launcher prepares REQUEST; native options are unique `key value` lines in `run-options.txt`.
Unknown/duplicate keys or invalid combinations fail before SDK access. Graph CPU token expectations
are in `REQUEST/expected.txt`. Defaults:

| Keys | Default / choices |
|---|---|
| `max_new_tokens`, `warmups`, `repeats` | `8`, `0`, `1`; common to both backends |
| `seed` | Component-only explicit sampling; absent means greedy |
| `block` | `1`; graph also accepts `4`, `8`, `16` under the restrictions above |
| `variant` | `merged-kv` or `merged-kv-fc1-split2` |
| `prefill_head` | `every-token` or `last-token`; both accept every block size |
| `readback` | `none`; `logits`, or `kv` with `cache_first_layer` / `cache_last_layer` |
| `readback_transport` | `sdk`; `mapped` requires logits/KV readback |
| `benchmark`, `lifecycle` | `0`; graph accepts `1`, mutually exclusive |

Graph layout is fixed to partitioned. Capacity/close guards, explicit ownership and payload limits
remain active. Shared weight preparation widens OPT LayerNorm affine parameters once at initialization;
its host dtype/bounds contract is covered by `np101_data_contract` without creating a device Context.

## Model, lifecycle and compilation diagnostics

```bash
python scripts/check_np101_model.py teacher --layer-count 24 --steps 2 --capacity 16 \
  --prepare-only --output .cache/runs/teacher-prepared
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --request-tokens-file /absolute/path/requests.json --max-new-tokens 2 \
  --lifecycle-check --cache-readback logits --output .cache/runs/rebuild
```

Graph generation owns token/logit/KV and lifecycle acceptance. `--graph-layers` still permits a
truncated reference for fault isolation. Lifecycle releases the old model and Context before rebuilding,
compares complete logits and checks close rejection/zero live storage; ordinary requests reuse one model.
Component `teacher`, `io`, `selection`, OPT decoder and Qwen regressions remain separate diagnostics.

`--trace-verify` uses `np101_verify_trace` (build that CMake target) for Verify/node/operation and
execution snapshots. It gates pinned v105 OpenVX/GAL hashes, rejects inherited preloads and never
modifies SDK objects. Lifecycle enables the compilation guard to reject missing operations before Run.
Detailed tracing is explicit fault diagnosis; no automatic expansion on failure.

- `SPECFERRY_VERIFY_EXECUTION_TRACE=0`: disable execution logging, retain Verify snapshots/guard.
- `SPECFERRY_VERIFY_EXECUTION_TRACE=summary`: fixed host storage for Process/Submit/Replay/Wait
  and HAL DeviceControl, dumped to `-execution-summary.tsv` at normal exit. Limit 4096 graph calls;
  discard interrupted collections. Wait includes DeviceControl; Submit includes Replay; USER_SIGNAL
  is a DeviceControl subset. Off/on/off comparisons retain identical guard and outer timing.
  Guard overhead itself is not calibrated; these traces alone do not prove numerical correctness.

## FFN collections and retained regression

`np101_projection_check OUTPUT MODE DEPLOYMENT FIXTURE` accepts `ffn-full`, `ffn-linear`,
`ffn-linear-fc1-split2`, `ffn-linear-batched`.
Batched mode adds `4`, `8` or `16` after FIXTURE. Fixtures contain OPT `components.txt` and 24
`teacher.0.layer.N.attention_norm.bin` inputs. Shared mutable weights and model linear builder
are reused; channel splitting preserves full reduction/FP16 boundaries before Concat/bias/ReLU/fc2.
The teacher preparation above writes first-token attention-norm/MLP references for replay;
these supplementary files do not add model readbacks. Reuse existing batched fixtures for wider inputs.

Each collection uses four warmup sweeps and three batches of eight sweeps through 24 graphs;
whole-batch time/CPU counters exclude preparation, explicit IO and release. Only `last-layer.bin`
is mapped back: compare with `teacher.0.layer.23.bin` (full), `teacher.0.layer.23.mlp.bin` (linear),
and the matched split baseline. `execution.json` records budgets/release. Native execution does
not check numerical agreement: use the guarded runner, existing oracle and kernel gates.
Each graph uses its own layer's weights. Batched fixtures contain M contiguous FP16 input/reference
rows, all final rows checked; report per-input operator cost. These collections do not establish model throughput,
all-layer correctness, physical residency or utilization. Replay omits real attention/KV gaps.

Retained Qwen3.5 regression uses its cached checkpoint:

```bash
python scripts/export_np101_dlm.py --verify-only
python scripts/reference_dlm.py --output .cache/runs/qwen-cpu
python scripts/check_np101_model.py qwen --trace .cache/runs/qwen-cpu/layer-0-3-sequential.npz \
  --layers 0 1 2 3 --steps 2 --prepare-only --output .cache/runs/qwen-prepared
python scripts/check_np101_model.py qwen --fixture .cache/runs/qwen-prepared/fixture \
  --output .cache/runs/qwen-slice
```

## Operator, cache and memory diagnostics

Select one case per process; [SDK issues](../docs/np101-sdk-driver-validation-report.md) records
expected/actual behavior and restrictions. Do not treat this table as an automatic sweep.

| Launcher | Cases / responsibility |
|---|---|
| `check_np101.py` | lookup, projection, sampling, fp16-preservation; frozen conv/recovery controls |
| `check_np101.py` | kv, cache-append, cache-packed, cache-packed64, cache-packed-block4: fixed writes/readers/causal attention |
| `check_np101_memory.py` | capacity, integrity, accounting, growth, weights |
| `check_np101_model.py` | OPT/Qwen regression, teacher reference, selection/io and prefix integration |

Const/nonconst payloads remain separately capped at 1 GiB. Only authorized v106 synthetic probes
use `--probe-limit` up to 4096 MiB; after refusal scan every successfully written block, including after
mismatches. Logical coverage, RSS and SDK counters do not establish physical coverage or leakage.

## Offline checks and outputs

```bash
python -m unittest discover -s tests -t .
ctest --test-dir build --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

Compute reports use `result.json` and `device/`; memory reports use the output directory.
`execution-evidence.json` records command, boot/binary/SDK identity, exit and resource usage;
`sdk.log` retains SDK messages. Calls TSV/ioctl tracing are optional fault diagnostics.
Preserve original evidence and names under ignored cache paths. Retired prefix diagnostics,
local-weight/group2 variants, preload sampling, split4 and shared-weight FFN experiments are archived
with the pre-cleanup sources at `.cache/runs/code-consolidation-20261010/before/`; their conclusions
remain in the status page. Removed CLI keys fail before device access; old commands require their
original archived implementation, not the current binaries.
