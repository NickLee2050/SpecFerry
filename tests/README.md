# Validation

Run from the repository root in Conda `SpecFerry`, after [setup](../README.md).
[Status](../docs/np101-status.md) owns accepted configurations, precision and performance;
[TODO](../TODO.md) owns unfinished work; [incidents](../docs/np101-recovery-incidents.md) owns
recovery gates and individual cold-restart evidence. Historical programs are not current commands.

## Offline checks

```sh
python -m unittest discover -s tests -t .
ctest --test-dir build -L cpu --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

Python covers checkpoint/export identities, generation policy/counts, arrays, budgets, KV/lifecycle
and recovery guards with mocked device calls. All five native CTest cases have `unit;cpu` labels
and never open NP101. Device diagnostics are explicit executables, never automatic CTest runs.

## EP baseline maintenance

EP exploration has concluded for the configurations pinned in the status document. Recommended
commands are in the main README; keep the component/operator/compiled regressions needed to
maintain these baselines. New acceptance runs require a concrete changed contract or observed
failure; do not repeat historical matrices as a release checklist. Hardware validation from 10-15
remains immutable; finalization changes entry defaults and records, with offline checks only.

## Device discipline

Read the ledger before each run. Serialize bounded processes; check boot, recovery marker,
full-boot kernel and device owner, monitor faults, and inspect at least 45 seconds after release.
Correct output, empty ownership or `recovery done` do not override faults. Preserve markers.
Recovery requires a user-confirmed NP101 board power cycle and agreed controls with numerical,
execution, budget, release and delayed-kernel acceptance; passing covers only that workload.
Never restore the deleted repository convolution or silently substitute vendor/old OPT controls.
`vendor-conv` remains an explicitly authorized diagnostic and cannot clear the recovery gate.

`--prepare-only` never opens the device. Read-only `sudo journalctl` has standing authorization;
`--sudo-read-only` enables it in the runner. Device-owner checks use unprivileged `fuser`.
Other sudo actions require approval. Const/nonconst payloads each remain ≤1 GiB, including peaks;
only authorized v106 synthetic capacity/integrity probes may use `--probe-limit` up to 4096 MiB.
After allocation refusal, scan all written blocks; logical offsets are not physical addresses.

## OPT generation

```sh
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --compare-cpu --output .cache/runs/opt-tokens
python scripts/generate_opt.py --backend graph --block 16 --capacity 64 \
  --variant merged-kv-fc1-split2 --prefill-head last-token \
  --max-new-tokens 16 --benchmark --warmups 1 --repeats 3 \
  --output .cache/runs/opt-benchmark
```

Defaults: merged KV, block1, every-token head, greedy, token-only readback. Blocks4/8/16 consume
known consecutive input from one request; tails/decode use block1 and capacity must divide by block.
Block8 model execution is unvalidated. `last-token` skips unused prompt heads while updating every
input's decoders/KV. `--request-tokens-file` supplies a JSON list for one resident model.
Sampling needs explicit `--sample --seed`; unsupported checkpoint settings fail.

Benchmark excludes validated warmups and maps measured files to input/repeat in `benchmark.json`.
Report actual generated count / generation wall separately from post-first-token decode; do not mix
input lengths. Preparation, reset and coarse diagnostics are excluded; required token IO is included.
CPU user/system overlap wall and cannot be added. `--cache-readback logits|kv` is explicit;
KV uses one-based inclusive `--cache-layer-range FIRST LAST`, with every layer still executing.
`--readback-transport mapped` uses calibrated ordinary rank1/2/3 Map/Unmap without SDK fallback.
Lifecycle mode releases the final old Context before rebuilding; it is separate from benchmarking.

Native generation takes `DEPLOYMENT REQUEST OUTPUT`; the Python launcher writes unique `key value`
settings in `run-options.txt`. Unknown, duplicate and incompatible options fail before SDK access.

## Source-bound compiled exports

The installed compiler runs in a device-free rootfs. Runtime packages contain NBGs and source/SDK/
budget/port manifests; ONNX, generated C and compiler logs stay outside the runtime package.

| Exporter | Maintained output |
|---|---|
| `export_np101_opt_ffn.py` | All 24 M1 FFN/residual/final-norm graphs, FP16 boundaries and FP32 normalization; per-layer CPU and generated-C audits |
| `export_np101_opt_projections.py` | All 24 independent QKV/out graphs plus one shared head; public and SDK port permutations |
| `export_np101_opt_model.py` | `--scope decoders|fullgraph --capacity 64|128`, FP32 KV, typed ports, optional `--package-output` |
| `export_np101_qwen_blocks.py` | One layer or `--all-layers`; `--part mixer|feed-forward`, optional reused first layer and runtime package |

```sh
PYTHONPATH=python python scripts/export_np101_opt_ffn.py \
  --checkpoint .cache/models/facebook/opt-350m --deployment .cache/np101/opt-350m \
  --rootfs /absolute/path/to/compiler/rootfs --output /absolute/path/to/ffn-package
python scripts/generate_opt.py --backend graph --compiled-ffn /absolute/path/to/ffn-package \
  --prefill-head last-token --prompt "The capital of France is" --max-new-tokens 16 \
  --output .cache/runs/optimized-opt-request --prepare-only
```

FFN `--import-artifacts` rechecks existing exports. `--head-artifacts` and `--projection-artifacts`
package the optional head and per-layer projections. `--projection-backend fcl|nbg-qkv|nbg-output|nbg`
selects projections explicitly; default within that package is FCL. Source hashes, generated FP32
arithmetic, FP16 ports, graph versions and budget bounds are checked before Context creation.
Direct FCL binds input/static weight without fused bias; bias stays a separate FP16 Add.

`generate_opt.py --backend graph --compiled-model PACKAGE` loads fullgraph/decoder packages.
The accepted fullgraph exporter repairs ArgMax to INT32 and runs the generated simulator before
re-saving. Verify-only re-saving was import-rejected. Re-saved graphs use public IO as their new
kernel order, which differs from the original NBG's node order. Metadata alone cannot prove dtype;
check generated C. Generated node C loads `network.export.data`, not the imported `model.data`.

Compiled OPT accepts the documented C64/C128/M1 scopes, bounded resident benchmarks and mapped
coarse diagnostics. Follow the current ledger before executing; no implicit recovery, lifecycle
rebuild or tracing override. Fixed acceptance/performance evidence lives in the status document.

## Pinned compiled acceptance

```sh
python scripts/check_np101_compiled.py --plan /absolute/path/plan.json \
  --case fullgraph --prepare-only --output .cache/runs/compiled-check
```

Schema version1 plans pin SDK/module/binary, named file sets, request/reference, kind, repetitions,
stop reason and separate payload bounds. Paths are absolute or repository-relative; `{output}` is
the child directory. Numerical references are independent and hashed. KV contracts state F16/F32,
shape and reference paths. `device_reset` checks coarse reset evidence and zero reset uploads.
The runner holds the lock through monitoring/delayed checks and reports performance only on pass.
`--audit-verify` requires pinned v105, disables execution tracing and checks complete outer
operations; it does not enumerate NBG internals. Legacy fixed binaries/plans remain immutable
under ignored caches. Formal generation replaces the retired projection-specific model runner.

## Cached Qwen generation

No new downloads are enabled. Export `--part feed-forward --all-layers --mixers-package PACKAGE`
combines 24 FFNs with the source-bound 18-mixer package. The explicit C64/v105 conservative
combination has passed bounded full-model acceptance and is now the generation entry default:

```sh
python scripts/generate_qwen.py --compiled-mixers /absolute/path/to/qwen-package \
  --prompt "What is 2 + 2? Answer with only the number." --max-new-tokens 8 \
  --output .cache/runs/qwen-preparation --prepare-only
```

Loading validates the full 42-NBG package, then uses the existing device MatMul FFN for zero-based
layer9; 18 DeltaNet and 23 FFN NBGs remain. No CPU FFN or `is_const` budget bypass is used.
GATHER_ND has INT32 `[1,1]` coordinates and shared FP16 weights; identical KV slots keep bindings.
Hi (13 inputs/one output) and two resident arithmetic requests (26 inputs/four outputs each,
displaying `4\n`) passed tokens, coarse hidden/recurrent/KV, unwritten slots, repeated state,
budgets/release and separate 45-second gates. Evidence: `qwen-matmul-fallback-coldboot-20261014/`.
A diagnostic-disabled comparison later hit a 4-byte EDMA timeout. After the user-confirmed board
power cycle on 10-15, the explicit `--token-readback mapped` alternative passed greeting, matched
SDK calls off/on/off arithmetic (two requests each) and natural generation (23 inputs/16 outputs):
5 processes/8 requests, coarse state/unwritten slots, repeat identity where applicable,
budget/release and separate 45-second gates. No SDK-copy fallback is introduced;
no new kernel faults occurred, and the previous marker was archived. Finalized Python/native defaults
select GATHER_ND, layer9 MatMul FFN and mapped token readback; explicit overrides remain diagnostic.
The complete generation entry requires a valid fallback layer; the request API rejects `None`.
Fault avoidance over longer runs, the original EDMA cause and Qwen C128 remain unresolved.

Evidence and accepted timings: `qwen-mapped-recovery-20261015/summary.json`. Off-case pooled decode
is 2942.13 ms/token; actual arithmetic output is 0.07707 token/s, and natural generation is
0.19347 output token/s (0.33876 post-first-token decode). Preparation/reset/coarse IO are excluded.
SDK calls recording added 4.22% generation wall and 3.48% decode time in the matched bounded
comparison; Verify auditing stayed enabled in every case. Full-boot logs retain multiline fault
timestamps; trace duration alone does not establish internal movement or utilization.

For an agreed recovery diagnostic, `--board-power-cycled --sdk-timing --repeats 1 --max-new-tokens 1`
retains the previous marker until full acceptance. User confirmation of an NP101 board power cycle
is required; host cold restart and a changed boot ID are not. Same-boot recovery acknowledges only
previously recorded timestamped faults, retaining the complete log and rejecting new occurrences.
The accepted fault baseline is published only after numerical/release/45-second checks pass; flags
alone do not authorize recovery controls.
The existing `reference_dlm.py`, `export_np101_dlm.py --verify-only` and model regression paths stay.

Optional vocabulary-prefix replacement preserves the default package and whole-vocabulary greedy:

```sh
python scripts/export_np101_qwen_blocks.py --part vocabulary-head --vocabulary-rows 65536 \
  --deployment .cache/np101/Qwen3.5-0.8B --rootfs /absolute/path/to/compiler-rootfs \
  --output .cache/runs/qwen-head-export --package-output .cache/np101/qwen-head-prefix
python scripts/generate_qwen.py --compiled-mixers /absolute/path/to/qwen-package \
  --compiled-head .cache/np101/qwen-head-prefix \
  --prompt "Write one friendly sentence inviting a neighbor to dinner tomorrow." \
  --max-new-tokens 16 --output .cache/runs/qwen-head-generation --prepare-only
```

This M1 head replaces only the first 65,536/248,320 vocabulary rows with 16 FCLs in one NBG;
the other 45 row blocks retain shared MatMul weights. Export uses the saved public NBG port order.
Native/Python loaders bind source hashes, ports, precision, graph version, SDK and constant budget.
Matched C64/v105 original/candidate/original requests passed CPU tokens, coarse state, release
and delayed gates: 0.195256 to 0.206645 actual output token/s; 0.340841 to 0.374409 decode token/s.
Extra const is 134,219,128 B, with measured const/nonconst peaks 1,021,362,671 / 669,684,140 B.
Evidence: `qwen-head-prefix-20261015/summary.json`. Full vocabulary replacement, C128 and longer
stability were not covered by that prefix experiment. The initial wrong-port numerical failure is
excluded from performance.

Full shared-weight FCL replacement is a separate explicit option:

```sh
python scripts/generate_qwen.py --compiled-mixers /absolute/path/to/qwen-package \
  --head-backend shared-fcl \
  --prompt "Write one friendly sentence inviting a neighbor to dinner tomorrow." \
  --max-new-tokens 16 --output .cache/runs/qwen-shared-head-generation --prepare-only
```

It retains the existing 61 embedding weight blocks for all 248,320 vocabulary rows, without
changing `is_const` or importing an embedded-weight head NBG. It cannot combine with `--compiled-head`;
the default remains `matmul`. Matched C64/v105 original/candidate/original requests passed CPU
tokens, coarse state/unwritten slots, budget/release, compilation and separate 45-second gates.
Actual output rose from 0.195944 to 0.249173 token/s; post-first-token decode from 0.341641 to
0.536712 token/s. Const/nonconst peaks stayed 887,143,543 / 669,684,140 B, including temporary
backing; SDK-internal packing/copies remain unknown. Evidence: `qwen-shared-head-20261015/summary.json`.
No new device faults occurred. This does not validate C128 or longer stability.

`--sdk-timing` records existing SDK calls and per-layer progress. The shared-head attribution
used matched calls off/on/off (reusing the accepted off-before request): the on case was
1.57%/1.53% slower in generation/decode than the off mean, with complete call-order coverage.
DeltaNet NBG execution was 32.40% of traced
decode wall, or 37.60% with its commit graphs. These spans include completion waits and internal
movement, not individual hardware kernels. Evidence: `qwen-shared-head-attribution-20261015/summary.json`.

DeltaNet export accepts independent `--outer-product multiply` and `--fuse-input-norm`
options with `--part mixer`; `--all-layers` exports all 18 mixers. The former replaces only the
FP32 rank-one state update. The latter consumes the raw decoder input, preserving FP32 RMSNorm,
the weight offset and the FP16 projection boundary. Its version-2 native manifest binds the
additional norm weight; version-1 packages keep the external norm. Combine mixer packages with
the accepted FFN package through `specferry.models.qwen3_5.compiled.combine_packages`.

Generation also accepts `--state-commit combined` (default: `per-layer`). FFNs then retain their
DeltaNet NBG activation directly, and one fixed graph copies all 18 recurrent/convolution pairs
after the decoder stack. New/old states remain distinct; the next token, reset and state readback
require a completed commit. Explicit close releases consumers before state owners. Keep the
full shared FCL head option when comparing these candidates; accepted scope and performance
are recorded in the implementation summary.

## Targeted diagnostics and evidence

| Entry | Retained purpose |
|---|---|
| `check_np101.py` | lookup, MatMul+Add, mapped readback, sampling, FP16 preservation; cache writes/sharing/dependencies; explicit vendor diagnostic |
| `check_np101_memory.py` | capacity/integrity/accounting/growth/weight questions; capacity is not integrity |
| `check_np101_model.py` | OPT/Qwen teacher/decoder regressions, selection and IO isolation |

Completed FFN split/batch collections and projection replacement programs are retired; their
sources, binary identities and measurements remain under ignored caches. Necessary regressions
use the maintained model/compiled entries. Never run archived commands with current binaries.

`--sdk-summary` aggregates in memory; `--sdk-timing` records individual calls, mutually exclusive.
Profiler is explicit and separate; use off/on/off and the status document's perturbation limits.
Outputs retain result.json, command/boot/binary/SDK identity, SDK log, measured generations,
budgets/release and full health snapshots. Raw checkpoints/NBGs and immutable incident evidence
stay ignored; `commit-preparation-20261014/before-manifest.json` pins this cleanup's retired sources.
