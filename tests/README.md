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

## Device discipline

Read the ledger before each run. Serialize bounded processes; check boot, recovery marker,
full-boot kernel and device owner, monitor faults, and inspect at least 45 seconds after release.
Correct output, empty ownership or `recovery done` do not override faults. Preserve markers.
Recovery requires a confirmed host/NP101 cold restart and agreed controls with numerical,
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
combination has passed bounded full-model acceptance:

```sh
python scripts/generate_qwen.py --compiled-mixers /absolute/path/to/qwen-package \
  --embedding-lookup gather-nd --matmul-feed-forward-layer 9 \
  --prompt "What is 2 + 2? Answer with only the number." --max-new-tokens 8 \
  --output .cache/runs/qwen-preparation --prepare-only
```

Loading validates the full 42-NBG package, then uses the existing device MatMul FFN for zero-based
layer9; 18 DeltaNet and 23 FFN NBGs remain. No CPU FFN or `is_const` budget bypass is used.
GATHER_ND has INT32 `[1,1]` coordinates and shared FP16 weights; identical KV slots keep bindings.
Hi (13 inputs/one output) and two resident arithmetic requests (26 inputs/four outputs each,
displaying `4\n`) passed tokens, coarse hidden/recurrent/KV, unwritten slots, repeated state,
budgets/release and separate 45-second gates. Evidence: `qwen-matmul-fallback-coldboot-20261014/`.
Only SDK calls diagnostic mode was measured; C128, normal-mode performance and long-term stability
remain unvalidated. Earlier hangs retain independent ledger records and unresolved causes.

For an agreed recovery diagnostic, `--cold-restarted --sdk-timing --repeats 1 --max-new-tokens 1`
retains the previous marker and requires a different clean boot plus full acceptance before clearing
it. The user must confirm both devices restarted; flags alone do not authorize recovery controls.
The existing `reference_dlm.py`, `export_np101_dlm.py --verify-only` and model regression paths stay.

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
