# SpecFerry

## EP baseline

The EP exploration phase is complete for the accepted configurations. Inference covers embedding,
all decoders, state/KV updates, final normalization, LM head and greedy selection on NP101;
tokenization and dispatch remain on the host. Long-term stability and peak utilization remain open.
Use the pinned v105 SDK/driver identities in each package manifest and follow the
[incident ledger](docs/np101-recovery-incidents.md) before any board run.

| Model | Recommended configuration | Accepted scope |
|---|---|---|
| OPT-350m | NBG QKV/out projections, fused M1 FFN/final norm and one NBG head; optional budgeted M16 prefill | Current NBG M1/M16: C64; earlier C128 configurations remain documented separately |
| Cached Qwen3.5-0.8B | GATHER_ND, zero-based layer9 device MatMul FFN, 18 DeltaNet + 23 FFN NBGs, mapped token readback | C64 greeting, repeated arithmetic and 16-token natural generation |

The [status document](docs/np101-status.md) pins package identities, measurements and acceptance
boundaries. [TODO](TODO.md) lists optional follow-up work; it does not block this phase's closure.

## Environment and build

Use Linux x86_64 with matching NP101 driver/SDK and read/write access to `/dev/galcore`.
Install Conda, a C++17 compiler, CMake 3.18+ and OpenSSL development headers.

```bash
conda env create -f environment.yml
conda activate SpecFerry
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4
```

Update with `conda env update -n SpecFerry -f environment.yml`. SDK defaults are `/usr/inc`
and `/usr/lib/ljmicro`; override `LJMICRO_SDK_INCLUDE_DIR` / `LJMICRO_SDK_LIBRARY_DIR`
at CMake configure time. `build/compile_commands.json` supports clangd.

## Prepare and generate

Only pinned `facebook/opt-350m` is enabled for download. The Qwen baseline uses its existing cached
checkpoint. Checkpoints/exports live in `.cache/models/` and `.cache/np101/`; source-bound NBG
packages and raw evidence remain ignored. Export commands are in the [test guide](tests/README.md).

```bash
python scripts/download_models.py --dry-run
python scripts/download_models.py
python scripts/export_opt.py
python scripts/export_opt.py --verify-only
```

Use an OPT FFN package containing all QKV/out projections and the compiled head, and a complete
42-NBG Qwen package. Local
accepted package locations and manifest hashes are in the status document. These commands prepare
requests and CPU references without opening the device; use a new output directory each time.

```bash
python scripts/generate_opt.py --backend graph --compiled-ffn /absolute/path/to/opt-hybrid-package \
  --projection-backend nbg --prefill-head last-token --capacity 64 \
  --prompt 'The capital of France is' --max-new-tokens 16 --sudo-read-only \
  --output .cache/runs/opt-baseline --prepare-only
python scripts/generate_qwen.py --compiled-mixers /absolute/path/to/qwen-package \
  --capacity 64 --prompt 'Hi' --max-new-tokens 1 --repeats 1 --sudo-read-only \
  --output .cache/runs/qwen-baseline --prepare-only
```

Remove `--prepare-only` to execute after the ledger's health/recovery checks. Qwen defaults to the
accepted conservative composition above; explicit operator overrides are diagnostic configurations,
with only the scopes recorded in the status document accepted. OPT's component backend remains the
default regression baseline; the recommended optimized path is selected explicitly by the command.
The optional Qwen DeltaNet package with broadcast outer products and fused input RMSNorm is used
with `--head-backend shared-fcl --state-commit combined`. Matched C64 generation and repeated-request
reset passed; package identities, performance and export options are in the status document and
[test guide](tests/README.md). The conservative default remains available.
Checkpoint generation defaults are preserved; sampling is limited to the component backend's
validated FP32-logit path and needs explicit `--sample --seed 42`.

An OPT package with the accepted partial M16 exports also supports `--block 16 --capacity 64`.
It keeps four ordinary FFNs and twenty NBG FFNs for prefill, shares FP16 KV and one head with M1,
and retains FP32 normalization. Packages containing QKV/out exports default to NBG projections;
the earlier head-only package retains its legacy FCL path. Direct FCL experiments are stopped.
Each device child is limited to 300 seconds, including preparation; release and the required
45-second delayed kernel check are separate.

A user-confirmed NP101 board power cycle is sufficient to begin agreed recovery checks; host cold
restart or a changed host boot ID is not required. Preserve fault markers until all acceptance and
at least 45 seconds of delayed checks pass. Read-only `sudo journalctl` has standing authorization.

## Measurement and regressions

Report generated output count / generation wall separately from post-first-token decode rate.
Preparation, reset and coarse diagnostics are excluded; required token IO stays in generation wall.
`graph_metrics` excludes explicit IO and includes SDK completion waits. Input processing rate is
not generated output throughput. Failed acceptance has no accepted performance result.

CPU-only regression and formatting commands:

```bash
python -m unittest tests.python.test_generation_cli tests.python.test_compiled_validation \
  tests.python.test_recovery
ctest --test-dir build -L cpu --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

[Test commands](tests/README.md) cover exports, model regressions and explicit board diagnostics.
[SDK issues](docs/np101-sdk-driver-validation-report.md) and the incident ledger retain unresolved
contracts and recovery evidence. Do not execute archived commands with current binaries.
