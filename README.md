# SpecFerry

## EP baseline

The EP exploration phase is complete for the accepted configurations. Inference covers embedding,
all decoders, state/KV updates, final normalization, LM head and greedy selection on NP101;
tokenization and dispatch remain on the host. Long-term stability and peak utilization remain open.
Use the pinned v105 SDK/driver identities in each package manifest and follow the
[incident ledger](docs/np101-recovery-incidents.md) before any board run.

| Model | Recommended configuration | Accepted scope |
|---|---|---|
| OPT-350m | NBG QKV/out projections, fused M1 FFN/final norm and one NBG head; optional budgeted M16 prefill | Current NBG M1/M16: C64 and bounded C256; earlier C128 configurations remain documented separately |
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

### SoC cross-build

Use the vendor RISC-V GCC V2.6.1 toolchain and a snapshot of the board's SDK and RISC-V
`libcrypto.so.1.1`. Keep host and board build directories separate. No board internet access is
needed. `soc_root` below must contain the board paths and compatible OpenSSL development headers;
the accepted build used OpenSSL 1.1.1f public headers with a checked LP64 configuration because the
board had no development headers. Do not copy host libraries into the target snapshot.

```bash
soc_toolchain=/absolute/path/to/NP101-riscv-gcc-linux-5.10.4-glibc-x86_64-V2.6.1
soc_root=/absolute/path/to/board-root
CCACHE_DIR="$PWD/.cache/ccache" cmake -S . -B build-soc \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/np101-riscv64.cmake" \
  -DNP101_TOOLCHAIN_ROOT="$soc_toolchain" -DCMAKE_FIND_ROOT_PATH="$soc_root" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DLJMICRO_SDK_INCLUDE_DIR="$soc_root/np101_driver/build/sdk/include" \
  -DLJMICRO_SDK_LIBRARY_DIR="$soc_root/np101_driver/build/sdk/drivers" \
  -DOPENSSL_INCLUDE_DIR="$soc_root/usr/include" \
  -DOPENSSL_CRYPTO_LIBRARY="$soc_root/usr/lib/libcrypto.so.1.1"
CCACHE_DIR="$PWD/.cache/ccache" cmake --build build-soc -j 4 \
  --target specferry_opt_graph_generate specferry_opt_generate
```

Transfer `build-soc/bin/` over SSH; on the board, use
`LD_LIBRARY_PATH=/np101_driver/build/sdk/drivers` and `VIV_MGPU_AFFINITY=1:0`.
Runtime EVIS compilation needs the SDK's CL headers in a private working directory, exposed through
`LJMICRO_SDK_DIR`. Native OPT packages can use the pinned `np101-soc-riscv64` SDK profile; existing
Python exporters/runners still target EP. Small NBGs and a complete decoder passed on SoC;
full-model inference remains unaccepted. The expanded 1792MiB pool has only passed a 42MiB
allocation/readback check. See [status](docs/np101-status.md#soc部署与内存) for deployment evidence.
SoC runs require board-side deadlines, kernel monitoring and release/delayed checks; retain host EP
isolation. Driver autoload remains disabled.

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
An explicitly C256-capable package supports `--block 16 --capacity 256` with NBG projections;
capacity/reset acceptance and the mixed prefill composition are in the status document.
Packages containing QKV/out exports default to NBG; earlier head-only packages retain FCL.
Conversation reuse may recompute a partial historical block when measured headless prompt costs
favor M16; full-prefix validation precedes state changes. No padding or extra weight copies are used.
Device deadlines and recovery requirements are centralized in the incident ledger.

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
