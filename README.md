# SpecFerry

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

## Download, export and generate

```bash
python scripts/download_models.py --dry-run
python scripts/download_models.py
python scripts/export_opt.py
python scripts/export_opt.py --verify-only
python scripts/generate_opt.py --backend graph --block 4 --capacity 64 \
  --prompt 'The capital of France is' --max-new-tokens 8 --compare-cpu \
  --output .cache/runs/opt-text
```

Only pinned `facebook/opt-350m` is enabled. Checkpoints/exports live in `.cache/models/`
and `.cache/np101/`; use `--http-only` if Hub Xet stalls. Use a new output directory each run.
Generation follows checkpoint defaults; sampling needs explicit `--sample --seed 42`
and the component backend's validated FP32-logit path. `--prepare-only` accesses no board.

The component backend remains the default regression baseline. The graph backend uses
resident merged-kv/block1/block4 graphs with shared weights/KV and defaults to token-only readback.
OPT token decoding, block prefill and the measured C16/C64 scope are accepted with one analyzed
precision exception. Rebuilding requires releasing the last old Context before creating a new one.
See [implementation and optimization conclusions](docs/np101-status.md) for scope, measured gains
and remaining unknowns; [TODO](TODO.md) contains unfinished work only.

## Benchmark and checks

```bash
python scripts/generate_opt.py --backend graph --block 4 --capacity 16 \
  --max-new-tokens 8 --benchmark --warmups 1 --repeats 3 --sdk-summary \
  --output .cache/runs/opt-benchmark
python -m unittest discover -s tests -t .
ctest --test-dir build --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

`result.json` separates generation wall time from `graph_metrics`; graph time excludes
preparation/compilation/reset/explicit IO and includes SDK completion waits. Failed checks
have no accepted metrics. Omit formatters' `--check` for safe formatting.
[Test commands](tests/README.md) describe diagnostics, native options and retained Qwen regression.
[SDK issues](docs/np101-sdk-driver-validation-report.md) describe failing contracts;
[the incident ledger](docs/np101-recovery-incidents.md) owns recovery gates and cold-restart events.
Raw evidence and pre-consolidation documents remain under `.cache/runs/`.
