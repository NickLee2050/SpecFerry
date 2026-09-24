# SpecFerry

## Environment and build

Use Linux x86_64 with the vendor's matching NP101 driver and SDK. Default SDK paths
are `/usr/inc` and `/usr/lib/ljmicro`; `/dev/galcore` must be readable and writable.
Install Conda, a C++17 compiler, CMake 3.18+, and OpenSSL development headers.

```bash
conda env create -f environment.yml
conda activate SpecFerry
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4
```

Use `conda env update -n SpecFerry -f environment.yml` to update the Python 3.12
environment. CMake generates `build/compile_commands.json` for clangd by default.
Override SDK locations with `-DLJMICRO_SDK_INCLUDE_DIR=...` and
`-DLJMICRO_SDK_LIBRARY_DIR=...` when configuring CMake.

## Download, export and CPU reference

```bash
python scripts/download_models.py --dry-run
python scripts/download_models.py
python scripts/export_opt.py
python scripts/export_opt.py --verify-only
python scripts/reference_opt.py --output .cache/runs/opt-cpu
```

Only pinned `facebook/opt-350m` is enabled. Downloads resume through the Hub client;
use `--http-only` if Xet stalls. Checkpoints live in `.cache/models/`, exports in
`.cache/np101/`. Use a fresh `--output` directory for each run.

## Generate and measure

```bash
python scripts/generate_opt.py --prompt 'The capital of France is' \
  --max-new-tokens 8 --capacity 64 --compare-cpu --output .cache/runs/opt-text

python scripts/generate_opt.py --prompt-file tests/fixtures/opt_continuation.txt \
  --prompt-tokens 32 --max-new-tokens 8 --capacity 64 --warmups 1 --repeats 3 \
  --compare-cpu --output .cache/runs/opt-benchmark
```

Output includes input/continuation text, TTFT and decode tokens/s. `result.json`
contains token IDs, timings and CPU comparison. Decode rate counts intervals after
the first output token; initialization, tokenization and warmups are excluded.
One output token has no decode throughput. Incorrect results have no accepted metrics.

Generation uses checkpoint defaults. `--sample --seed 42` explicitly enables the
validated FP32-logit sampling path; do not combine it with `--compare-cpu`.
Use `--model` and `--checkpoint` for nondefault directories. `--prepare-only`
prepares inputs and optional CPU expectations without accessing the board.

The experimental `--backend graph --block 4` runs an A/B/A token and KV check
against the CPU. It remains unvalidated on hardware and is not the default.
It requires 1..32 new tokens, capacity divisible by the block, and no sampling or warmups.

## Checks and formatting

```bash
python -m unittest discover -s tests -t .
ctest --test-dir build --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

Omit `--check` to format. [Test commands](tests/README.md) cover explicit board runs,
model slices, memory diagnostics and retained Qwen regression. [TODO](TODO.md)
contains unresolved acceptance gates.
