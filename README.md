# SpecFerry

## Prerequisites

Use Linux x86_64 for the NP101 SDK tools. Install Conda, Git, a C++17 compiler,
CMake 3.18 or newer, OpenSSL development headers/libraries, and `strace` for driver
traces. On Debian/Ubuntu, the native build packages are `build-essential`, `cmake`,
`libssl-dev` and `strace`.

Install the vendor's matching NP101 driver and user-space SDK following its
instructions. The default paths are `/usr/inc` for headers and `/usr/lib/ljmicro`
for libraries. Device commands require a host terminal that can access
`/dev/galcore` with read/write permission; ask the device administrator to configure
access if needed. CPU references and Python unit tests do not require the board.

Run the commands below from the repository root. Use a **new output directory for
each run** so that previous results are preserved.

## Create the Python environment

```bash
conda env create -f environment.yml
conda activate SpecFerry
python --version
```

The environment uses Python 3.12 and installs the pinned CPU PyTorch, reference,
download and formatting dependencies. Update an existing environment with:

```bash
conda env update -n SpecFerry -f environment.yml
```

Without shell activation, prefix commands with
`conda run --no-capture-output -n SpecFerry`.

## Build the native code

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4
```

CMake writes `build/compile_commands.json` for clangd by default. Native diagnostics
are placed in `build/tests/`; the OPT generation executable is
`build/bin/specferry_opt_generate`.

For a nondefault SDK installation, configure
`-DLJMICRO_SDK_INCLUDE_DIR=/path/to/include` and
`-DLJMICRO_SDK_LIBRARY_DIR=/path/to/lib`. Check each runner's `--help` for runtime
`--sdk-lib` and `--shader-header` overrides. The OPT generation and validation
runners currently use the default SDK paths above.

## Download and export OPT-350M

```bash
python scripts/download_models.py --dry-run
python scripts/download_models.py
python scripts/export_opt.py
python scripts/export_opt.py --verify-only
```

Only `facebook/opt-350m` is enabled for download. Checkpoint files are stored in
`.cache/models/facebook/opt-350m`; the native weight pack is written to
`.cache/np101/opt-350m`. Downloads resume interrupted transfers and preserve the
locked revision. For an existing export, use `--verify-only` or export to a new
`--output` directory.

If Xet transfers stall, retry with `--http-only`. For interrupted large HTTP files,
add `--range-workers 12`. Do not run two downloads against the same output directory.
Use `--output` to select another model-cache root, then pass the resulting model
path with `--model` to the export and reference scripts.

## Run the CPU reference

```bash
python scripts/reference_opt.py --output .cache/runs/opt-cpu
```

Inspect `reference.json` and `trace.npz` in that directory. This runs the official
CPU model and prepares the trace used by decoder-slice validation.

## Generate text on NP101

After building and verifying the OPT export:

```bash
python scripts/generate_opt.py --prompt 'The capital of France is' \
  --max-new-tokens 32 --output .cache/runs/opt-generation
```

Read the displayed continuation and `result.json` in the output directory.
`--capacity` defaults to 512 tokens; use values from 1 to 512 and a prompt that fits.
Generation follows the checkpoint's policy, which resolves to greedy decoding for
this OPT checkpoint. Add `--compare-cpu` to compare the generated token IDs with
the official CPU model. Unsupported checkpoint policies are rejected.

To explicitly request full-vocabulary sampling at temperature 1:

```bash
python scripts/generate_opt.py --prompt 'The capital of France is' \
  --sample --seed 42 --max-new-tokens 32 --output .cache/runs/opt-sampling
```

Do not combine `--sample` with `--compare-cpu`; the SDK and CPU random streams differ.
Use `--model` and `--checkpoint` when the native export or checkpoint is stored
outside the default directories.

## Run validation

Host checks:

```bash
python -m unittest discover -s tests -t . -v
ctest --test-dir build --output-on-failure
```

CTest runs only host unit tests. For decoder checks, first prepare expectations
without opening the board:

```bash
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 --steps 8 --capacity 8 \
  --output .cache/runs/opt-slice-prepared --prepare-only
```

Then run the same check on the board with a fresh output directory:

```bash
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 --steps 8 --capacity 8 \
  --output .cache/runs/opt-slice --diagnostic
```

`--diagnostic` accepts numerical/lifecycle agreement while keeping missing hardware
execution evidence visible. It does not turn numerical failures into passes.
Without this flag, these validation runners return 2 when hardware acceptance
is still pending. Read the generated report as well as the exit status.

Additional run instructions:

- [Test index and shared operator checks](tests/README.md)
- [OPT decoder slices](tests/np101-opt.md)
- [Complete-model input/output and generation](tests/np101-generation.md)
- [Sampling](tests/np101-sampling.md)
- [Weight and FP16/FP32 capacity diagnostics](tests/np101-capacity.md)

## Run the retained Qwen regression

This optional path requires the existing Qwen3.5-0.8B checkpoint and its validated
`download-manifest.json` in `.cache/models/Qwen/Qwen3.5-0.8B`.

```bash
python scripts/inspect_checkpoint.py --output .cache/runs/qwen-inventory.json
python scripts/reference_dlm.py --output .cache/runs/qwen-cpu --threads 8
python scripts/export_np101_dlm.py
python scripts/export_np101_dlm.py --verify-only
python scripts/check_np101_export.py --output .cache/runs/qwen-export-check.json
```

If the Qwen export already exists, skip its creation and run `--verify-only`.
Use the resulting
`.cache/runs/qwen-cpu/reference-trace/deployment-fp16/layer-0-3-sequential.npz`
with the [DeltaNet](tests/np101-delta-net.md), [Attention](tests/np101-attention.md)
and [decoder](tests/np101-decoder.md) validation commands. Small synthetic regression
commands are in the [component guide](tests/np101-components.md).

## Inspect device failures

Record the environment before a device investigation:

```bash
python scripts/collect_environment.py --output .cache/runs/environment
```

Inspect each run's report, SDK log and execution evidence; `strace` output is
included when available. Current driver-specific reproduction details are in the
[capacity diagnostic guide](tests/np101-capacity.md).

If a run times out, exits on a signal or leaves a child process, further device
runs are blocked by `.cache/runs/np101-recovery-required.json`. Have the device
maintainer confirm recovery before retrying. After a host reboot, the next runner
recognizes the new boot identity. For recovery without reboot, archive the marker
only after the maintainer confirms both device health and process termination.
Do not remove it merely to force another run.

## Format the code

```bash
python scripts/format_cpp.py
python scripts/format_python.py
python scripts/format_cpp.py --check
python scripts/format_python.py --check
```

C++ uses LLVM style with a 100-column limit; Python uses Ruff with the same line
width, safe lint fixes and sorted imports. Both tools are pinned in the Conda
environment. Use `--check` for verification without editing files.
