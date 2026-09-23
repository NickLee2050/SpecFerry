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

List the model-independent operator checks and verify an exported weight pack
without opening the board:

```bash
python scripts/check_np101_operators.py --list
python scripts/check_np101_allocation.py --deployment .cache/np101/opt-350m \
  --prepare-only --output .cache/runs/opt-weights-prepared
```

To load and read back those weights on the board, omit `--prepare-only` and use a
fresh output directory. State allocation is optional and requires `--state-spec FILE`.
Inspect `inputs.json` and `summary.json`; this check verifies weight bytes and
release, without running the model's computation graph. For the retained Qwen
operator catalog, add `--profile qwen3.5` to the operator command.

Additional run instructions:

- [Test index and shared operator checks](tests/README.md)
- [OPT decoder slices](tests/np101-opt.md)
- [Complete-model input/output and generation](tests/np101-generation.md)
- [Repeated generation, timing and hardware acceptance](tests/np101-acceptance.md)
- [Sampling](tests/np101-sampling.md)
- [Weight and FP16/FP32 capacity diagnostics](tests/np101-capacity.md)
- [Model-independent memory growth, accounting and integrity](tests/np101-memory.md)
- [SDK call timing, progress and startup latency](tests/np101-sdk-timing.md)

Check startup latency without downloading a model:

```bash
python scripts/check_np101_acceptance.py --selection-only \
  --output .cache/runs/selection-startup
```

For per-layer SDK call timing, add `--sdk-timing summary` to the full-model command
below. Progress is displayed every ten seconds. Use `--progress-interval 0` to
disable progress messages; timing is disabled by default.

To run the full-model correctness gates followed by warmups and repeated generation:

```bash
python scripts/check_np101_acceptance.py --output .cache/runs/opt-acceptance
```

For exact natural-text prefixes and readable timing/continuation reports:

```bash
python scripts/check_np101_acceptance.py --lengths 32 \
  --output .cache/runs/opt-throughput-short --diagnostic
python scripts/check_np101_acceptance.py --lengths 128 512 2048 \
  --warmups 0 --repeats 1 --cycles 1 --timeout 7200 \
  --output .cache/runs/opt-context-lengths --diagnostic
```

Read `results.md` for decode tokens/s, prefill tokens/s, TTFT and output text;
`acceptance.json` retains the raw checks. OPT's 2048-position limit leaves only one
prediction after a 2048-token prompt, so that case has no decode throughput sample.
Use `--lengths 2033` with 16 new tokens to leave room for a continuation near the limit.

Exit code 2 with `hardware_pending` means numerical
checks passed but execution/residency evidence remains open. Use `--prepare-only`
to prepare CPU fixtures without opening the board. Timing runs disable strace and
token printing; their reported latency is host wall time. See the linked guide
for prompt, repeat, capacity and SDK-path options.

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

To measure retained allocation separately from data integrity, use fresh directories:

```bash
python scripts/check_np101_capacity.py --target-mib 4096 --readback none \
  --output .cache/runs/allocation-bound
python scripts/check_np101_capacity.py --target-mib 2840 --readback all \
  --output .cache/runs/readback-map
```

Choose the scan target from the allocation result. `none` skips readback; `all`
records every corrupt block in `readback-blocks.jsonl` and returns 1 for mismatches.
Inspect `allocation_and_release_passed`, `full_scan_completed` and
`readback_and_release_passed` separately in `summary.json`. Set `--dtype F32`
or `--storage mutable` to compare precision and storage modes.

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
