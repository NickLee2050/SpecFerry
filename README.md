# SpecFerry
A testbed for speculative decoding across heterogeneous edge devices and a local inference hub.

The initial deployment target is full text inference of `Qwen/Qwen3.5-0.8B`
on NP101 as the draft language model (DLM). The repository currently provides
environment capture, checkpoint download, CPU references, operator capability probes, streamed text-weight export, and native SDK
validation tests. Full DLM inference on NP101 is not yet implemented. Project constraints are in
[AGENTS.md](AGENTS.md).

Full-model resident weight allocation is currently blocked by an effective limit
near 1 GiB in the tested constant-tensor path. The NP101 team is evaluating the
allocation mechanism. [Engineering follow-up](TODO.md) records the evidence,
closure criteria, dependencies, and isolated component work that can continue.

## Set up the Conda environment

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate SpecFerry
python --version
```

The shared `SpecFerry` environment uses Python 3.12 for downloads, CPU references,
and tests. Conda manages the interpreter; the environment's pip installs the
pinned Python dependencies, including CPU-only PyTorch. Model weights and run
records stay in `.cache/`; project source stays in `scripts/` and `python/`.

To apply later dependency changes, run `conda env update -n SpecFerry -f environment.yml`.
For a shell without activation, prefix a command with
`conda run --no-capture-output -n SpecFerry`, for example:

```bash
conda run --no-capture-output -n SpecFerry python scripts/download_models.py --dry-run
```

## Download the Qwen3.5 draft model

```bash
conda activate SpecFerry
python scripts/download_models.py --dry-run
python scripts/download_models.py
```

Only Qwen3.5-0.8B is enabled. The script downloads the original checkpoint and
tokenizer under `.cache/models/Qwen/Qwen3.5-0.8B`, pins the revision, resumes
interrupted transfers, and writes `download-manifest.json` only after validating
the files. This does not convert or deploy the model. The original checkpoint
also contains vision/MTP tensors; the NP101 text exporter must select the text
model tensors later.

Future target language model (TLM) entries are commented out in `scripts/download_models.py`. After
enabling a required entry, `--parallel-models 2 --file-workers 4` allows two model
downloads concurrently with up to four file workers each. A single weight shard
does not become multiple shards through this option. Use `--output` to choose a
different storage root. Re-running a download retains its locked revision; use a
new output directory to intentionally select a different revision.

If the Xet transfer service stalls, stop that invocation and re-run with
`--http-only` to use the Hub's HTTP transfer path. Do not run two copies against
the same output directory simultaneously.

For repeated single-file HTTP interruptions, `--http-only --range-workers 12`
downloads the public weight file in bounded, resumable byte ranges. Each response
must match its requested range; the assembled file must match the Hub SHA256
before it is published. Range fragments remain under the model's `.cache/`.

The download script works on both Windows and Linux. The current full reference
environment is validated on Linux x86_64; the native NP101 SDK is required to
build and run the native SDK validation tests.

## Record the host and device environment

Run from the repository root, in a session that can see `/dev/galcore`:

```bash
python scripts/collect_environment.py --output .cache/runs/local
```

`--sdk-include` and `--sdk-lib` override `/usr/inc` and `/usr/lib/ljmicro`.
`environment.json` records toolchain versions, library fingerprints, PCI driver
binding and memory-pool parameters. `hardware-evidence.md` states the limits of
this evidence. A sandbox that hides device nodes cannot validate board access.
No driver or system configuration is changed by these tools.

## Run CPU reference inference and validation

The model download must finish and write its verified `download-manifest.json`
before running these commands.
Use the same activated `SpecFerry` Conda environment for the CPU reference.
CUDA and FLA are not required.

```bash
conda activate SpecFerry
python scripts/inspect_checkpoint.py --output .cache/runs/local/checkpoint-inventory.json
python scripts/reference_dlm.py --output .cache/runs/local --threads 8 --max-new-tokens 32
```

The loader checks the entire checkpoint and strictly maps all text weights,
excludes vision/MTP, and verifies the shared embedding/head. Both reference modes
run entirely on the host CPU. `original-fp32` expands the original BF16 weight
values losslessly to FP32. `deployment-fp16` uses FP16 projections and explicit
FP32 sensitive operations and recurrent states; native FP32 parameters stay FP32.
`reference-policy.json` records each boundary and the exact reference source hash.

Outputs include Chinese/English greedy generation, token IDs, step logits, cache
shapes, prefill-versus-sequential comparisons at 1/2/4/8 tokens, and actual layer
0/3 intermediates in `reference-trace/`. All model loading is local and offline.
Check `reference-summary.json` and `tolerances.json`; a failed numeric comparison
returns nonzero and does not silently widen the thresholds. These checks do not
prove NPU support, full DLM deployment, or general model quality.

## Build and validate convolution, ReLU, and pooling

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4
python scripts/check_np101_conv_relu_pool.py --output .cache/runs/conv-relu-pool --repeats 10
```

CMake generates `build/compile_commands.json` by default during configuration,
providing clangd with the compiler flags and include paths for code completion
and diagnostics.

Configure `LJMICRO_SDK_INCLUDE_DIR` and `LJMICRO_SDK_LIBRARY_DIR` through CMake
when the SDK is elsewhere; also pass `--sdk-lib` to the runner. This builds
host C++17 code from `native/np101/`; the supplied `demo/` stays untouched.
The test uses the demo's FP16 NCHW CONV2D→RELU→POOL layout and CPU tolerance
0.1, checks allocations and NaN/Inf, reuses one graph while varying inputs, and
checks that release clears both handles. Release checks do not establish absence
of every internal SDK allocation leak.

`conv-relu-pool.json` contains numeric results and phase timing; `conv-relu-pool.log`,
`conv-relu-pool.strace`, and `conv-relu-pool-evidence.json` contain SDK output, successful driver
calls, the new binary hash and resolved runtime paths. The runner has a 1200 s
timeout (`--timeout` overrides it). A numeric pass is separate from proof of
individual node execution on NPU. Generic driver IO and a target label alone
do not meet the latter requirement; review the execution evidence before
concluding that the tested operations ran on the NPU.

## Check DLM operator and state capabilities

Native diagnostics live in `tests/native/np101/`; reusable SDK code lives in
`native/np101/`. CMake places diagnostic executables under `build/tests/` and
requires the OpenSSL Crypto development library for weight integrity checks.

```bash
python scripts/check_np101_operators.py --list
python scripts/check_np101_operators.py --output .cache/runs/operators-small
python scripts/check_np101_operators.py --scale model --output .cache/runs/operators-model
python scripts/check_np101_graph_sharing.py --output .cache/runs/graph-sharing
```

Cases cover FP16 projections, dynamic FP32 matrix inputs, normalization, activations,
embedding lookup, short convolution, masking, cache writes, layouts, conversions,
argmax ties, and state feedback/reset. Every graph receives changing inputs across
at least two executions. Use repeatable `--case NAME` options to isolate a failure;
`--prepare-only` writes fixtures without opening the device.

Each case retains its readable `graph.txt`, exact input/expected bytes, tensor shapes,
source hashes, SDK execution report, logs, and driver trace. The suite snapshots its
executable and records pending cases. It writes `op-capabilities.json` incrementally.
The default 300-second timeout applies to each process, including SDK teardown.
Hardware diagnostics are serialized through a repository-local lock.

Fixed-input errors use the frozen reference tolerances. Exit code 2 means the
capability gate is blocked, including when individual node execution is unverified.
`--diagnostic` permits exit 0 for numeric passes while retaining the hardware blocker.
Header availability, graph verification, numeric agreement, and device residency
are distinct evidence. Copy-backed RNN state and an explicitly selected software
argmax path are reported as blockers. Cross-graph attachment is checked as an
optional SDK symbol because some library builds declare it without exporting it.

## Export text weights and check memory allocation

```bash
python scripts/export_np101_dlm.py --output .cache/np101/Qwen3.5-0.8B
python scripts/export_np101_dlm.py --verify-only
python scripts/check_np101_export.py --output .cache/runs/export/torch-verification.json
python scripts/check_np101_allocation.py --output .cache/runs/weight-allocation
```

The exporter validates the fixed checkpoint and exact 320-tensor text contract,
streams BF16-to-FP16 conversion, preserves native FP32 tensors, and excludes vision
and MTP. It publishes the output directory only after SHA256 checks. The package
contains `weights.bin`, `weights.index`, `deployment-manifest.json`, and
`memory-budget.json`. Tensor offsets are 64-byte aligned; each tensor records its
source dtype, shape, conversion, digest, and SDK axis order.

Weights retain row-major source axes. Linear projections use `X @ W.T`;
normalization weights are not folded. `lm_head.weight` aliases the embedding table,
and row-block descriptors avoid creating a second head copy. SDK-specific layouts
still require validation before model integration.

The memory report includes weights, double-buffered FP32 recurrent states and short
convolution caches, compact KV, and auxiliary buffers. `--pool-bytes` records an
observed pool size, not free memory. SDK packing/copies and activation/workspace
costs remain unknown unless supplied as explicit estimates with
`--sdk-overhead-bytes` and `--workspace-bytes`. No estimate establishes memory fit.

The allocation diagnostic verifies the native pack, uploads bounded weight blocks,
and keeps all weight and state tensors alive together. It records allocation failures
and host peak RSS. It does not include full model graphs or their workspace, and
an SDK allocation alone does not prove physical board residency.

The independent export check compares all 320 tensors, byte for byte, with PyTorch's
conversion of the original checkpoint. It uses bounded chunks and no NPU.

If an SDK run times out, exits on a signal, or leaves a child running, the runner
records `.cache/runs/np101-recovery-required.json` and stops further hardware
diagnostics. Process exit alone does not establish driver recovery. A host reboot
changes the recorded boot identity and clears the block on the next invocation.
For recovery without reboot, the board maintainer must confirm that the device is
healthy and the recorded processes have exited before the marker is archived.
The runner never reloads drivers or restarts the host. Inspect processes from the
same host PID namespace, since a sandbox can hide a still-running device process.

After exporting, test captured reference inputs against actual exported weights:

```bash
python scripts/check_np101_operators.py \
  --reference-trace .cache/runs/local/reference-trace/deployment-fp16/layer-0-3-sequential.npz \
  --case reference_delta_qkv --case reference_attention_q_gate \
  --output .cache/runs/operators-reference
```

Use the trace from an existing successful CPU reference run. These cases compare
against captured module outputs and record both trace and weight hashes.

## Format C++ code

The repository's `.clang-format` uses LLVM style with a 100-column line limit
and a blank line between function definitions. Control statements (`if`, `else`,
`for`, `while`, and `do`) require braces even for a single-statement body;
clang-format inserts missing braces automatically.
The Conda environment includes clang-format 18.1.8 for consistent formatting.

```bash
conda activate SpecFerry
python scripts/format_cpp.py
python scripts/format_cpp.py --check
```

The script formats C++ sources and headers listed by Git, including new,
untracked files while respecting Git ignore rules. `--check` reports formatting
differences and returns nonzero without modifying files. Use `--clang-format`
to select an executable at a custom path. The script works from any directory.

## Format and lint Python code

The repository uses [Ruff](https://docs.astral.sh/ruff/) with a 100-column format
target, Python 3.12 syntax, standard error checks, unused-code checks, and import
sorting. The version is pinned in `environment.yml`. Safe lint fixes run before
formatting; unresolved diagnostics return a failure and require review.

```bash
python scripts/format_python.py
python scripts/format_python.py --check
```

The script works from any directory and respects the exclusions in `ruff.toml`.

## Regression checks

```bash
python -m unittest discover -s tests -t . -v
ctest --test-dir build --output-on-failure
```

## Pending implementation

Full NP101 model inference, onboard CPU deployment, TLM inference, and communication
between the draft and target models are not yet implemented. Capability diagnostics
and exported weights are preparation for full deployment; review their unresolved
hardware and memory acceptance items before proceeding.
