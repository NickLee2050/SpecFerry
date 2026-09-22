# SpecFerry
A testbed for speculative decoding across heterogeneous edge devices and a local inference hub.

The active deployment target is full text inference of `facebook/opt-350m`
on NP101 as the draft language model (DLM). OPT now has an official FP16 CPU
reference, a byte-preserving checkpoint exporter, and a native decoder-slice
validation path. Biased projections use documented MatMul/Add after an isolated
FCL graph reproduced an SDK optimizer crash. OPT single-layer and four-layer
numerical/lifecycle checks now pass. Qwen3.5-0.8B references, mixers
and decoder checks remain a regression baseline. Shared NP101 graph construction, projections,
normalization, Attention/DeltaNet arithmetic, SwiGLU, tensor bindings and KV storage
now take explicit contracts. Model configuration, checkpoint names and decoder
composition live under `native/models/` and `python/specferry/models/`.
The complete OPT embedding-to-token SDK path is implemented, including all 24 layers.
Numerical validation is separate from the pending exclusive-NPU/residency acceptance.
See the [completed-module decoupling checklist](docs/model-decoupling-plan.md),
[component contracts and checks](tests/np101-components.md), and
[project constraints](AGENTS.md).

The Qwen full-model allocation experiment encountered an effective limit near
1 GiB in both tested constant and mutable tensor paths. The NP101 team is
evaluating the allocation mechanism. OPT's smaller weight payload needs its own
full-graph allocation, workspace and residency checks; it is not automatically
blocked or validated by the Qwen result. [Engineering follow-up](TODO.md) records
the evidence and remaining acceptance gates.

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

## Download the OPT draft model

```bash
conda activate SpecFerry
python scripts/download_models.py --dry-run
python scripts/download_models.py
```

Only OPT-350M is enabled. The script downloads the original checkpoint and
tokenizer under `.cache/models/facebook/opt-350m`, pins the revision, resumes
interrupted transfers, and writes `download-manifest.json` only after validating
the files. It selects one weight format: safetensors when available, otherwise
official PyTorch weights. OPT-350M uses `pytorch_model.bin`; TensorFlow and Flax
copies are excluded. This does not convert or deploy the model. Existing Qwen
downloads and cached validation artifacts are preserved.

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

## Validate the OPT checkpoint and decoder slices

Use the activated `SpecFerry` environment and a new run directory:

```bash
python scripts/reference_opt.py --output .cache/runs/opt-cpu
python scripts/export_opt.py
python scripts/export_opt.py --verify-only
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target np101_opt_decoder_check -j 4
python scripts/check_np101_opt.py --trace .cache/runs/opt-cpu/trace.npz \
  --layers 0 1 2 3 --steps 32 --output .cache/runs/opt-slice --prepare-only
```

The CPU reference checks prefill against sequential execution and cached decoding
against fresh prefixes. The exporter preserves all 388 source FP16 tensors, verifies
every byte independently, and aliases the LM head to the embedding. The default
deployment directory is `.cache/np101/opt-350m`.

`--prepare-only` creates independent official CPU expectations without accessing the
board. Use a new output directory and replace `--prepare-only` with `--diagnostic`
to execute the slice. Start with `--layers 0 --steps 8 --capacity 8`, then check four
layers at 32 and 512 steps. A SIGSEGV on the
first FCL attempt set a recovery marker; it was archived after basic health and
the corrected OPT lifecycle checks passed. Honor any new recovery marker.
`--diagnostic` permits numerical success without proving exclusive NPU execution
or physical residency. See [OPT slice validation and the SDK crash diagnosis](tests/np101-opt.md), and
[complete-model validation](tests/np101-generation.md).

## Generate text with OPT-350M

After verifying the OPT export and building the native targets:

```bash
cmake --build build -j 4
python scripts/generate_opt.py --prompt 'The capital of France is' \
  --max-new-tokens 32 --output .cache/runs/opt-generation
```

The C++ runtime keeps the embedding/head table, all 24 decoder layers and their
KV buffers alive together. Normal generation uploads scalar controls and reads
only predicted token IDs. Python tokenizes and displays text; it does not compute
model layers. Capacity defaults to 512 and batch size is one. The default decoding
policy comes from the checkpoint's `generation_config.json`, with omitted fields
resolved by the pinned Transformers version. For the retained OPT-350M checkpoint,
this resolves to greedy decoding (`do_sample=false`, `num_beams=1`). Generation
length remains an application setting: `--max-new-tokens` defaults to 32.
Reports include stop reason, consumed length, transfer counts, initialization and
wall-clock token timings. The last returned token has not yet been consumed into KV.
Use a new output directory for each run.
Add `--compare-cpu` to check generated token IDs with the official CPU model
after native execution using the checkpoint's default decoding policy; normal
generation does not load that reference. Reports retain the generation-config
hash, resolved policy and any explicit sampling override. Unsupported checkpoint
policies fail before SDK execution rather than being silently replaced.

To explicitly override the model policy with full-vocabulary sampling, add `--sample --seed 42`.
This uses temperature 1 without top-k/top-p filtering. The SDK samples logits
promoted exactly from FP16 to FP32; model weights remain FP16. Sampling uploads
16 seed bytes per prediction and still reads only the resulting token ID.
`--compare-cpu` is restricted to greedy mode because SDK and CPU random streams
are not interchangeable. See [sampling validation](tests/np101-sampling.md).

The [complete-model validation guide](tests/np101-generation.md) documents input/output,
4/8/24-layer checks, exact tie handling, bounds, reset/recreation and the preserved
hardware evidence gates. SDK execution success is not proof of exclusive NPU execution.

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

## Run the retained Qwen CPU reference and validation

This reference path still requires the existing Qwen3.5-0.8B checkpoint and its
verified `download-manifest.json`. Use `scripts/reference_opt.py` for OPT;
the two adapters share reference utilities but retain their own model policies.
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

## Check DLM operator capabilities

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
argmax ties, mixed constant/mutable weights, and individual FP32 state-update
operations. There are 49 synthetic cases and four optional captured-reference
projections. Every graph receives changing inputs across multiple executions;
these runs do not feed an output back into the next execution. Use repeatable
`--case NAME` options to isolate a failure;
`--prepare-only` writes fixtures without opening the device.

SDK RNN feedback and temporary buffer experiments have been retired. Their
[investigation record](tests/state-feedback-investigation.md) links the archived
source and evidence. Operator suite reports mark state reuse as
`not_evaluated_by_operator_suite`. The separate DeltaNet module now tests state
reuse/reset numerically; hardware residency remains unaccepted (`NP101-STATE-001`).

Each case retains its readable `graph.txt`, exact input/expected bytes, tensor shapes,
source hashes, SDK execution report, logs, and driver trace. The suite snapshots its
executable and records pending cases. It writes `op-capabilities.json` incrementally.
The default 300-second timeout applies to each process, including SDK teardown.
Hardware diagnostics are serialized through a repository-local lock.

The runner stages the installed `cl_viv_vx_ext.h` in each execution directory for
the SDK's runtime shader compiler. Its source and hash are recorded; use
`--shader-header PATH` when the SDK header is installed elsewhere. The default
per-case tensor payload limit is 768 MiB (`--max-payload-mib`); SDK overhead is
additional, so this limit is not a guarantee of available board memory. Invalid
declared token/cache indices are rejected before SDK context creation.

Use `--readback final` to read only the last output, and `--cycles 20` to repeat
complete graph creation/execution/release within one process. Final-only checks
do not replace comparison of each execution's output. Reports separate numerical comparison,
completed lifecycle, hardware evidence, and state residency, and include phase
timing, application read counts, host RSS, and file descriptor samples.

See [the operator acceptance record](tests/np101-operator-acceptance.md) for measured
results, remaining gates, and commands for targeted rechecks.

Fixed-input errors use the frozen reference tolerances. Exit code 2 means the
capability gate is blocked, including when individual node execution is unverified.
`--diagnostic` permits exit 0 for numeric passes while retaining the hardware blocker.
Header availability, graph verification, numeric agreement, and device residency
are distinct evidence. An explicitly selected software argmax path is reported
as a blocker. Cross-graph attachment is checked as an
optional SDK symbol because some library builds declare it without exporting it.

## Validate one DeltaNet layer

The C++ `np101_delta_net` library implements layer 0's mixer with fixed A-to-B and
B-to-A graphs, FP32 recurrent state and FP16 convolution history. Its computation
and step scheduling stay in C++; Python prepares an independent Transformers
reference and compares results. Decoder input normalization, residual and MLP
are not part of this mixer.

```bash
cmake --build build --target np101_delta_net_check -j 4
python scripts/check_np101_delta_net.py \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/delta-net --steps 32 --diagnostic
```

Use the `layer-0-3-sequential.npz` produced by the CPU reference. The runner checks
zero/nonzero initial states, reset, a fresh instance, and final-only readback.
`--prepare-only` prepares fixtures without device access. `--diagnostic` permits
exit 0 for numerical/lifecycle agreement; otherwise missing hardware evidence
returns 2. Fixed tensor sharing is a narrowly scoped exception to the vendor
guide, explained with ownership, precision and results in the
[DeltaNet validation record](tests/np101-delta-net.md). SDK matrix-node creation
warnings and unknown execution/residency evidence remain visible.

## Validate Attention and single-buffer KV storage

The layer-3 Attention mixer uses one preallocated FP16 K/V cache with capacity
512. It writes only the current token's 2 KiB K/V into slot views and reads the
same parent storage from a fixed Attention graph. Cache payload is 1 MiB per
layer. Reset and truncation change the valid prefix without copying the cache.

```bash
cmake -S . -B build
cmake --build build --target np101_attention_check np101_kv_cache_check -j 4
python scripts/check_np101_attention.py --storage-only \
  --output .cache/runs/kv-storage --diagnostic
python scripts/check_np101_attention.py \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/attention --steps 512 --timeout 1200 --diagnostic
```

The real-weight capacity suite passed 1,042 appends across initial, truncated,
reset, final-only and fresh trajectories, with 207 successful checks. The runner
also supports `--steps 2`, `8` or `32`, and `--prepare-only`. As with DeltaNet,
numerical success does not establish hardware execution or physical residency.
The SDK reverifies the small slot-copy graph when its destination changes; that
overhead remains. See the [Attention validation record](tests/np101-attention.md)
for ownership, API exceptions, the Softmax layout correction and acceptance limits.

## Validate complete decoder layers

The `np101_decoder` library composes real-weight layers 0-3, including input/post-mixer
RMS normalization, residuals, and MLP. Fixed shared tensors connect the graphs; only
the group input crosses the host boundary during a step. DeltaNet keeps separate
state for each layer, while Attention retains one 512-token KV allocation. Partial
SDK failure invalidates the whole group; reset is supported after successful steps,
but group truncation is not a substitute for restoring recurrent state.

```bash
cmake --build build --target np101_decoder_check -j 4
python scripts/check_np101_decoder.py \
  --trace PATH/TO/deployment-fp16/layer-0-3-sequential.npz \
  --output .cache/runs/decoder --steps 32 --diagnostic
```

Use `--first-layer 3 --steps 2` to isolate the complete Attention decoder, and
`--steps 512 --timeout 900` to check the full group at capacity. Each suite includes
reset, recreation, and a final-only trajectory of up to 32 steps. It checks that
normal steps upload only 2,048 input bytes plus eight Attention control bytes,
with no explicit activation/state readback. `--no-trace` disables driver tracing
for separate host timing observations; it does not establish NPU-only execution.
See the [decoder validation record](tests/np101-decoder.md) for precision, ownership,
results, memory accounting, and the remaining hardware evidence gates.

## Export retained Qwen text weights and check memory allocation

These CLI defaults enforce the retained Qwen checkpoint contract. The shared pack
reader/writer, integrity checks and memory accounting are model independent; the
Qwen adapter selects names, aliases and target dtypes. OPT uses `scripts/export_opt.py`;
its complete-model SDK initialization now passes; physical residency remains unproven.

```bash
python scripts/export_np101_dlm.py --output .cache/np101/Qwen3.5-0.8B
python scripts/export_np101_dlm.py --verify-only
python scripts/check_np101_export.py --output .cache/runs/export/torch-verification.json
python scripts/check_np101_allocation.py --weight-storage mutable \
  --output .cache/runs/weight-allocation-mutable
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
and keeps all weight and state tensors alive together. `--weight-storage mutable`
sets every weight to `is_const=false`, creates it with `vsi_nn_AddTensor`, then
uploads its bytes with `vsi_nn_CopyDataToTensor`. The existing `constant` mode
remains the default; the flag does not establish which physical pool is used.
After all allocations, the check reads every weight block back and compares its
bytes with the export. Both modes retain the same chunk sizes and state buffers.
The Python wrapper writes an explicit versioned `allocation-states.txt`; direct
native invocation requires `--state-spec FILE`. The historical allocation-only KV
shape is retained for reproduction and is distinct from the execution layout.
Reports record completed weights, uploaded/verified bytes, the current chunk,
allocation failures, and host peak RSS. The executable is snapshotted per run.
The test does not include full model graphs or their workspace, and an SDK
allocation alone does not prove physical board residency.

The [2026-09-17 all-mutable experiment](tests/np101-mutable-allocation.md) failed
at the same 1,070,874,144-byte upload boundary as the constant baseline. Changing
`is_const` alone does not resolve `NP101-MEM-001`; do not repeat the full-capacity
probe without a relevant change or vendor guidance.

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
  --case reference_delta_qkv --case reference_delta_gate \
  --case reference_attention_q_gate --case reference_mlp_up \
  --output .cache/runs/operators-reference
```

Use the trace from an existing successful CPU reference run. These cases compare
against captured module outputs and record both trace and weight hashes.

## Format C++ code

The repository's `.clang-format` uses LLVM style with a 100-column line limit
and a blank line between function definitions. Control statements (`if`, `else`,
`for`, `while`, and `do`) require braces even for a single-statement body;
clang-format inserts missing braces automatically.
Includes are sorted alphabetically within two groups separated by a blank line:
quoted project/SDK/third-party/platform headers first, angle-bracket C/C++ standard
headers second.
For a matching `.hpp`/`.cpp` pair, dependencies required by the header belong in
`.hpp`; implementation-only dependencies belong in `.cpp`. The source includes
its matching header without repeating that header's direct includes. SDK headers
follow the same rule; unrelated headers must not supply SDK declarations implicitly.
Remove unused includes. The matching header has no special sort priority.
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

OPT complete-model computation and greedy generation are implemented through the SDK.
Final NPU execution/residency and performance acceptance remain open. Onboard CPU
deployment, TLM inference and draft/target communication remain later work.
