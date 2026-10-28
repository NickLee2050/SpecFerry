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

Select the execution target explicitly: `ep` (Linux x86_64) or `soc` (Linux riscv64).
Use separate build directories; a configured directory cannot switch platforms. CMake selects
one platform implementation and checks SDK/OpenSSL compilation and linking without running it.
No EP/SoC preprocessor macros are used; the SDK's required `LINUX` definition remains.

### EP build

Use matching EP driver/SDK libraries; device execution also requires access to `/dev/galcore`.
Install Conda, a C++17 compiler, CMake 3.18+ and OpenSSL development headers.

```bash
conda env create -f environment.yml
conda activate SpecFerry
cmake -S . -B build/ep -DSPECFERRY_TARGET=ep -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/ep -j 4
```

Update with `conda env update -n SpecFerry -f environment.yml`. SDK defaults are `/usr/inc`
and `/usr/lib/ljmicro`; override `LJMICRO_SDK_INCLUDE_DIR` / `LJMICRO_SDK_LIBRARY_DIR`
at CMake configure time. Python generation/check commands use `build/ep/bin` and `build/ep/tests`.
Existing binaries under `build/bin` are not selected. Point clangd at the chosen target's
`build/ep/compile_commands.json` or `build/soc/compile_commands.json` when reviewing that platform.

### SoC cross-build

Use the vendor RISC-V GCC V2.6.1 toolchain and a snapshot of the board's SDK and RISC-V
`libcrypto.so.1.1`. Keep host and board build directories separate. No board internet access is
needed. `soc_root` below must contain the board paths and compatible OpenSSL development headers;
the accepted build used OpenSSL 1.1.1f public headers with a checked LP64 configuration because the
board had no development headers. Do not copy host libraries into the target snapshot.

```bash
soc_toolchain=/absolute/path/to/NP101-riscv-gcc-linux-5.10.4-glibc-x86_64-V2.6.1
soc_root=/absolute/path/to/board-root
CCACHE_DIR="$PWD/.cache/ccache" cmake -S . -B build/soc -DSPECFERRY_TARGET=soc \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchains/np101-riscv64.cmake" \
  -DNP101_TOOLCHAIN_ROOT="$soc_toolchain" -DCMAKE_FIND_ROOT_PATH="$soc_root" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DLJMICRO_SDK_INCLUDE_DIR="$soc_root/np101_driver/build/sdk/include" \
  -DLJMICRO_SDK_LIBRARY_DIR="$soc_root/np101_driver/build/sdk/drivers" \
  -DOPENSSL_INCLUDE_DIR="$soc_root/usr/include" \
  -DOPENSSL_CRYPTO_LIBRARY="$soc_root/usr/lib/libcrypto.so.1.1"
CCACHE_DIR="$PWD/.cache/ccache" cmake --build build/soc -j 4 \
  --target specferry_opt_graph_generate specferry_opt_generate
```

The SoC deployment command packages the native binary, weights, one full-model NBG, prepared
requests/references and SDK CL headers. It pins the installed board libraries and places shader
headers in a private runtime directory. Source NBGs and EP packages are unchanged; the derived
package uses the `np101-soc-riscv64` profile. Python model exporters still run on the host.

```bash
python scripts/np101_soc.py prepare --package /absolute/path/to/opt-fullgraph-package \
  --request /absolute/path/to/prepared-request --sdk /absolute/path/to/board-sdk \
  --output .cache/deploy/opt-soc
python scripts/np101_soc.py deploy .cache/deploy/opt-soc --host np101
python scripts/np101_soc.py run .cache/deploy/opt-soc --host np101 \
  --output .cache/runs/opt-soc --timeout 180
# Resume evidence retrieval after a transport interruption; never relaunch the model:
python scripts/np101_soc.py collect .cache/deploy/opt-soc --output .cache/runs/opt-soc
```

`prepare` is offline; `deploy` transfers files and checks hashes without opening the NPU.
`run` accepts fullgraph C64/C128, mapped-slot KV, independent reset requests and no Context
rebuild. Optional block16 prefill uses `prepare --prefill-package` and request options
`block 16`, `prefill_head last-token`. The original handwritten-attention/MLP-NBG candidate failed
Linux memory acceptance. The compiled-attention path uses nineteen shared ordinary-weight FFNs
and five full decoder NBGs. SoC packages use batch-axis convolution for the first four FFNs;
an optional verified cache package extends that prefix to twelve or all nineteen
(`package_np101_opt_verify_cache.py --layers 19`), with FCL for any remainder.
The cache packager, pinned identities and lifecycle boundary are described in
[validation](tests/README.md#soc-formal-generation); EP selection is unchanged.
FCL also replaces the M16 input
projection, avoiding numerical errors observed in the original MatMul path. SoC compiled-attention
prefill uses 8192-row embedding chunks (at most 8MiB); head configuration and EP remain unchanged.
Native `preparation.json` separates weight, prefill and M1 package validation costs. SoC validates
M1 last before construction while preserving every integrity check. Three reset requests
with 17/49/17 input tokens and 16 outputs each passed the original C64 independent-cache checks.
Compiled-attention prefill retains M1's FP32 input KV. On SoC, all decoder NBGs share one fixed
output K/V pair; each layer commits its current block before the next producer runs. Canonical
inputs remain separate, and EP keeps independent outputs. C128 passed 97/17 input tokens and
16 outputs each with both independent and shared input KV;
the shared change has not been separately accepted on C64. Shared inputs remove the final prefix
handoff and duplicate reset. Shared SoC outputs first passed two 17-input/two-output reset requests,
reducing sampled SYSTEM peak by 23.03MiB. With the 8192-row embedding policy, the latest combination
also passed 97/17-input reset requests with 16 outputs each and final-layer KV/unused-slot checks.
Initialization and request-specific performance are reported separately in the status document.
Handwritten attention retains its independent FP16 cache and widening
handoff. The implementation is `native/models/opt/block_prefill.*`, without platform macros.
M1 consumes the remaining 1–16 prompt tokens and all generated tokens.
M1's embedded head still executes on every M1 step. This path does not support continuation or the
resident benchmark entry. Acceptance scopes and measurements are recorded in the status document.
It requires the documented 1792MiB pool, stuckDump=1, board Python 3.9+, a matching SDK and
host EP isolation. The detached board supervisor owns the child deadline (maximum 300 seconds),
kernel monitoring, Linux memory reserve and 45-second delayed release checks. SSH interruptions
retry discovery of the same run; `collect` performs host numerical acceptance. Existing recovery
markers block ordinary runs. `--diagnostic-baseline` is only for explicitly authorized diagnostics:
it binds exact boot, old timestamped fault occurrences, marker/bundle hashes and deadline; success
retains the old marker. Diagnostic authorization does not automatically enable per-SDK-call tracing.
Neither command loads or resets the driver. Qwen on SoC, whole-pool
integrity and long-term stability remain unvalidated.
See [status](docs/np101-status.md#soc部署与内存) for deployment evidence and acceptance boundaries.
SoC runs require board-side deadlines, kernel monitoring and release/delayed checks; retain host EP
isolation. Driver autoload remains disabled.

### Source and platform boundaries

| Location | Responsibility |
|---|---|
| `native/np101/` | Shared SDK wrappers, tensors, operators, budgets and package checks |
| `native/models/`, `native/inference/`, `native/cli/` | Shared model composition, generation and native commands |
| `native/platform/runtime_profile.hpp` | Common platform interface; ordinary functions, no mode macros or virtual dispatch |
| `native/platform/ep/`, `native/platform/soc/` | Target-specific SDK paths and driver identity checks |
| `cmake/platforms/` | Target architecture, SDK defaults and implementation source selection |
| `cmake/toolchains/`, `cmake/np101-sdk.cmake` | Compiler/sysroot settings and shared SDK linkage |
| `python/specferry/platforms/soc/`, `scripts/np101_soc.py` | SoC packaging, SSH transport and standalone board supervision |
| Other `scripts/`, `python/specferry/` modules | Shared host export/reference/acceptance tools and existing EP launchers |

The selected `specferry_platform_ep` or `specferry_platform_soc` library implements the same
interface; `np101_sdk_identity` retains shared package validation. A package for the other target
is rejected before driver inspection. Native generation records the linked target, architecture,
required SDK identity and runtime library directory in `platform.json`, outside inference timings.
Compilation does not select a KV transfer policy. Existing configurations retain their device
copy/update path. For a fullgraph C64/C128 package, `--kv-transfer mapped-slot` writes
`kv_transfer mapped-slot` in the request's `run-options.txt`; omission means `device-copy`.
The mapped strategy keeps fixed, independent inputs and outputs in one NBG graph, then copies
only the current FP32 KV slot through a reused host buffer after each synchronous run. It has
no extra device copy/reset graphs. Requests reset KV; continuation and other capacities are
rejected. Initialization/reset and coarse diagnostics remain outside generation timing.
Use `--prepare-only` to prepare requests; the Python launcher still executes locally on EP,
not remotely on SoC. The native SoC entry accepts the same request setting with a SoC package.

For graph-internal KV updates, `export_np101_opt_model.py --kv-update select` exports FP32
broadcast selection in place of ScatterND. It preserves the runtime ports and pairs with the
mapped-slot writeback above. SoC C64/M1 is validated; the default export remains `scatter` to
preserve accepted EP configurations. Performance and scope are recorded in the status document.

The SoC optimization `--scope fullgraph --kv-update narrow-select` keeps canonical KV inputs in FP32,
selects narrowed FP16 values for attention, and publishes only the current FP32 K/V slot. Its package
requires mapped-slot commits from output slot zero to the requested input position. Decoder packages and
M16 exports do not support this option; existing export defaults remain unchanged.

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
ctest --test-dir build/ep -L cpu --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

[Test commands](tests/README.md) cover exports, model regressions and explicit board diagnostics.
[SDK issues](docs/np101-sdk-driver-validation-report.md) and the incident ledger retain unresolved
contracts and recovery evidence. Do not execute archived commands with current binaries.
