# Build and run

Start with the [project overview](../README.md) for the hardware and execution modes.
Run the commands below from the repository root.

Terms used in this guide:

- **SDK** (software development kit): the vendor headers and libraries used to build and run
  device code. **Rootfs** is a copy of the board filesystem used to locate compatible libraries
  during cross-compilation (building board executables on the host).
- **Checkpoint**: the original model weights and configuration. Export converts these into the
  weights and metadata consumed by SpecFerry.
- **NBG** (Network Binary Graph): a device graph compiled offline by the vendor tools. A model
  package combines these graphs with metadata describing their inputs, outputs and source weights.
- **KV cache**: stored attention keys and values reused when generating subsequent tokens.
  **Prefill** processes the input prompt; **decode** generates subsequent tokens one at a time.
- **FFN** (feed-forward network): the per-token projection block within a decoder layer.
  **FCL** (fully connected layer): an SDK operation used for linear projections.
- **C128** means a capacity of 128 tokens; **M8** means a graph processes eight token rows at once.
  These names describe dimensions, not different models.

## Requirements

Use Conda, Python 3.12, CMake 3.18+, Ninja, a C++17 compiler and OpenSSL development files.
Vendor SDKs, drivers, compiler rootfs, model checkpoints and NBG packages are **not included**.
The pinned SDK and model identities must match the package manifests; an arbitrary SDK version
is not a supported replacement. See [SDK limitations](np101-sdk-driver-validation-report.md).

```sh
conda env create -f environment.yml
conda activate SpecFerry
```

Only OPT is enabled in the download script. Qwen commands require an existing checkpoint.
Generated packages and measurements belong under ignored `.cache/`, not in the source tree.

## Build

Use Ninja and select one target per build directory. EP and SoC share model/operator code and link
different platform implementations; no platform-selection macros are needed. When migrating an
existing Make build, remove its build directory before configuring Ninja. Generated build files
are disposable; model packages and accepted evidence belong under `.cache/`.

### EP

```sh
cmake -G Ninja -S . -B build/ep -DSPECFERRY_TARGET=ep -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/ep
```

SDK defaults are `/usr/inc` and `/usr/lib/ljmicro`. Override `LJMICRO_SDK_INCLUDE_DIR` and
`LJMICRO_SDK_LIBRARY_DIR` for a matching installation. Execution requires `/dev/galcore` access.

### SoC

Use the vendor RISC-V GCC V2.6.1 toolchain and a snapshot of the board SDK, compatible OpenSSL
headers and RISC-V `libcrypto.so.1.1`. Do not substitute x86 libraries. The board needs no internet
access. Replace these paths with your local toolchain and rootfs snapshot:

```sh
soc_toolchain=/absolute/path/to/NP101-riscv-gcc-linux-5.10.4-glibc-x86_64-V2.6.1
soc_root=/absolute/path/to/board-root
cmake -G Ninja -S . -B build/soc -DSPECFERRY_TARGET=soc \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchains/np101-riscv64.cmake" \
  -DNP101_TOOLCHAIN_ROOT="$soc_toolchain" -DCMAKE_FIND_ROOT_PATH="$soc_root" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DLJMICRO_SDK_INCLUDE_DIR="$soc_root/np101_driver/build/sdk/include" \
  -DLJMICRO_SDK_LIBRARY_DIR="$soc_root/np101_driver/build/sdk/drivers" \
  -DOPENSSL_INCLUDE_DIR="$soc_root/usr/include" \
  -DOPENSSL_CRYPTO_LIBRARY="$soc_root/usr/lib/libcrypto.so.1.1"
cmake --build build/soc
```

Both builds produce `specferry_opt_generate`, `specferry_opt_graph_generate` and
`specferry_qwen_generate` under `build/<target>/bin`. CPU tests are enabled by `BUILD_TESTING`;
device checks additionally require `SPECFERRY_BUILD_DEVICE_TESTS=ON` and never run through CTest.
If ccache is installed, add `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` to either configure command.
Leave build parallelism to Ninja by default; do not add `--parallel` or `-j` unless an explicit
resource limit is needed.

## Prepare and run

Use new output directories. A weight deployment contains `weights.bin`, `weights.index` and its
manifest; an NBG package contains graphs and source/port/SDK contracts. They are different inputs.
See [exporters](../tests/README.md#exporters) to create them with the vendor converter.

Prepare a Qwen prompt and CPU reference without opening the NPU:

```sh
python tools/deploy/execute/np101/ep/qwen.py --model /absolute/path/to/qwen-weights \
  --checkpoint /absolute/path/to/qwen-checkpoint \
  --compiled-mixers /absolute/path/to/qwen-package --capacity 128 \
  --prompt 'Write a short invitation to dinner tomorrow.' --max-new-tokens 16 --repeats 1 \
  --output .cache/requests/qwen-dinner --prepare-only
```

The generated native request/reference directory is `.cache/requests/qwen-dinner/request`.
For SoC, package the cross-built binary with the weights, NBGs and request:

```sh
python tools/deploy/execute/np101/soc/cli.py prepare --model qwen \
  --deployment /absolute/path/to/qwen-weights --package /absolute/path/to/qwen-package \
  --request .cache/requests/qwen-dinner/request --sdk /absolute/path/to/board-sdk \
  --qwen-projections shared-fcl --qwen-kv-write-mode mapped \
  --output .cache/deploy/qwen
python tools/deploy/execute/np101/soc/cli.py deploy .cache/deploy/qwen --host np101
python tools/deploy/execute/np101/soc/cli.py run .cache/deploy/qwen --host np101 \
  --output .cache/runs/qwen-dinner --prepare-timeout 240 --timeout 300
```

`np101` is an SSH alias configured by the operator. Read the
[recovery ledger](np101-recovery-incidents.md) and ensure the board's loaded driver, memory layout
and SDK match the bundle before running. Commands do not
load/reset drivers or change board networking. Preparation and deployment do not open the NPU.
If SSH disconnects, collect the **same run** instead of starting another inference:

```sh
python tools/deploy/execute/np101/soc/cli.py collect .cache/deploy/qwen --output .cache/runs/qwen-dinner
```

SoC Qwen supports C64/C128; C512 has also passed bounded requests with the M8 FFN combination.
With a complete M8 or M16 FFN package and a C128/C512 request, preparation
may select `--qwen-ffn-batch 8 --qwen-decoder-layout fused-attention-norm
--qwen-state-commit ffn`. These options require shared FCL projections and mapped KV. The M1
fallback remains available; archived experimental Delta batch graphs are not a formal option.

OPT uses the same SoC workflow with `--model opt`, a compiled model package and an OPT request;
`--prefill-package` adds a source-bound M16 prefill package. Current OPT packaging pins a legacy
512 MiB Linux/2304 MiB NPU layout; Qwen pins 768 MiB Linux/2048 MiB NPU. **Do not deploy a bundle
onto a different layout.** Existing accepted bundles retain their original parameters.

For local EP generation, use `tools/deploy/execute/np101/ep/opt.py` or
`tools/deploy/execute/np101/ep/qwen.py` with their accepted package;
`--prepare-only` leaves execution disabled. EP baselines are frozen and documented in status.

