# SpecFerry

SpecFerry runs small language models on an accelerator board. It is being developed for
**speculative inference**: a small draft model proposes tokens (units of text), and a larger
model checks them. This repository currently implements the draft-model inference path;
coordination with the larger model and rejection handling are not implemented yet.

The current hardware target is **NP101**, an accelerator board with a neural processing unit
(**NPU**) and a RISC-V CPU capable of running Linux. SpecFerry supports two execution modes:

- **EP** (PCIe endpoint): an x86 Linux host controls the accelerator through PCIe.
- **SoC** (system on chip): the control program runs directly on the board's Linux system.

OPT-350m and the Qwen3.5-0.8B text model are supported in the
[validated configurations](docs/np101-status.md). Model computation runs on NP101; Python tools
prepare weights, tokenize text, deploy packages and check results against a CPU reference.
Device execution combines graphs built through the vendor's OpenVX/ovxlib API with graphs
compiled ahead of time by the vendor tools. OpenVX/ovxlib provides the graph and tensor interfaces.

## Repository layout

```text
src/
  cli/                 C++ command-line entry points
  inference/           Shared generation loop and timing
  models/              OPT and Qwen model implementations
  devices/np101/api/   NP101 graphs, tensors and SDK integration
    ep/                Host execution policy
    soc/               Board execution policy
tools/
  deploy/              Export → package → execute
  reference/           Independent CPU model implementations
  validate/            Numerical and execution checks
  benchmark/           Performance measurement
  format/              Code formatting
cmake/                 Build targets and cross-compilation settings
tests/                 Unit, integration and explicit device tests
docs/                  Usage, design and validation records
```

EP and SoC share model and operator code; the build selects their execution-specific code.
Generated builds go in `build/`, and local model packages and measurements go in ignored `.cache/`.

## Getting started

You need Conda, CMake, Ninja, a C++17 compiler and the matching vendor SDK (software development
kit). Vendor libraries, drivers, model weights and compiled model packages are not included.
SoC builds additionally require the vendor RISC-V cross compiler and board libraries.

```sh
conda env create -f environment.yml
conda activate SpecFerry
```

Then follow [Build and run](docs/getting-started.md) for EP or SoC. It covers compilation,
model preparation, deployment and execution. Only OPT download is enabled; Qwen requires an
existing checkpoint, the original model weights and configuration.

## Further reading

- [Development guide](docs/development.md): code boundaries and contribution workflow.
- [Tests and exporters](tests/README.md): local checks and model-package generation.
- [Status and performance](docs/np101-status.md): validated configurations and measurement scope.
- [Device recovery](docs/np101-recovery-incidents.md): required checks before board execution.
- [TODO](TODO.md): remaining work and dependencies.
