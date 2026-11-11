# Development guide

See the [project overview](../README.md) and [terminology/build guide](getting-started.md)
for the hardware, execution modes and graph terminology. Maintained documentation, comments and
user-facing messages use English; original vendor material and archived evidence retain their
source language. Keep status, plans and incident evidence separate rather than duplicating diaries.

## Read the code in this order

```text
tools/deploy/execute/np101/soc/cli.py
    → tools/deploy/pack/np101/                 assemble the deployment
    → tools/deploy/execute/np101/soc/remote.py transfer and collect
    → board_runner.py                        deployment and recovery policy
        → src/cli/supervise.cpp              independent process watchdog
        → src/cli/qwen_generate.cpp          check arguments and report errors
        → src/models/qwen3_5/execution.cpp   load, generate, report and close
        → src/models/qwen3_5/                order layers and advance state
        → src/devices/np101/api/             graphs, tensors and vendor SDK

src/inference/ supplies request parsing, token scheduling and generation reports.
src/io/ supplies shared binary file output, without device or model policy.
```

| Layer | Owns | Must not own |
|---|---|---|
| `src/cli/` | Argument count, usage and exception-to-exit conversion | Model loading or token loops |
| `src/inference/` | Execution paths, token requests/options, token loop, EOS/capacity and result metrics | Model tensor names, SDK operators |
| `src/io/` | Binary file output | Model state or deployment trust policy |
| `src/supervision/` | Child process groups, readiness, fixed deadlines and termination | SDK calls, device recovery authorization |
| `src/models/` | Checkpoint names, layer ordering, model-specific precision/state | SSH, deployment, shell commands |
| `src/devices/np101/api/ops/` | Reusable graph construction and explicit tensor contracts | Model selection and conversational task codes |
| `src/devices/np101/api/` | SDK lifetime, tensors, shared weights, budgets, package integrity | Request scheduling |
| `src/devices/np101/api/{ep,soc}/` | Target SDK/runtime policy | A second implementation of model math |
| `tools/deploy/export/`, `tools/deploy/pack/` | Checkpoints, compiled artifacts and deployment packages | Device execution |
| `tools/deploy/execute/np101/` | Request preparation, SSH, bounded execution and recovery | Model arithmetic |
| `tools/reference/` | Independent CPU reference | Device recovery exceptions |
| `tools/validate/`, `tools/benchmark/` | Numerical acceptance and measurement | Automatic policy waivers |

The runtime remains deliberately small. Model classes call NP101 graph builders, and API runtime
profiles retain accepted execution policies. Model adapters own their loading and release sequence;
shared token scheduling does not imply a universal graph or state interface. No IR implementation or empty
placeholder directory is provided.

Run tool files with the SpecFerry Python interpreter from the repository root, or use their
`python -m tools...` module entry. Commands retain their arguments; only their paths changed.
There are no duplicate `scripts/` entry points or compatibility imports for the old Python package.
Recovery markers and deployed `native/` output directories retain their existing names and semantics.

CMake selects a single platform implementation. `cmake/targets.cmake` lists shared targets
and dependencies; `tests/CMakeLists.txt` separates CPU regressions from explicit device
checks. `cmake/toolchains/` only selects the cross compiler and sysroot. Keep platform differences
at these boundaries rather than inserting preprocessor branches into model execution.
The documented generator is Ninja. Keep `build/ep` and `build/soc` separate, and use a clean
build directory when changing generators. ccache remains an optional compiler launcher;
changing the build executor does not change the compiler, SDK or runtime graph verification.

## Configuration contracts

| Owner | Contents | Validation boundary |
|---|---|---|
| Model configuration | Dimensions, vocabulary, layer types and precision/state semantics | Derived from the checkpoint and existing request contracts; not arbitrary CLI settings. |
| Execution options | OPT `GraphOptions`; Qwen `ModelOptions`; preparation-side `QwenExecutionOptions` | Named choices only. Model/package checks still reject unsupported combinations. |
| Compiled artifacts | OPT `CompiledArtifacts`, existing Qwen block maps | Own source-validated paths/port contracts, not additional device weights. |
| Preparation JSON | Existing source paths and a restricted `request_options` subset | `prepare --config`; CLI > file > existing defaults. This is not the generated bundle manifest. |
| Deployment and runtime policy | SDK/file identity, memory layout, budgets, supervision | Remains in existing packer/runner contracts; preparation JSON cannot override it. |

The OPT constructor separates execution choices, artifact metadata and progress callbacks instead
of mixing them in a long positional list. Validation order, graph construction and lifecycle remain
unchanged. Qwen already uses a named C++ model-options object; its preparation options now share one
validator/default definition across CLI and packing. Existing artifact and request schema versions remain unchanged.

Host JSON and native text contracts intentionally coexist for now: packers generate the exact native
inputs consumed by reusable C++ request loaders in each model module. Do not remove
checks at a separate trust boundary merely because the host also validates a field. Shared JSON IO
remains in the self-contained policy adapter; the native process supervisor only writes its own
small lifecycle records and does not duplicate deployment parsing.

## Model construction and execution

The three native commands call model-level execution functions with `inference::ExecutionPaths`
(deployment, request, output); Qwen also receives its compiled package path. They retain the existing
positional command interface. `opt/request.cpp` and `qwen3_5/request.cpp` reject unsupported request
combinations before creating a Context. Package loaders remain authoritative for weights, SDK identity,
ports and budgets; deployment checks are not replaced by request parsing.

OPT `partitioned_graph_build.cpp` and Qwen `model_build.cpp` own constructor validation and graph
construction. Their model implementation files own generation, prefill/decode, state access and close.
Private `*_internal.hpp` files declare shared implementation storage; consumers use public model
headers. State and graph ownership order is unchanged. No additional Context or weight copy is created.

`inference/report.cpp` writes the existing raw generation fields plus `generated_tokens`,
`decode_intervals`, `output_tokens_per_second` and `decode_tokens_per_second`. Output throughput is
actual generated tokens divided by generation wall time; decode rate uses only intervals after the
first token. Undefined rates are JSON `null`. Reporting and metric validation run after generation.
Python retains independent aggregate validation for old and new evidence; the live native process no
longer requires Python to calculate its single-request rates. Tokenization, graph export, CPU reference
and deployment/recovery policy remain Python responsibilities. Process supervision runs in C++.

Qwen's `ModelOptions` names graph choices instead of encoding them in a long positional argument
list. Package manifests supply batch width, precision, public ports and weight identities. Invalid
combinations are rejected before model construction. Defaults preserve the conservative path;
choosing a new option does not establish hardware acceptance.

`Model::generate` delegates token scheduling to `inference::generate_tokens`. `DecoderLayer`
composes normalization, a mixer and FFN; mixer state advances only for real tokens. Batched FFNs
process independent prompt rows. During decode a single row is padded into the same FFN graph,
so a second copy of the FFN weights is not needed. Padding must never advance recurrent state.

Separate `canonical` and `next` recurrent buffers are intentional. A successful mixer produces a
pending next state; the corresponding copy must finish before the layer confirms it and consumes
another token. FFN-integrated commit uses fixed internal tensor bindings and preserves the imported
NBG's original external ports. Expanding those ports previously corrupted outputs on the pinned SDK.

## Ownership and failures

- A `Context` outlives every graph and tensor using it. Consumers close before storage owners.
- `WeightBank` shares backing storage; application budgets count it once. SDK workspace is measured
  separately and cannot be inferred from application payload sizes.
- Graph bindings are fixed after construction. Mapped KV append writes the current slot without
  rebinding or reverifying a copy graph. Preserve mapping/unmapping and cache synchronization.
- A failed SDK call poisons the model/layer. Do not continue with partially updated state. Explicit
  `close()` reports release errors; destructors make best-effort cleanup without replacing exceptions.
- Set individual `vsi_nn_AddNode` public fields. Never zero its parameter union or free SDK-owned locals.

The runtime contains bounded supervision and optional public-API timing, not private SDK memory
inspection. Timing is off by default; summary mode accumulates in memory. Call-level tracing is an
explicit diagnostic and must not be used to report ordinary throughput.

## Independent supervision

`specferry_supervise` is a separate C++ process with no SDK link dependency. The SoC policy adapter
starts it with fixed preparation/execution limits and the already-selected model command. The model
runs in a new process group with a private readiness pipe and `/dev/null` standard input. One valid
monotonic timestamp changes the deadline from preparation to inference; subsequent notifications
cannot restart it. The watchdog's active loop performs no SD or debugfs reads/writes.

The Python adapter retains deployment receipts, SDK hashes, full kernel checks and streaming health
samples, memory margins, delayed observation, release acceptance and recovery-marker authorization.
It sends a stop reason through a private nonblocking channel. Channel closure or adapter death also
stops the model. Sampling or evidence publication in Python can block without blocking the native
deadline. These policies are intentionally not duplicated in C++; a faulted or incomplete run still
requires the existing ledger rules and host numerical acceptance.

The guard sends a bounded startup record through a pipe; the adapter writes `child-start.json`.
Only the guard writes the final `child.json`, so delayed startup publication cannot overwrite final
exit evidence. It signals the entire model group, resumes a stopped group for graceful termination,
escalates to SIGKILL within a fixed grace period, and reports any remaining processes. If release or
final publication remains blocked, the adapter retains incomplete evidence and stops board work.
No process supervisor can recover an uninterruptible kernel/driver operation.

Both builds produce the companion binary beside the model executables. SoC packaging requires it,
records `supervisor_implementation: cpp-process-v1`, and checks its executable inventory entry,
RISC-V ELF identity, digest and deployment metadata. Frozen older bundles keep their original runner;
there is no silent fallback to Python process supervision. EP production runners are unchanged;
EP builds exercise the native guard with fake children only.

Board Python 3.9 standard-library support remains necessary for policy, deployment and collection.
This delivery moves process supervision, not all board utilities, into C++. No tokenizer, JSON parser,
SDK wrapper or extra language dependency was introduced.

## Packages and evidence

Exporters bind NBGs to source weight hashes, port permutations, precision, SDK identities and
constant payload bounds. Qwen export and SoC packaging share the native manifest serializer;
all three native model entries share bounded token parsing. The runtime does not invoke the offline converter, but SDK Setup/Verify
can still allocate and prepare device work. Deployment verifies payload checksums once; SoC launch
checks the matching receipt, file metadata, structure and SDK identity. Modified packages must be
redeployed. EP retains startup payload hashes.

Immutable accepted packages belong in ignored storage. Rebuilding a runner changes its identity;
do not substitute it into an old acceptance record or silently edit a deployment manifest. Historical
experimental packages need their original binaries/runners, not compatibility branches in current code.

## Making a change

1. Identify the model/operator boundary and the affected contract. Reuse existing implementation and
   references; do not add a mode unless it has a concrete supported use.
2. Build EP and SoC when shared native code changes. Run relevant CPU tests and formatting checks.
3. For hardware acceptance, reuse a representative request and CPU reference. Separate initialization,
   inference and diagnostics; report counts, numerics, actual memory and release status.
4. Keep temporary probes in ignored storage. Merge a candidate only after a useful whole-model result;
   leave stable conclusions in status and pending work in TODO. Device incidents go in their ledger.

The repository does not ship model weights or vendor libraries. Native builds require a compatible
SDK. Before running the complete Python suite, build EP as described in the test guide: supervision
integration tests execute its native guard with fake children. CPU checks require no board access.
