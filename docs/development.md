# Development guide

## Read the code in this order

```text
tools/deploy/execute/np101/soc/cli.py
    → tools/deploy/pack/np101/                 assemble the deployment
    → tools/deploy/execute/np101/soc/remote.py transfer and collect
    → board_runner.py                        bounded child supervision
        → src/cli/qwen_generate.cpp          parse package and request
        → src/models/qwen3_5/                order layers and advance state
        → src/devices/np101/api/             graphs, tensors and vendor SDK

src/inference/ supplies the shared token loop and generation timing.
```

| Layer | Owns | Must not own |
|---|---|---|
| `src/inference/` | Token loop, EOS/capacity, generation timing and request semantics | Model tensor names, SDK operators |
| `src/models/` | Checkpoint names, layer ordering, model-specific precision/state | SSH, deployment, shell commands |
| `src/devices/np101/api/ops/` | Reusable graph construction and explicit tensor contracts | Model selection and conversational task codes |
| `src/devices/np101/api/` | SDK lifetime, tensors, shared weights, budgets, package integrity | Request scheduling |
| `src/devices/np101/api/{ep,soc}/` | Target SDK/runtime policy | A second implementation of model math |
| `tools/deploy/export/`, `tools/deploy/pack/` | Checkpoints, compiled artifacts and deployment packages | Device execution |
| `tools/deploy/execute/np101/` | Request preparation, SSH, bounded execution and recovery | Model arithmetic |
| `tools/reference/` | Independent CPU reference | Device recovery exceptions |
| `tools/validate/`, `tools/benchmark/` | Numerical acceptance and measurement | Automatic policy waivers |

This is a directory migration, not a new device-independent model implementation. Model classes
still call the NP101 graph builders, and API runtime profiles still contain accepted execution
policies. Separating these responsibilities is a later change. No IR implementation or empty
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

## Model construction and execution

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

The repository does not ship model weights or vendor libraries. A source-only checkout can run CPU
contract tests after installing Python dependencies; native builds also require a compatible SDK.
