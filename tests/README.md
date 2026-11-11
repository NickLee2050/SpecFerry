# Tests and exporters

Start with the [build instructions](../README.md) and [architecture](../docs/development.md).
Accepted hardware scopes and measurements live in [status](../docs/np101-status.md), not this guide.

## Layout

`unit/` contains local contracts; `integration/` covers command and mocked deployment flows.
`device/np101/` contains explicit board checks, `support/` holds test-only helpers and
`fixtures/` contains small inputs. Tests are grouped by scope and responsibility, not language.

## CPU checks

```sh
python -m unittest discover -s tests -t .
ctest --test-dir build/ep -L cpu --output-on-failure
python tools/format/python.py --check
python tools/format/cpp.py --check
```

Python tests cover package/source/port integrity, generation policy and token counts, numerical
acceptance, memory budgets and mocked supervision/recovery. The four native CTest cases cover
data contracts, generation, public SDK timing and memory budgets; they never open NP101.
Cross-builds produce CPU test binaries but do not register them for execution on the host.

## Explicit device checks

Device checks are excluded from the default build and never registered with CTest:

```sh
cmake -G Ninja -S . -B build/ep -DSPECFERRY_TARGET=ep -DSPECFERRY_BUILD_DEVICE_TESTS=ON
cmake --build build/ep
```

Use the same option with an existing SoC cross-build if a specific device control is required.
Before every run follow the [recovery ledger](../docs/np101-recovery-incidents.md). Bare native
checks have no watchdog; use the existing bounded runner and full resource/kernel checks.
Do not run a historical diagnostic merely because its executable is available.

| Command | Maintained purpose |
|---|---|
| `tools/deploy/execute/np101/ep/opt.py` | Complete OPT inference, component baseline or graph backend, request/lifecycle acceptance |
| `tools/deploy/execute/np101/ep/qwen.py` | Complete Qwen request preparation/CPU reference and local EP inference |
| `tools/deploy/execute/np101/soc/cli.py` | Offline prepare, deploy, bounded remote run and evidence collection |
| `tools/validate/np101/compiled.py --plan PLAN --case CASE` | Source-bound compiled model acceptance |
| `tools/validate/np101/model.py` | OPT decoder, selection and IO contract checks |
| `tools/validate/np101/operators.py` | Lookup, projections, sampling, KV append and FP16 preservation contracts |
| `tools/benchmark/opt_conversation.py` | Existing OPT reuse/reset conversation comparison, separate processes |

Use `--help` for supported inputs and `--prepare-only` where available. Diagnostics are not supported
model configurations. Raw native checks require frozen fixtures and agreed recovery controls.
The memory control allocates 1/4/16 MiB per const/nonconst segment and checks retained contents;
it does not run a graph or establish whole-pool integrity. The compiled-graph control uses a small
NBG and existing one-to-three input references; it does not establish complete-model performance.

## Model acceptance

- Match binary, SDK, model source, package ports, request, precision and memory layout.
- Reuse the same CPU oracle when inputs and contracts are unchanged. Check actual generated tokens,
  coarse hidden/state/KV, unused slots, release and full kernel/delayed health results.
- For batched prefill, cover a partial block and reset. A divisible prompt alone cannot accept tails.
  Padding FFN rows must not advance DeltaNet or KV state.
- Keep original tolerances and record strict single-layer differences separately. Never silently
  discard a failed array or replace actual output with reference text.
- Report generated outputs / generation wall separately from post-first-token decode. Setup/Verify,
  initialization, reference evaluation and diagnostic snapshots are not inference performance.
- New timing instrumentation needs the same workload off/on/off. Public `--sdk-summary` aggregates
  in memory; `--sdk-timing` records individual calls and is unsuitable for normal throughput claims.
- Timeout, pending state, failed release or new kernel faults are not numerical passes. A recovered
  signal or correct text does not clear a recovery marker.

## Exporters

Python exporters run on the host. Offline NBG compilation requires the vendor's device-free
compiler rootfs. The resulting runtime package contains NBGs and source/SDK/port/budget manifests;
compiler ONNX, generated C and logs remain outside the runtime package.

| Script | Output |
|---|---|
| `tools/deploy/export/opt/cli.py` | OPT packed weights and native index |
| `tools/deploy/export/qwen3_5/cli.py` | Cached Qwen text weights, native index and capacity estimate |
| `tools/deploy/export/opt/ffn.py` | Fused M1 FFN/final-normalization graphs and optional head/projections |
| `tools/deploy/export/opt/projections.py` | QKV/output/head graphs with explicit public port permutations |
| `tools/deploy/export/opt/model.py` | Decoder/full-model NBGs, fixed capacity and KV precision/update contracts |
| `tools/deploy/export/qwen3_5/blocks.py` | DeltaNet or FFN blocks and combined source-bound packages |

For Qwen, export normalized DeltaNet mixers with `--outer-product multiply --fuse-input-norm`;
`--joint-state-reads` selects the accepted combined FP32 state-read formulation. Export all FFNs
with `--part feed-forward --all-layers --batch-tokens 1|8|16`, then combine with the matching mixer
package using `--mixers-package` and `--package-output`. Width4 is available for isolated export
contracts, not the formal SoC model entry. SoC batched FFN requires all24 layers at one common width.
Each script validates its prerequisites; source weights, public ports and SDK identities must agree.

OPT's compiled prefill package shares M1 KV inputs and scratch outputs, committing each layer
before reuse. `select`/`narrow-select` and FP16 KV have different port contracts; do not interchange
packages or infer support from an exporter accepting an option. See status for accepted combinations.

## Retired diagnostics

Private-offset EP Verify interposition and the unhelpful shared-residual/publish-suppression layout
are removed. Existing packages selecting that layout must use their original frozen runner/binary;
current preparation rejects them. Shared FCL, norm fusion, batched FFN and FFN state commit remain.
Experimental probes belong in ignored storage. Preserve their source identities, measurements and
fault evidence, but do not add temporary modes to production commands.
