# Validation layout

| Directory | Purpose | Device required |
|---|---|---|
| `python/` | Download, export integrity/precision, memory budget, and numerical comparison unit tests | No |
| `native/generation_test.cpp` | BOS/EOS, capacity, generation limits and consumed-token semantics | No |
| `native/np101/opt_io_check.cpp` | Shared embedding/head blocks, learned positions and greedy selection | Yes |
| `native/np101/sampling_check.cpp` | Categorical frequencies, seed replay and full-vocabulary bounds | Yes |
| `native/np101/data_test.cpp` | Tensor/component contracts, native weight reader, and fixture parser unit tests | No |
| `native/np101/conv_relu_pool_test.cpp` | FP16 convolution, ReLU, and max-pooling numerical regression | Yes |
| `native/np101/op_check.cpp` | File-driven operator checks with changing inputs | Yes |
| `native/np101/graph_sharing_check.cpp` | Cross-graph tensor attachment and execution | Yes, when the SDK exports the API |
| `native/np101/weight_allocation_check.cpp` | Constant/mutable weight allocation, state coexistence, and full weight readback | Yes |
| `native/np101/delta_net_check.cpp` | Real-weight DeltaNet trajectories, nonzero state, reset, recreation and final-only readback | Yes |
| `native/np101/kv_cache_check.cpp` | Exact slot writes, untouched cache rows, fixed-reader visibility and bounds | Yes |
| `native/np101/decoder_check.cpp` | Explicit decoder slices, shared bindings, state/reset, transfer counts and capacity | Yes |
| `native/np101/opt_decoder_check.cpp` | OPT post-norm decoder slices, affine projections, LayerNorm and KV/lifecycle checks | Yes |
| `native/np101/attention_check.cpp` | Real-weight Attention, single-buffer KV, capacity, truncation, reset and final-only readback | Yes |

Run host tests from the repository root with the `SpecFerry` Conda environment:

```bash
python -m unittest discover -s tests -t . -v
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

CTest registers only host unit tests. Hardware checks are explicit `scripts/check_np101_*.py`
commands documented in the root README. Each uses a fresh output directory under
`.cache/runs/`; generated fixtures and logs are not source files.

The operator fixture format is a versioned test protocol, not a model compiler.
NumPy arrays are row-major; graph tensor dimensions list the contiguous axis first.
Files contain exact little-endian FP16, FP32, INT32, or byte-bool values.
Version 2 adds inclusive INT32 input bounds; the native reader accepts the
stateless subset of versions 1 and 2. Retired `feedback`, `reset_after`, and
`handle` storage fixtures are rejected before device initialization; use the
archived source/binaries to reproduce historical feedback experiments.
Node inputs must come from initialized tensors, graph inputs, or earlier nodes.
The Qwen catalog supplies fixed-input tolerances from `python/specferry/models/qwen3_5/tolerances.json`.
The generic comparator requires an explicit policy; it does not select model budgets.
Both nonfinite values and wrong output sizes fail validation.

Use `build/tests/np101_op_check --validate-only PATH/graph.txt` to validate native
fixture parsing and input sizes without creating a device context.

`test_operator_acceptance.py` protects final-only readback, independent graph
lifetimes, payload rejection before device access, failed release despite correct
outputs, and the distinction between numerical agreement, hardware proof, and
device residency. The current catalog contains 49 synthetic operator cases plus
four optional captured-reference projections, with no cross-execution feedback.
The [operator acceptance record](np101-operator-acceptance.md) preserves historical
results, and the [state investigation record](state-feedback-investigation.md)
documents the retired experiments. The [DeltaNet module check](np101-delta-net.md)
now covers recurrent numerical state reuse/reset; the [Attention module check](np101-attention.md)
covers single-buffer KV. Device-residency evidence remains pending (`NP101-STATE-001`). `test_delta_net.py`
protects reference-input contracts and rejects incorrect or incomplete trajectory,
reset, final-only and lifecycle results without opening the device.
`test_attention.py` additionally protects cache-prefix comparison and exact
masked-suffix rejection. `test_decoder.py` checks pre-normalization fixture inputs,
independent per-layer reference state, exact repeat checks and lifecycle rejection.
The [decoder group check](np101-decoder.md) validates complete layers using the same
serialized device runner; it remains separate from the small mixer regressions.

Keep host unit tests small and independent of downloaded weights. Hardware failures,
timeouts, unsupported APIs, and missing execution evidence must remain visible;
never convert them into successful deployment acceptance.

SDK node parameters contain private state allocated by `vsi_nn_AddNode`. Change
individual public fields only; never clear the whole parameter structure or
overwrite `pool.local`. The convolution regression uses inferred virtual tensors
between operators, constant FP16 weight/bias bytes, and a 0.1 absolute tolerance,
matching the vendor's convolution demo. That demo tolerance does not replace the
separate DLM operator tolerances. Explicit diagnostic tensors and persistent
state must not be indiscriminately converted to virtual tensors.

## Reusable component regression

[Component contracts and checks](np101-components.md) document the shared/model boundary,
versioned component fixtures, small synthetic decoder and alternate KV layout.
`test_components.py` covers non-Qwen checkpoint names, explicit FP16 preservation,
alias rejection, independent memory resources and invalid component configuration.
The native data test parses component contracts without linking or opening the SDK.

[OPT validation](np101-opt.md) records the original FP16 checkpoint, CPU baseline,
byte-preserving export, independent decoder expectations and the corrected SDK crash.
`test_opt.py` checks import/precision rejection, chunk identity, trace ordering,
valid-prefix comparison and lifecycle/mask/repeat failures without a model download.

[Complete OPT generation](np101-generation.md) uses the production
`build/bin/specferry_opt_generate` executable for teacher-forced 4/8/24-layer
checks and normal text generation. `test_generation.py` protects token bounds,
scalar-only transfer accounting, exact resets and honest hardware-evidence gates.
`autoregressive_generation_contract` is a host CTest and never opens the SDK.

[Sampling validation](np101-sampling.md) covers the documented RANDOM_MULTINOMIAL
operator and OPT's optional sampling head. `test_sampling.py` rejects out-of-range
or distribution-ignoring output even when SDK execution reports success.
