# Validation layout

| Directory | Purpose | Device required |
|---|---|---|
| `python/` | Download, export integrity/precision, memory budget, and numerical comparison unit tests | No |
| `native/np101/data_test.cpp` | Tensor sizes, native weight reader, and fixture parser unit tests | No |
| `native/np101/conv_relu_pool_test.cpp` | FP16 convolution, ReLU, and max-pooling numerical regression | Yes |
| `native/np101/op_check.cpp` | File-driven operator and state feedback checks | Yes |
| `native/np101/graph_sharing_check.cpp` | Cross-graph tensor attachment and execution | Yes, when the SDK exports the API |
| `native/np101/weight_allocation_check.cpp` | Simultaneous weight/state tensor allocation | Yes |

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
Fixed-input tolerances come from `python/specferry/reference/tolerances.json`.
Both nonfinite values and wrong output sizes fail validation.

Use `build/tests/np101_op_check --validate-only PATH/graph.txt` to validate native
fixture parsing and input sizes without creating a device context.

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
