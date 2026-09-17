# Retired state-feedback investigation

Recorded and archived on 2026-09-16. This is a historical diagnostic record,
not a supported state implementation or completed hardware acceptance.

## Findings

- SDK RNN feedback with handle-created tensors returned incorrect results from
  the second execution. Ordinary tensors passed numerically but used SDK host
  state buffers, which do not meet the resident-inference requirement.
- Two fixed OpenVX ADD graphs sharing the same tensors passed A-to-B/B-to-A
  execution, including FP16/FP32 and 32 updates of a real-shaped FP32 state.
  This establishes a candidate reuse mechanism, not full DeltaNet/Attention,
  reset correctness, physical residency, or per-node hardware execution.
- Delays, explicit waits, and flushing did not fix the swap path. Rebuilding
  after each swap plus reading the source before the destination passed the
  small control. Binding and data visibility remain possible explanations;
  the exact root cause was not established.
- All 48 temporary configurations exited normally: 17 passed numerically and
  31 reproduced the swap-path failure. No device recovery was required.

The temporary runner and its CMake target, plus 11 handle and 11 ordinary SDK
RNN cases, were removed from active code. Neither `SetupRNNConnections` nor
direct OpenVX graph construction is described as an application interface in
the chip team's `demo/ref_op_api_guide.md`. A future exception requires a concrete
semantic limitation or significant predictable performance cost and validation.

## Reproduction archive

Local archive (ignored by Git):
[state-feedback-20260916.zip](../.cache/archives/state-feedback-20260916.zip).
Size: 248,601,609 bytes. SHA-256:

```text
349229409e04bafec4809908804911f77e6e6a86b0a0cf91f846d668d6cd5cc1
```

The archive contains the complete working-tree source before removal, vendor
demo/guide, and both original evidence directories:

- `.cache/runs/operator-acceptance-20260915/`
- `.cache/runs/state-buffers-20260916/`

All 10,879 source/evidence files were checked against their archived SHA-256
values. `archive-manifest.json` records these hashes. `ARCHIVE-README.md` explains
building in a separate extraction directory with Conda `SpecFerry`; the original
diagnostic READMEs preserve execution commands. Reports retain process arguments,
SDK hashes, executable snapshots, fixtures, logs, and failed comparisons. SDK
shared libraries and model checkpoints are not bundled. The original evidence
directories remain intact; the archive is not part of the build or test suite.

## Remaining acceptance

`NP101-STATE-001` remains open. Test the chosen implementation with actual
DeltaNet/Attention state when those modules are built. Keep three representative
checks: a correct changing-input trajectory, reset matching a fresh instance,
and continued execution without a per-step host state transfer. The last check
needs evidence about SDK behavior as well as application readback counts.
Numerical results, resource cleanup, hardware execution, and physical residency
remain separate acceptance criteria. Retiring failed tests closes none of them.

Cleanup validation: 38 Python unit tests, the native CPU CTest, Ruff/clang-format,
and native validation of all 53 retained historical fixtures passed. Targeted
device regression passed three convolution iterations plus mixed FP16 weights,
FP32 DeltaNet update, and cache writes over two graph lifetimes each. All four
processes released normally; no SDK process or recovery marker remained. Reports
under `.cache/runs/state-cleanup-20260916/` retain hardware acceptance as unverified
and state reuse as not implemented. The retired hardware matrix was not rerun.
