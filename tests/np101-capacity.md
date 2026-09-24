# Bounded tensor capacity probe

`scripts/check_np101_capacity.py` supervises `np101_capacity_check` with the
existing device lock, timeout, process checks and recovery marker. This diagnostic
does not build or execute an operator graph. It measures simultaneously retained,
initialized SDK tensor payload, not model workspace or proven physical residency.

**Current submission policy, 2026-09-24:** const and non-const application payloads
are limited separately to 1024 MiB. The Python and native capacity entry points
reject larger targets before SDK access; the default target is 64 MiB. This does
not downgrade the installed driver or restore its former physical layout. The
larger results below are historical evidence, not enabled current test targets.

## Allocation bound and complete corruption map, 2026-09-23

The requested capacity experiment ignores content mismatches when determining
the accepted allocation budget. All four dtype/storage combinations retain **355
initialized 8 MiB blocks: 2,840 MiB = 2,977,955,840 bytes = 2.7734375 GiB**.
The next 8 MiB block is rejected. This is the observed boundary at 8 MiB
granularity for this allocation pattern; smaller final allocations, other block
sizes, mixed-pool layouts and physical exhaustion are not established.

| Dtype | Storage | Accepted initialized payload | Rejection on next block | Full scan |
|---|---|---:|---|---|
| F16 | constant | 2,840 MiB | `AddTensor`: create-tensor-from-data failure | Block 109, 24 mismatched bytes |
| F16 | mutable | 2,840 MiB | `CopyDataToTensor`: status -5 | Block 109, 24 mismatched bytes |
| F32 | constant | 2,840 MiB | `AddTensor`: create-tensor-from-data failure | Block 109, 24 mismatched bytes |
| F32 | mutable | 2,840 MiB | `CopyDataToTensor`: status -5 | Block 109, 24 mismatched bytes |

The installed `/usr/inc/VX/vx_types.h` defines -5 as `VX_ERROR_NOT_ALLOCATED`.
Mutable handle creation succeeds for the next block but initialization does not;
that uninitialized handle is excluded from the accepted payload and released
with the graph. The result does not prove a hardware fault or identify a pool's
physical capacity. It also does not demonstrate usable payload near 4 GiB.

The F16 constant experiment first skips readback while seeking the bound, then
uses a fresh process to scan all 2,840 MiB. The other three combinations seek the
same bound and then scan every retained block, even after finding a mismatch.
All **355 blocks** are examined in each large scan. Only block 109 (zero-based;
the 110th block) differs. Its other bytes and all 354 remaining blocks match.
All four modes return zeros in the same three intervals:

| Offset within block 109, bytes `[begin,end)` | Cumulative payload offset, bytes `[begin,end)` | Size |
|---|---|---:|
| `[1,274,680, 1,274,688)` | `[915,632,952, 915,632,960)` | 8 bytes |
| `[1,274,744, 1,274,752)` | `[915,633,016, 915,633,024)` | 8 bytes |
| `[1,274,808, 1,274,816)` | `[915,633,080, 915,633,088)` | 8 bytes |

These intervals lie in logical page 311 within that block. They correspond to
12 FP16 values or six FP32 values. The findings reproduce the earlier corruption
locations, but now establish that the remainder of the successfully retained
2,840 MiB also compares correctly in these runs. They **do not** identify a
physical bad page or justify skipping that logical page in the allocator.

All four initial 64 MiB controls pass. No run times out, exits on a signal or leaves
a device process; graph/context release completes, including after a rejected
allocation. The final 64 MiB control also passes after the matrix. The driver
reference count is zero and no recovery marker is present after all ten runs.
No computation graph, driver change or sudo command is involved.

Local evidence: `.cache/runs/capacity-map-20260923/` contains `control-*`,
`limit-f16-constant`, `scan-*`, `post-control-f16-constant`, host environment,
source/binary hashes and `results.json`. Per-block JSONL and first-mismatch binary
dumps are retained for each scan. The first F16 mutable wrapper conservatively
rejected an upload-stage bound; `summary-review.json` records its later offline
evaluation using the explicit initialized-payload contract. Its original native
report, execution evidence and initial summary remain unchanged.

The follow-up closure work is in the
[remaining acceptance checklist](../docs/remaining-acceptance-plan.md).
Corrupt bytes still fail deployment integrity; the allocation-only success does
not close `NP101-MEM-001`.

## Method

- Allocate nonvirtual FP16 or FP32 tensors in blocks of at most 8 MiB, retaining
  every successful tensor until the end. Both dtypes use the same byte budget;
  the FP32 tensors have half as many elements. Deterministic finite, nonzero
  values depend on the block and full element position (`splitmix64-finite-v1`).
  This replaces the old pattern's 4 KiB repetition and tests page aliasing as well
  as zero-filled or overwritten data. Memory readback does not validate FP32 ops.
- Constant mode supplies bytes to `vsi_nn_AddTensor`. Mutable mode creates the
  tensor and explicitly uploads with `vsi_nn_CopyDataToTensor`.
- Stop at the requested target or the first SDK rejection. The readback mode
  controls whether to stop at the first mismatch (`first`, default), scan all
  successful blocks (`all`), or skip readback (`none`). Every block coexists until
  readback is finished; then release the graph and context. Host staging remains
  bounded to one block and its readback.
- `target_reached` establishes a tested payload lower bound. `allocation_limit`
  records a rejected next block and requires full retained-byte verification and
  clean release. It is a bound for this allocation pattern, not proof of physical
  exhaustion or which driver pool supplied the memory. A signal, timeout, missing
  data or incomplete release is a failed experiment, never a capacity result.
  On a data mismatch, both expected and actual bytes of the first failed block
  are saved as `first-mismatch-expected.bin` and `first-mismatch-actual.bin`.
  Length is checked as well as byte contents. The structured `first_mismatch`
  records dtype, tensor name/ID, block index, chunk offset, first differing byte,
  expected/actual lengths and byte values (null when absent in a short read).

Targets are expressed in MiB; 1024 MiB is 1 GiB and 4096 MiB is 4 GiB.
Advertised 4 GB could mean decimal bytes (about 3814.7 MiB); usable payload may
also be reduced by reserved memory, SDK allocations, alignment and fragmentation.
Never equate the sum of module pool parameters with verified usable memory.

## Commands after device recovery

Use the `SpecFerry` Conda environment and fresh output directories. Start small,
review each result, and advance only after complete readback and normal release:

```bash
cmake -S . -B build
cmake --build build --target np101_capacity_check -j 4
python scripts/check_np101_capacity.py --dtype F16 --storage constant --target-mib 64 \
  --output .cache/runs/capacity-control
python scripts/check_np101_capacity.py --dtype F16 --storage constant --target-mib 1024 --readback all \
  --output .cache/runs/capacity-within-1gib
```

Repeat with `--dtype F32` and `--storage mutable` to compare dtype and
initialization paths, starting each configuration with a small control. Do not
launch a larger target after an earlier rejection in the same configuration.
The executable stops at the first rejected block and caps all requests at
1024 MiB. A normal exit with `allocation_limit` is an observed SDK rejection, not a
claim that the requested target was reached. Inspect `summary.json` and retain
`capacity.json`, SDK output, driver trace and executable snapshot.

The default integrity acceptance still requires matching retained bytes before
increasing a deployment budget. On 2026-09-23 the user separately requested an
allocation-bound experiment that ignores content errors, followed by a complete
corruption map. The same modes remain available within the current ceiling; this does
not authorize treating corrupt memory as usable model storage.

```bash
python scripts/check_np101_capacity.py --dtype F16 --storage constant \
  --target-mib 1024 --readback none --output .cache/runs/capacity-allocation-bound
python scripts/check_np101_capacity.py --dtype F16 --storage constant \
  --target-mib 1024 --readback all --output .cache/runs/capacity-corruption-map
```

`none` still initializes every tensor, avoiding a measurement of unmaterialized
handles alone. `allocation_and_release_passed` records accepted payload and clean
teardown independently from `readback_and_release_passed`; skipped readback never
passes the latter. A rejected next block is recorded with its SDK error and
`rejection_phase`. No graph compilation/workspace is included.
For mutable tensors, creation of a handle may succeed before initialization
fails. The accepted payload counts only completely initialized blocks; an
`upload` rejection is reported separately from rejection in `AddTensor`. It is
not silently converted to a successful bare-handle allocation.

`all` writes one `readback-blocks.jsonl` row per retained block, including intact
blocks. It reports exact mismatched-byte counts, contiguous half-open byte ranges,
and all affected logical 4 KiB page intervals. Detailed byte ranges are capped at
256 per block with an explicit truncation flag; total counts and page coverage
are never truncated. Block indices and offsets start at zero. The cumulative
`payload_offset_bytes` is the sum of preceding payloads, **not a physical address**.
First-mismatch binary dumps remain available. The scanner does not rewrite tensors.

`full_scan_completed` requires complete coverage, consistent per-block totals and
normal release/exit. It may be true when integrity fails: a complete scan with
content errors still returns exit code 1 and `readback_mismatch`. Such a normal
diagnostic result permits the next explicitly requested comparison; SDK read errors,
signals, timeouts, recovery markers or failed release do not. `verified_bytes`
counts only wholly intact blocks, while `scanned_bytes` includes corrupt blocks.

Both allocation scripts accept `--binary`, `--sdk-lib`, `--output` and `--timeout`.
The capacity binary requires `REPORT_JSON constant|mutable F16|F32 TARGET_MIB`
and accepts an optional final `first|all|none` argument.
Rebuild it with the matching wrapper; old reports remain historical evidence and
cannot satisfy the new version-2 acceptance contract. Binary hashes live in
`execution-evidence.json`; a duplicate `binary.json` is no longer generated.

Shared native helpers are in `tests/native/np101/allocation_support.*`; shared
runner arguments and report acceptance are in `python/specferry/validation/allocation.py`.
SDK allocation/upload rejection is caught only around SDK calls. Pattern, file IO
or report-writing exceptions fail the experiment instead of becoming capacity bounds.
Reports distinguish `error`, `readback_error`, and `release_error`, retain the
original `failure_phase`, and require explicit `released=true` for acceptance.
An inconsistent report with an error and success counters is rejected.

The real-weight check uses the same comparison and teardown helpers. It preserves
the original weight/state allocation order and zero state initialization, then
reads states back after weights, even if a weight byte mismatch was found. State
upload and verification counts are separate from weight counts; both must cover
their expected totals. Zero-state readback is not a nonzero FP32 stress test:
use the synthetic F32 path for that comparison. Only the first mismatch is saved.

Host checks: `python -m unittest tests.python.test_capacity` and
`ctest --test-dir build -R allocation_readback_contract --output-on-failure`.
These cover dtype/storage and byte-count contracts, incomplete/contradictory
reports, release/process failure, short/long readback, page aliases, deterministic
finite patterns and CLI overrides without opening the device.

## Generic weight-pack readback

The weight checker consumes the common `deployment-manifest.json`, `weights.index`
and `weights.bin` format. Supply a directory explicitly; `--model` is retained as
an alias for `--deployment`. Generic verification checks file/tensor integrity,
shapes, offsets and aliases; it does not assert a model-specific architecture or
replace the stricter OPT/Qwen export validation.

```bash
# Host-only verification of an existing pack; no state buffers are implicit.
python scripts/check_np101_allocation.py --deployment .cache/np101/opt-350m \
  --prepare-only --output .cache/runs/opt-weight-check-prepared

# Explicitly reproduce the historical Qwen weight/state allocation after a relevant fix.
python scripts/check_np101_allocation.py --deployment .cache/np101/Qwen3.5-0.8B \
  --state-spec tests/fixtures/qwen3_5_allocation_states.txt \
  --output .cache/runs/qwen-weight-state-recheck
```

Without `--prepare-only`, the first command performs weight-only device readback.
Optional state files begin with `specferry-allocation-states 1`, followed by
`DTYPE SDK_SHAPE COPIES` rows. The historical fixture is now an explicit data file,
replacing automatic Qwen-specific generation; its shape/order and 46,071,808-byte
payload are preserved. No-state checks require all state counters to be zero.

`inputs.json` retains model identity/revision when supplied by the manifest,
manifest/pack hashes, a snapshot/hash of the optional state specification, and
expected physical weight count and payload totals. Acceptance compares the native
report with those requested totals, in addition to integrity and lifecycle checks.
It never counts a tied-weight alias as another physical tensor. A prepared or
successful readback result does not claim complete-model fit or hardware execution.

Capacity acceptance now shares explicit payload/boundary checks between its
allocation and integrity predicates; it never rewrites reported verification
counts or process exit codes to manufacture an intermediate passing report.

Refactor validation on 2026-09-23 is retained under
`.cache/runs/test-cleanup-20260923/`. Host preflight passed for the cached OPT pack
without states and the cached Qwen pack with the historical state fixture.
Two tiny device controls passed readback and release: 32 bytes of constant
FP16/FP32 weights without states, and the same mutable weights plus 272 bytes of
explicit state. The driver reference count returned to zero and no recovery
marker was created. These controls do not revalidate full-model loading or resolve
the large-allocation corruption below. Offline review of all ten existing
capacity-map records preserved their allocation/scan verdicts.

## New package preflight, 2026-09-22

The installed `ljmnp` package is 1.0.6. Its update note states that it corrects
NPU virtual-address mapping; it does not specify a new usable-memory capacity.
All 617 checked library/header/driver-source files match the supplied package.
The loaded module and installed DKMS module report srcversion
`8D46803809F15FA0FD98A92`. Live parameters still show `exclusiveSize=1073741824`
and `externalSize=2008023040`; these values alone cannot establish usable capacity.

The established convolution/ReLU/pooling preflight exited with SIGFPE before
producing a numerical result. The trace reports `FPE_INTDIV` at
`libNNArchPerf.so+0x21749`, inside `NNTransposeCycleCount_V9+0x1f9`, at an
`idiv %ecx` instruction. Without a core/register capture, division by zero and
integer quotient overflow cannot be distinguished. The performance library is
byte-identical to the one supplied in the new package; its unchanged hash alone
is not evidence of an incomplete installation.

The process group exited, with no timeout or residual children. This does not
prove driver health or reproduce the historical kernel hang. A recovery marker
was retained, stopping automatic follow-up device runs. The new capacity program
builds successfully and its host report-contract test and format checks pass;
those checks alone do not validate the device allocation path.

Evidence is retained under `.cache/runs/memory-driver-20260922-environment/` and
`.cache/runs/memory-driver-20260922-health-before/`. The latter includes the
signal trace, executable snapshot, disassembly, package checksums and SDK version
comparison. No large allocation was attempted during this failed preflight.

Static comparison of the convolution caller against `demo/main.c` found the
same shapes, convolution/pool parameters, virtual intermediates and initialized
FP16 weights. It does not overwrite `nn_param` or `pool.local`. Graph IO IDs are
copied by `vsi_nn_SetGraphInputs`, confirmed from the installed library. Differences
include extra HAL information queries and construction order. These observations
do not prove the caller is fault-free; a controlled original-demo comparison
remains needed before attributing the computation fault to the SDK alone.

## Historical FP16 allocation-only results

The user explicitly authorized an exception for this user-space SIGFPE: preserve
the incident, archive its recovery marker, start with 64 MiB of allocation/readback,
and continue only after success. Before archiving, the process group was absent
and the module reference count was zero. This did not declare the computation
fault resolved. The authorization and original marker are saved with the incident.

| Storage | Requested payload | Allocation/upload | Retained-byte check | Exit |
|---|---:|---|---|---|
| Constant | 64 MiB | All succeeded | All passed | Clean, explicit release |
| Constant | 1152 MiB | All succeeded | First failure at block 109 | Code 1; no residual process |
| Constant | 1024 MiB | All succeeded | Block 109: 12 FP16 values became zero | Clean, explicit release |
| Constant | 864 MiB | All succeeded | All passed | Clean, explicit release |
| Mutable | 64 MiB | All succeeded | All passed | Clean, explicit release |
| Mutable | 1152 MiB | All succeeded | Same failed block and bytes as constant | Clean, explicit release |

All targets retained 8 MiB blocks simultaneously. At 1152 MiB the SDK accepted
144 blocks totaling 1,207,959,552 bytes, exceeding 1 GiB. Acceptance alone did
not establish usable capacity: both storage modes failed retained-byte checks.
The first run threw on the mismatch and relied on RAII cleanup; subsequent runs
recorded explicit release after capturing the mismatch. No allocation-only run
timed out, received a signal or left residual processes.

In the saved constant-1024 and mutable-1152 runs, the failed 8 MiB blocks are
byte-identical. Block 109 uses zero-based numbering, after 872 MiB of earlier
blocks. Twelve FP16 values were zero instead of the supplied nonzero pattern:
three eight-byte spans starting at offsets 1,274,680, 1,274,744 and 1,274,808 within
the block. All other bytes of that captured block matched. This is a localized
data-integrity failure, not evidence that physical memory ends at that position.
No cause such as reserved-region overlap, mapping error, or asynchronous writes
has been established by this experiment.

The largest fully verified payload in this sweep was 864 MiB. It is a tested
lower bound, not the device's physical upper bound. The 4 GiB pressure probe was
not run because the earlier data-integrity requirement failed. Reserved memory
and total usable capacity therefore remain unmeasured. Full-model allocation
and new-package hardware acceptance remain open.

Evidence directories use the prefix `.cache/runs/memory-driver-20260922-` followed
by `constant-64`, `constant-1152`, `constant-1024`, `constant-864`, `mutable-64`, or
`mutable-1152`. `environment/experiment-summary.json` collects all six outcomes;
`constant-1024/mismatch-analysis.json` records the byte-level comparison.

## Vendor-command reproduction, 2026-09-22

At the user's request, the existing model allocation target was rebuilt against
the installed SDK and run without changes to its test logic:

```bash
python scripts/check_np101_allocation.py \
  --output .cache/runs/allocation-repro_0922_1 --timeout 600
```

This is the historical invocation. With the current generic runner, explicitly
supply `--deployment .cache/np101/Qwen3.5-0.8B` and
`--state-spec tests/fixtures/qwen3_5_allocation_states.txt` to select the same fixture.

All 320 logical Qwen text weights uploaded successfully (1,504,791,232 bytes).
With the state buffers, all 502 SDK tensors were allocated and initialized,
totaling 1,550,863,040 bytes (about 1.444 GiB). The tensor count and payload exactly
match the engineer's supplied report. The old allocation/upload boundary was
therefore crossed with the original model fixture, as well as the synthetic probe.

The subsequent retained-weight check failed at
`model.layers.17.mlp.gate_proj.weight`, offset 0, chunk size 7,340,032 bytes.
Earlier chunks totaling 912,175,872 bytes matched the export; this chunk did not.
The diagnostic reports `status=failed`, `phase=verify_weight` and exits with code 1.
It did not time out, receive a signal, or leave residual processes; the module
reference count returned to zero. This run does not save differing bytes, so it
does not establish that the mismatch has the same byte pattern or cause as the
synthetic probe.

The engineer's pasted JSON has the same schema as the historical allocation-only
test in commit `3746f4f`: success at `phase=release`, without retained-byte checks.
Commit `3c7f38b` added full weight readback and a `phase=complete` success condition.
Their exact source/binary revision is unknown, but the supplied report does not
demonstrate readback verification. Allocation success and the local readback
failure are therefore compatible observations, not conflicting capacity results.

Local evidence, including `allocation.json`, `sdk.log`, `driver.strace`, binary
snapshot and `execution-evidence.json`, is under
`.cache/runs/allocation-repro_0922_1/`. `comparison.json` distinguishes the
engineer's user-supplied report from the locally measured result. Data integrity,
physical residency, graph workspace and the near-4-GiB bound remain unverified.

## FP16/FP32 comparison with version-2 diagnostics, 2026-09-22

All four configurations use 8 MiB blocks and identical total byte budgets.
The source data contains no zero or nonfinite floating-point values.

| Dtype | Storage | 64 MiB control | 1152 MiB allocation/upload | 1152 MiB readback |
|---|---|---|---|---|
| FP16 | Constant | Passed | All succeeded | 12 FP16 values became zero |
| FP16 | Mutable | Passed | All succeeded | 12 FP16 values became zero |
| FP32 | Constant | Passed | All succeeded | 6 FP32 values became zero |
| FP32 | Mutable | Passed | All succeeded | 6 FP32 values became zero |

In every 1152 MiB run, the first failure was zero-based block 109, byte offset
1,274,680. Three eight-byte regions at offsets 1,274,680, 1,274,744 and 1,274,808
were zero instead of their expected data; all other elements in the captured
block matched. Each run verified the preceding 872 MiB before stopping readback.
The failure therefore affects both dtypes and both initialization paths. There
is no computation or precision conversion in these comparisons; they do not
identify whether the defect originates in upload, storage, mapping or readback.

The model allocation test passed both constant and mutable paths with the existing
small component fixture: 118,156 weight bytes and 2,624 state bytes, fully read back.
The complete Qwen constant run allocated all 502 tensors (1,550,863,040 bytes) and
again failed at `model.layers.17.mlp.gate_proj.weight`. Its first mismatch is now
captured precisely: tensor/block 208, chunk offset 0, byte offset 3,400,312, expected
byte 120, actual byte 0. All 46,071,808 state bytes passed the subsequent zero-value
readback. Unread weight chunks after the failure remain unverified.

All 11 experiments explicitly released their graphs and contexts, exited without
a signal or timeout, and left no residual process. Failed readbacks returned code
1 and were rejected by the wrappers. The final module reference count was zero;
no recovery marker was created. No 4 GiB probe or operator graph was run, and the
separate convolution SIGFPE remains unresolved.

Evidence: `.cache/runs/allocation-review-20260922/summary.json` aggregates all runs.
Subdirectories are `f16-constant-64`, `f16-mutable-64`, `f32-constant-64`,
`f32-mutable-64`, the corresponding four `*-1152` paths, `weights-small-constant`,
`weights-small-mutable`, and `weights-full-constant`. Failed runs include expected
and actual dumps and `mismatch-analysis.json`. The 64 MiB controls preceded a
teardown-reporting adjustment; the 1152 MiB and weight runs use the final binary,
which also saves `phase=release` before teardown. Allocation and pattern logic
were unchanged by that adjustment; each run retains its own executable snapshot.

Validation of the diagnostic implementation: full build, 73 Python tests,
3 CTest tests, Ruff and clang-format checks passed. Hardware failures above remain
failed memory checks; passing diagnostic unit tests does not resolve them.
