# Fixed-graph pipeline diagnostics

Use these checks to isolate the experimental OPT graph's zero outputs. They do
not change the default generator or prove full-model correctness. Device commands
use the shared diagnostic lock, timeout handling and recovery guard.

The implementation is `tests/native/np101/graph_pipeline_check.cpp`; the Python
runner only supplies paths, serializes device access and records execution evidence.

## Segmented-memory regression

All explicit tensor allocations now obey separate const/non-const 1024 MiB
budgets. WeightBank weights remain non-const; they share the budget with caches
and intermediate tensors. Aliases are counted once, temporary wrapper backing
is counted at its peak, and closed graphs return their reservations.
`memory-budget.json` must show both peaks within the limit and zero remaining
application payload after normal release. SDK internal allocation is excluded.

This runner enables complete byte-for-byte shared-weight checks in
`weight-readback.tsv`: before SetupGraph, after SetupGraph, after VerifyGraph,
and after the lookup/layer run or the prefix's first RunGraph. Each row identifies
the checkpoint tensor, chunk size/offset, mismatch count and first differing byte.
A mismatch stops the case; numerical output cannot override it. Readback staging
is bounded by one weight chunk and its comparison copy. These extra transfers
are diagnostic overhead, so these runs must not be used for throughput claims.
Other runners leave this optional audit off; historical reports are unchanged.

Follow-up order (currently blocked by the first gate's driver hang, recorded below):

| Order | Check | Total process budget | Advance only after |
|---|---|---:|---|
| 1 | Synthetic `lookup` | 120 s | All inputs/outputs, full weights and normal release pass |
| 2 | `graph_cache` stack and stack-block | 60 s each | History, causal reads, resets and release pass |
| 3 | Real lookup, then layer 0 | 60 / 120 s | Full weights, numerical stages and two-step KV pass |
| 4 | Prefix with capacity 16 and two teacher tokens | 360 s | All weight phases and prefix KV pass; partial prediction is diagnostic only |
| 5 | Full decode, then prefill | 360 / 600 s | Previous gate, CPU tokens/KV and normal release pass |

Do not repeat the old capacity-64 prefix as the first test. Prepare a matching
capacity-16 teacher fixture offline with `check_np101_generation.py --mode teacher
--layers 24 --capacity 16 --steps 2 --prepare-only`; use its `fixture` directory for
`--mode prefix --layers 1`. Inspect each stage before increasing the layer count.
The 2026-09-24 cold boot was confirmed, but the first test triggered a new recovery
guard. Do not continue the sequence or retry just by increasing the timeout.

## Segmented cold-boot attempt, 2026-09-24 11:00 CST

Boot ID `35df8e09-5d70-41b0-a89d-89b235dc509f` differs from the earlier incident;
`/dev/galcore` was readable/writable (0666). The old marker was copied to
`.cache/runs/segmented-regression-20260924/previous-recovery-marker.json` before
the runner cleared it on the changed boot. SDK library hashes match the previous
passing lookup. No driver/system settings or memory flags were changed.

The **first synthetic lookup did not complete**. Evidence is in
`.cache/runs/segmented-regression-20260924/lookup-small/`; the authorized kernel
capture is `../kernel.log`, and `../results.json` joins the observations.

| Observation | Result |
|---|---|
| Explicit const / non-const payload peak | 76 / 1020 bytes, both far below 1 GiB |
| All five retained weight chunks, 592 bytes | Exact before SetupGraph, after SetupGraph and after VerifyGraph |
| VerifyGraph | Returned SDK success after 22.372 s; submitting-thread ioctl took 22.271 s |
| First RunGraph | Entered at process 22.892 s; did not return before termination |
| Original process deadline | 30 s; SIGTERM at 30.089 s; process group exited |
| Numerical generation / normal SDK release | Not verified; no prediction or KV comparisons completed |

The kernel reports `[galcore]: NP[0] hang, automatic recovery.` at **11:00:51.004**,
then `recovery done` before VerifyGraph returns. It reports the same hang again at
**11:01:21.724**, after the process was terminated and `release galcore` was logged
at 11:00:58.136. A successful VerifyGraph status and an exited process therefore
do not establish device health. The interrupted 7.083 s RunGraph ioctl is not a
completed execution-time measurement. There were also 72 PAT notices; those alone
are not the hang diagnosis.

The former 30 s lookup deadline left insufficient room after the unexpectedly
slow verification. Its default is now 120 s for a future recovered-device run;
**no retry was made**, because kernel hang reports are a separate blocker. The
current recovery marker refers to this lookup, not the earlier prefix timeout.
Cache gates, real lookup/layer, prefix and full decode/prefill were not launched.
The independent CPU teacher fixture (capacity 16, two steps, all 24 layers) was
prepared successfully at `../teacher-reference/fixture`.

After recovery is confirmed, compare the previous passing tiny path with the new
complete-weight-readback path before increasing graph size. The added readback
schedule and device/SDK state remain candidate differences; this run establishes
neither a root cause nor a need for larger model memory.

## Checks and commands

Activate the `SpecFerry` Conda environment and build the diagnostic:

```bash
cmake -S . -B build
cmake --build build --target np101_graph_pipeline_check -j 4
```

Each command needs a fresh output directory. A recovery marker blocks device work;
the commands below do not bypass it.

`lookup` uses the production shared-weight bank, packed INT32 slices and blocked
embedding helper. It compares the first weight block before setup, after setup
and after execution; token/position/slot and raw embedding rows must match exactly.
The projected embedding uses the established FP16 tolerance, atol 0.08 and rtol 0.02.
It requires an IO fixture with `cases.txt` and `embedding.N.bin` CPU expectations.

```bash
# Offline fixture preparation; no device execution.
python scripts/check_np101_generation.py --mode selection --prepare-only \
  --output .cache/runs/pipeline-selection-reference

python scripts/check_np101_graph_pipeline.py --mode lookup \
  --deployment .cache/runs/pipeline-selection-reference/fixture/deployment \
  --fixture .cache/runs/pipeline-selection-reference/fixture \
  --output .cache/runs/pipeline-selection
```

For real weights, prepare an IO fixture with the existing
`check_np101_generation.py --mode io --prepare-only`, then pass its `fixture`
directory and the real deployment to the same `lookup` command.

`layer` directly uploads the first two CPU hidden states into one joint layer-0
graph: Q/K/V projection, per-head indexed append, attention and FFN. Only its final
output is declared as a graph output. Ordinary intermediate tensors and the
cache owner are read after execution for independent comparisons. It uses an
existing `check_np101_opt.py` fixture containing two steps and capacity at least 2:

```bash
python scripts/check_np101_graph_pipeline.py --mode layer \
  --deployment .cache/np101/opt-350m \
  --fixture .cache/runs/o4-opt-regression-20260924/fixture \
  --output .cache/runs/pipeline-layer --timeout 90
```

`prefix --layers N` directly uses `GraphModel`, including embedding and LM head,
but with a diagnostic layer prefix. It takes the first two tokens from a teacher
fixture and compares each prefix layer's KV with `teacher.1.layer.N.keys.bin` and
`values.bin`. A partial model's predicted token is printed only as an observation,
not accepted as a full-model output. Capacity comes from the fixture. The one-layer,
capacity-64 attempt below timed out in verification; do not repeat it as a quick
health check or extrapolate the small-layer timing to this combined graph.

## Results on 2026-09-24

Boot: `e9e10f53-b6d8-450f-ae02-233f9c2b5702`. Evidence is under `.cache/runs/`.

| Case | Result | Evidence directory |
|---|---|---|
| Synthetic lookup, all seven cases including block boundaries/tail | 38 checks pass, normal release, 0.72 s | `graph-pipeline-lookup-20260924` |
| Real OPT layer 0, two steps, capacity 8 | 30 checks pass, nonzero KV, normal release, 33.03 s | `graph-pipeline-layer-20260924` |
| Real 50,272-row vocabulary, tokens 2/133 at positions 0/1 | 13 checks pass, normal release, 5.42 s | `graph-pipeline-real-lookup-20260924` |
| Production graph constructor, one layer plus embedding/head, capacity 64 | `VerifyGraph` did not return within the 120 s process budget; no `RunGraph` or numerical comparisons | `graph-pipeline-prefix1-20260924` |

The real lookup reused independent CPU embeddings from
`o0-cold-32-20260923/teacher-fixture`; its derived IO fixture records that source in
`graph-pipeline-real-lookup-fixture-20260924/reference-source.json`. It allocates all
13 vocabulary blocks, but the two queried rows are both in the first block. Only
the synthetic case covers queried block boundaries.

The joint layer's maximum K error is 0.000977 and V error is 0.000122; the two final
layer-output errors are 0.007813 and 0.001953. The layer has 1,024 then 2,048 nonzero
K elements, and 1,019 then 2,042 nonzero V elements. These observations rule out
an always-broken indexed KV writer at the tested real head width (64) and capacity 8.
They do not prove complete-graph or different-capacity behavior.

The joint layer spent 28.76 s in `VerifyGraph`; its submitting-thread ioctl maximum
was 0.060 s. A worker-thread ioctl overlapped other work for 30.72 s, so it must not
be added to submitting-thread time or used alone to diagnose a driver hang.

The combined prefix spent over 110 s in `VerifyGraph`. At elapsed 81 s its process
was running at 97.2% CPU, with 78 s CPU time. It was terminated by the configured
deadline, not SIGSEGV. Its process group exited; kernel logs show device-file
release and no corresponding Oops/OOM/driver error. Normal SDK teardown was not
confirmed. A **new recovery marker is active** for this attempt; the earlier
successful health probe does not validate this subsequent incident.

The original complete graph used capacity 16; the prefix attempt used 64. This is
not a controlled layer-count comparison, and the new verification delay has not
been shown to share the original zero-output cause. No root cause is assigned.

## Why the PAT log count is large

Authorized kernel-log capture: `graph-pipeline-kernel-20260924.log`; per-process
counts and device open/release records: `graph-pipeline-log-summary-20260924.json`.

| Case | `AddTensor` calls | PAT notices |
|---|---:|---:|
| Synthetic lookup, no KV writer | 72 | 72 |
| Joint layer with KV | 532 | 533 |
| Real lookup, no KV writer | 272 | 246 |
| Combined prefix, never reached execution | 911 | 886 |

For the successful joint layer, the final PAT notice is at 10:02:39.185863; its
first computation is around 10:03:07. Thus the notices precede the first KV append.
The prefix also produces hundreds of notices before any computation. The counts
track tensor allocation/mapping scale approximately, not KV write count.

The message reports a write-back mapping request receiving write-combining memory
attributes. It is not a tensor numerical comparison. The current graph builder
materializes ordinary tensors even for intermediate values; the full graph's
13,055 `AddTensor` calls and 13,053 notices are consistent with that design creating
many mappings. This evidence does not prove that mapping attributes cause the
zero output, nor does it equate log counts with physical allocations or bytes.

## Header options and limits

| Interface | Installed definition | Consequence |
|---|---|---|
| `vsi_nn_tensorstackconcat_param` | SDK-owned `local` plus public `axis` only | No public append batching, cache policy or map-suppression option; preserve `local` |
| `vsi_nn_tensor_attr_t.vtl`, `vxCreateVirtualTensor2` | Graph-local intermediate objects; no external access/sharing | Candidate for pure internal intermediates; keep persistent KV, shared weights, host IO and diagnostic taps ordinary |
| `vsi_memory_type_e`, `vxCreateTensorFromHandle2` | Host, uncached host, DMA-BUF, physical/internal import types | These describe memory import/ownership contracts, not a validated switch for ordinary SDK allocations; no new import mode was enabled |
| `vxFlushHandle` | Applies to objects created from external handles | Current KV buffers are ordinary SDK tensors; this is not a general missing flush for them |
| Graph VIP/AXI SRAM preload attributes | Weight preloading size | No documented fix for persistent KV append or x86 PAT messages |

Read alongside `demo/README.md` and `demo/ref_op_api_guide.md`:

- `/usr/inc/acuity-ovxlib-dev/ops/vsi_nn_op_tensorstackconcat.h:44`
- `/usr/inc/acuity-ovxlib-dev/vsi_nn_tensor.h:131`
- `/usr/inc/acuity-ovxlib-dev/vsi_nn_types.h:136`
- `/usr/inc/VX/vx_khr_nn.h:477` (virtual objects), `:516` (import), `:535` (flush)
- `/usr/inc/VX/vx_khr_nn.h:50` (SRAM attributes)

The earlier investigation changed no SDK-owned state or memory-policy flags. The
current application payload cap does not modify the driver or tensor memory flags.

## Next bounded checks after recovery

1. Recheck complete shared-weight integrity under the segmented payload cap at
   construction/setup/verification/execution, starting with existing small gates.
2. Use the tiny synthetic table to combine lookup and LM head in one graph, then
   compare with separate graphs. This isolates shared-table multiple consumers
   without including real model weights or increasing layer count.
3. Keep capacity 16 and identical teacher tokens when comparing joint layer,
   embedding-plus-layer, and embedding-plus-layer-plus-head. Record entry/exit of
   each public SDK call; a completed call remains distinct from numerical success.
4. Only after those pass, increase the prefix from one to two/four layers, reading
   the first failing layer's input/QKV/cache. Reserve time for verification and
   normal teardown based on the matching configuration's observations.
5. Investigate graph-local virtual intermediates separately with the same fixture.
   Do not switch persistent state to virtual tensors or combine a mapping-policy
   experiment with the numerical diagnosis.
