# Validation

Run in Conda `SpecFerry` after [setup](../README.md). [Status](../docs/np101-status.md)
owns accepted configurations, precision and performance; [TODO](../TODO.md) owns pending work;
[incidents](../docs/np101-recovery-incidents.md) owns recovery gates. EP exploration is closed.
Repeat device checks only for a changed contract, observed failure or required acceptance gate.

## Offline checks

```sh
python -m unittest discover -s tests -t .
ctest --test-dir build/ep -L cpu --output-on-failure
python scripts/format_python.py --check
python scripts/format_cpp.py --check
```

Python covers source/package identity, generation policy/counts, numerical/state acceptance,
port permutations, memory budgets and device/recovery guards with mocked device calls.
The five native CTest cases have `unit;cpu` and target labels and never open NP101. The data
contract also checks that foreign-platform packages fail before driver inspection. Device tests
are explicit. Cross-builds produce test binaries but do not register them with host CTest.
The pinned x86 Verify tracing library is built only for EP.

## Device discipline

Read the ledger before every run. Runners serialize device ownership, preserve full kernel logs,
monitor faults and retain the recovery marker until agreed checks and delayed checks pass (SoC
10 seconds; EP 45 seconds). Correct output or automatic recovery does not clear a fault. Recovery requires a
user-confirmed NP101 board power cycle; a host cold restart or changed boot ID is unnecessary.
Read-only `sudo journalctl` has standing authorization (`--sudo-read-only`); other sudo actions
require approval. Do not restore deleted convolution controls or silently substitute vendor/old OPT.
Const/nonconst application payloads each stay ≤1 GiB, including peaks and shared backing once.
SoC OPT formal runs have a preparation budget of at most 240 seconds and a separate post-ready
workload budget of at most 300 seconds. Readiness is reported once through a private pipe, independently
of the sampler. Legacy/unphased diagnostics and EP retain the 300-second total limit. Never extend
a running deadline; preserve the gate on expiry and change the approach. Delayed checks are separate.
The shorter SoC observation does not rule out the recorded recovery Oops after about 33 seconds;
every subsequent run must still read the complete kernel log.

`--prepare-only` prepares fixtures without opening the board. Recovery flags require prior agreement
and do not themselves authorize a control. Archived binaries/commands are evidence, not test entries.

### SoC formal generation

Use the separate `np101_soc.py prepare/deploy/run/collect` workflow in the root README. Packaging
checks source weights, NBG ports and SDK identities; accepted board runs use explicit C64/C128 mapped-slot
requests with M1 or packaged M16 prefill and the documented driver parameters. Reuse a CPU reference when
the request is unchanged. Host numerical checks share the OPT validation code with EP.
`python -m unittest tests.python.test_soc` checks packaging integrity, architecture/path rejection,
watchdog termination despite blocked sampling, fault handling and retained recovery gates without
opening a device. SSH failure alone never authorizes another model submission.
Qwen SoC packaging is selected with `np101_soc.py prepare --model qwen`; its C64 or C128 single-token
request reuses the source-bound EP weight/NBG package and complete CPU oracle. The shared supervisor
has explicit model argument ordering and requires the selected executable in the verified inventory.
`python -m unittest tests.python.test_qwen_soc` covers model routing, policy/precision rejection,
unwritten KV and nonfinite state rejection, and required SoC identity/composition. EP and SoC share
Qwen's numerical acceptance function; packaging and cross-compilation alone are not device evidence.
Qwen C128 may explicitly select `--qwen-ffn-batch 8|16` with mapped KV and shared-FCL
projections. These packages contain all 24 fixed-weight FFNs at one common width: prefill processes
real rows in blocks, while decode pads a single row into the same graph. They do not retain a second
M1 FFN package. The default remains M1. Acceptance checks the declared composition, exact output
tokens, final hidden/representative state and KV, unwritten slots and release; a divisible prompt does
not cover a partial prefill block or repeated reset. See the status document for measured scope.
`--qwen-decoder-layout fused-attention-norm` merges input RMSNorm into attention QKV;
`shared-residual` additionally binds preceding FFN output blocks as residual inputs and omits
unconsumed scalar publications. Both require the same mapped/shared-FCL batch composition;
`separate` remains the default. Request, bundle and initialization layout records must agree.
`--qwen-state-commit ffn` is an experimental M8/M16 candidate: fixed state copies run in
the FFN wrapper on the final real prefill row and every decode step; intermediate rows retain
their local commit graph. `combined` remains the default. Contract tests reject M1 and
policy/metadata mismatches. Keep the imported FFN wrapper's original graph IO: promoting
state copies to extra graph ports corrupted FFN output on the pinned SoC SDK. Ordinary retained
copy tensors preserve the bindings. The repaired M8 96-input/16-output run matched baseline tokens
and captured hidden/state/KV bitwise, with clean release/kernel checks and about 3.9% higher actual
output throughput. The latest M8 combination also passed the frozen two-turn reset/replay conversation
(28/66 inputs, 20/49 outputs, both EOS), covering partial prefill blocks. M16 has a single-request
comparison only; its tail/reset behavior and long-term stability are not established.
Independent shared-state DeltaNet candidates bind M4/M8 outputs to the original M1 next-state
buffers, keeping canonical inputs distinct and the FFN graph IO unchanged. Both selected layers
passed 96 rows, M1 continuation, reset, 8+8+2 rows and M1 again; before/after controls were bitwise
identical. Two-layer whole-model candidates passed the frozen 96-input/16-output CPU oracle and
health/release gates. Existing pipeline tolerances pass; strict recurrence differences remain recorded.
These are archived candidates, not default deployment options. Their full-model multi-turn/reset
acceptance and 18-layer capacity are not established; do not substitute the baseline's conversation.
Shared prefill acceptance additionally checks the declared input-cache policy and reduced transfer
counts; older records retain independent-cache accounting. Current shared-input hardware evidence
covers C128 long/short reset requests; capacity and stability limits remain in the status document.
SoC compiled prefill additionally binds all decoder K/V outputs to one fixed scratch pair,
committing each layer before reuse. Input/output and duplicate-output aliases are rejected before
node creation. The original C128 memory acceptance reused the CPU causal prefix for two
17-input/two-output reset requests and 58 checks. The latest combination with 8192-row SoC embedding
chunks also passed 97/17-input reset requests, 16 outputs each, with 56 token/KV/unused-slot and
execution contracts plus release and delayed logs. Different prompts do not require repeated-prompt
equality checks. A separate real/boundary-input comparison confirmed bit-exact embedding outputs
between 4096/8192-row chunks. EP output ownership and embedding policy are unchanged.
The subsequent C128 boundary collection uses 113/17/113/17-input reset requests with 16 outputs
each: 82 checks cover full-capacity rejection without IO, repeated KV equality and unused slots,
plus release and delayed health checks. This bounded collection does not establish long-term service
stability. C256 also passed a 241/17/241/17-input collection, sixteen outputs each, after confirmed
restart and recovery controls: 82 checks cover full-capacity rejection with zero extra IO, three
resets, repeated final-layer KV equality and unused slots. Original SDHCI/hang incidents remain;
this bounded acceptance does not establish a driver fix. Do not treat storage faults as isolated
NP hangs.

SoC prefill compiles the ordinary FFNs in the running process. Historical FFN Verify-cache
bundles are archived with their original runner and executable. FP16 KV regressions cover port
precision, the shared prefill state and byte accounting.

The retired standalone KV transfer experiment is archived outside the source tree. Its maintained
replacement is the formal OPT entry: one resident Context processes long/short/repeated-long reset
requests, checking tokens, coarse final-layer KV, unused slots, repeated-state equality and release.
Performance comparisons use matching input/output lengths; the short isolation request is separate.

### Small SoC memory check

With the [cross-build setup](../README.md#soc-cross-build), configure `BUILD_TESTING=ON` and build
only target `np101_memory_check`. This explicit device diagnostic is not registered with CTest.
It takes an output directory, allocates FP32 tensors of 1, 4 and 16MiB for each of const/nonconst
(21MiB each), and checks all retained contents through mapped reads after each allocation.
It adds no operators and calls neither Setup, Verify nor RunGraph; passing is not compute acceptance.
Graph/Context release must return application budgets to zero. Use an external bounded supervisor
and the full ledger checks; this binary has no watchdog. Accepted 256/1792MiB pool runs used a
90-second deadline and checked pool release and delayed logs; measurements live in the status
document. This 42MiB check is not whole-pool integrity or model acceptance.

### Small compiled-graph compatibility check

Build `np101_compiled_graph_check` with the SDK matching the board. This explicit device test is
not registered with CTest. Run `np101_compiled_graph_check FIXTURE OUTPUT` for one pass through
the existing one-to-three reference cases; append `12` only for a bounded twelve-call cycle of
three input cases. The fixture uses the archived `contract.txt`, `model.nb` and `caseN` binary
inputs/references. Contracts and complete fixture sizes are checked before Context creation.
NBG constants/files are limited to 32MiB, ports to 1MiB, and the graph has one imported NBG node.

Every call uploads the selected inputs to fixed independent tensors, runs synchronously, then
maps/unmaps and checks all outputs. There is no unchanged-input/no-readback replay mode, asynchronous
submission, additional copy graph or tensor aliasing. Each RunGraph has a flushed begin/end record;
numerical rows survive a later failure. Healthy completion explicitly closes Graph then Context
and requires zero application live bytes. The external supervisor must enforce the deadline,
kernel/owner/pool/GFP gates and delayed observation; this binary has no watchdog and cannot clear
a recovery marker. Report graph, explicit IO, preparation and release separately; the diagnostic
trace and reference comparisons do not establish model output throughput.

## Generation and compiled packages

Recommended commands and package hashes are in [README](../README.md) and the status document.

| Entry | Maintained purpose |
|---|---|
| `generate_opt.py` | Complete component baseline or `--backend graph`: merged KV, split fc1, compiled FFN, decoder or fullgraph packages |
| `generate_qwen.py` | Cached complete C64 model, conservative defaults and explicit shared FCL head / combined state commit |
| `check_np101_compiled.py --plan PLAN --case CASE` | Pinned binary, request/reference, SDK, ports, budgets, repetitions and stop reason |
| `check_np101_model.py` | OPT teacher/decoder, selection and IO isolation |
| `check_np101.py` | Lookup, projection/mapped-readback, sampling, KV/cache append and FP16 preservation contracts |

OPT graph defaults are block1, every-token head, greedy and token-only readback. Blocks4/16 batch
known prompt tokens; tails/decode use M1. Block8 model execution is unvalidated. `last-token` skips
unused prompt heads but updates every decoder/KV. `--request-tokens-file` supplies a JSON request list.
Sampling requires explicit overrides and the component backend's validated FP32-logit path.

`--benchmark --warmups N --repeats N` measures repeated requests with a resident model; warmups are
validated and excluded. `--cache-readback logits|kv` enables coarse diagnostics; KV ranges are
one-based inclusive. `--readback-transport mapped` never silently falls back to SDK copy. Lifecycle
rebuild releases the last old Context first; v105 Contexts share the process shader-slot table.
Native OPT takes `DEPLOYMENT REQUEST OUTPUT`; unique settings live in `run-options.txt`.
Unknown, duplicate and incompatible settings fail before SDK access.

Qwen keeps GATHER_ND, zero-based layer9 device MatMul FFN, 18 DeltaNet + 23 FFN NBGs and mapped
selected-token readback. Full shared FCL head reuses all 61 embedding blocks without extra weights.
The optimized mixer package adds FP32 broadcast outer products and fused input RMSNorm; use it with
`--head-backend shared-fcl --state-commit combined`. Version-2 manifests bind the norm weight;
version-1 packages retain external norm. New/old recurrent states remain distinct, pending commits
block the next token/reset/read, and consumers close before state owners. Qwen C128 is unvalidated.

## Offline exporters

The compiler runs in a device-free rootfs. Packages contain NBGs plus source/SDK/budget/port manifests;
ONNX, generated C and compiler logs stay outside runtime packages.

| Exporter | Output |
|---|---|
| `export_np101_opt_ffn.py` | 24 M1 fused FFN/final-norm NBGs; optional head/projection artifacts, existing-artifact revalidation |
| `export_np101_opt_projections.py` | 24 QKV/out graphs and one head, with public/SDK port permutations |
| `export_np101_opt_model.py` | `--scope {decoders,fullgraph} --capacity {64,128,256}`, FP32 KV and optional runtime package; C256 awaits board acceptance. `--kv-update select` preserves ports; fullgraph-only `narrow-select` narrows before selection and emits FP32 current slots, requiring mapped-slot commits |
| `export_np101_qwen_blocks.py` | `--part mixer|feed-forward`, one/all layers, optional runtime package and combined mixer/FFN package |

OPT's `--prefill-artifacts` imports existing source-bound M16 exports into the M1 package: layers
0–3 retain ordinary FFNs, layers 4–23 use M16 NBGs, and both execution sets share one head. This
requires the compiled head; acceptance covers C64/block16 and explicitly declared C256 packages
with 241/3/241-input requests, 16 outputs each, capacity rejection and repeated KV checks.
The C256 conversation comparison is recorded separately below. Full M16 const residency exceeds the
budget. M16 uses ordinary attention; QKV/out NBGs remain M1 graphs. Numerical, KV/reset,
capacity and release evidence is in the status document.

`benchmark_opt_conversation.py --package PACKAGE --output NEW_DIRECTORY --history-mode reuse|reset`
runs three turns in one resident NBG/M16 model; use separate processes for the two modes. It requires
a package explicitly declaring C256
for the fixed M1/M16 ports; original C64 packages remain unchanged. Three English turns use a
pinned CPU greedy trajectory, last-layer mapped KV checks and the normal release/kernel gates.
Conversation response metrics add `state_preparation_seconds` (including reset) to generation
timings; initialization, tokenization, CPU reference, diagnostics and user think time are separate.
The retained path consumes the pending last output once. Full-prefix validation precedes optional
partial-block replay, selected using measured headless M1/M16 costs including control IO.
`replayed_prompt_tokens` records recomputation. The old six-turn `compare` mode is retired;
use independent `reuse` and `reset` runs, under the respective platform's deadline policy.

Qwen feed-forward export accepts `--batch-tokens 1|4|8|16`; package contracts bind every layer's
original weights, port shapes and common width. Qwen mixer export accepts `--outer-product multiply`,
`--fuse-input-norm`, and the opt-in
`--joint-state-reads` (shared FP32 K/Q state product with an algebraic rank-one output update).
The latter preserves state precision and public ports; the default recurrence remains unchanged. OPT's
`--projection-backend fcl|nbg-qkv|nbg-output|nbg` selects explicit implementations. Packages with
QKV/out artifacts default to NBG; older head-only packages retain FCL. Direct FCL experiments are
stopped. Python/native loaders reject changed identities before Context creation. Generated
ArgMax intermediate precision and re-saved public port order must match the accepted package contract;
see the status document. `reference_dlm.py` and export verification remain CPU-only references.

## Measurement and retired experiments

Report actual generated output count / generation wall separately from post-first-token decode rate.
Exclude preparation, reset, diagnostics and file output; include required token IO and any runtime
Verify. Input processing and operator replay are not generation throughput. CPU time overlaps wall.
`--sdk-summary` aggregates in memory; `--sdk-timing` records calls. New instrumentation needs the same
workload off/on/off; VPD collection has substantial measured perturbation and no operator timeline.

For SoC hang localization, retain both abnormally long completed calls and unmatched begins;
the final graph entered can follow an earlier stalled call. Formal supervision forces SDK timing
off; enable the existing calls mode only in an explicit diagnostic deployment, never by silently
changing the benchmark default. A sysfs `stuckDump=1` readback is not proof of effective dumping:
the current driver copies options during construction and keeps internal dumping disabled when
automatic recovery is enabled. Disabling recovery can stop the driver to preserve the scene;
follow the incident ledger before changing that policy.

Completed allocation stress/integrity/accounting/growth, packed-cache matrices, Qwen layer-slice and
handwritten recurrent diagnostics, vocabulary-prefix NBG and vendor diagnostic commands are retired.
Complete inference, optimized configurations and necessary contract regressions remain. Immutable
source snapshots, HEAD identity and retired paths are under ignored
`.cache/runs/repository-compaction-20261016/`; historical measurements and incident evidence remain.
