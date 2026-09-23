# Complete OPT generation and timing checks

Use the `SpecFerry` Conda environment, a current build and a verified OPT-350M
export. Run one device experiment at a time, with fresh output directories.

```bash
cmake --build build -j 4
python scripts/check_np101_acceptance.py --lengths 32 \
  --output .cache/runs/opt-short --diagnostic
python scripts/check_np101_acceptance.py --lengths 128 512 2048 \
  --warmups 0 --repeats 1 --cycles 1 --timeout 7200 \
  --output .cache/runs/opt-contexts --diagnostic
```

`--prepare-only` prepares all CPU references and tokenized requests without opening
the board. An execution command prepares its own fixtures; use a fresh directory.
`--model`, `--checkpoint`, `--binary`, `--io-binary`, `--kv-binary`, `--sdk-lib` and
`--shader-header` override the corresponding local paths.

## Functional gates

The runner uses the full 24-layer model, original FP16 weights and the checkpoint's
default greedy policy. It runs these gates before performance measurements:

1. Exact selection/embedding-sharing fixtures.
2. A natural prompt followed by CPU-predicted tokens, crossing prefill and decode.
   Every selected checkpoint compares all layer outputs and valid K/V prefixes
   against the official CPU model. Historical KV bytes must remain identical after
   append. Reset/replay and a fresh model are checked independently.
3. Capacity-eight teacher execution: last slot, full-capacity prediction, invalid
   token/capacity rejection without changing state, reset and recreation.
4. Shared KV storage at capacities 8, 64 and every requested generation capacity:
   first/adjacent/last slots, overwrite, fixed-reader visibility without prior parent
   readback, and a final-only readback pass. Unused slots and bounds are checked.
5. A → B → A generation in one full-model instance, with different prompt lengths
   and contents. Each request is compared with a fresh independent CPU reference;
   B must mask A's old suffix and the second A must restart positions correctly.

Diagnostic KV/intermediate readbacks are excluded from timed generation. Measured
requests reuse production `Model::generate`; no alternate model arithmetic exists.
They check 104 uploaded control bytes and 24 KV writes per consumed token, and
four readback bytes per predicted ID. The final predicted token has not yet been
consumed, so it is not counted in KV length. Normal release and process exit are
required. SDK-internal transfers and execution engines remain unproven.

## Prompts and repetition

`--lengths` takes exact input-token counts, including the tokenizer's BOS. It uses
continuous prose in `fixtures/opt_continuation.txt`, with no random IDs or padding.
Truncation may end mid-sentence or mid-subword. Full input text and IDs are saved.
Capacity is at least `--capacity` and includes requested decode space, up to 2048.
OPT's learned-position limit means **2048 input tokens permit only one prediction**.
That case tests full prefill and the boundary, but has no decode interval. Use
`--lengths 2033 --max-new-tokens 16` for a near-limit continuation.

Without `--lengths`, repeatable `--prompt TEXT` selects inputs; defaults are two
short English prompts. Default capacity is 64, generation limit 16, one warmup,
three measured requests and two fresh model processes per prompt. `--steps` caps
the available natural teacher trajectory snapshots (default 8, allowed 1..64).
The capacity-eight boundary gate is separate and always uses eight steps.

`--capacity` accepts 1..2048, `--max-new-tokens` 1..512, `--warmups` 0..5,
`--repeats` 1..20 and `--cycles` 1..5. A single measured request is an exploratory
observation, not a stable throughput distribution. The longer-prefix command above
uses one request per fresh process to bound known SDK host-allocation growth.
Before execution, available host memory must exceed an estimate of 4 MiB per
consumed token plus a 4 GiB reserve; this estimate is not a proven memory bound.

Any failed correctness gate, transfer check, signal, timeout or incomplete release
stops the suite. The existing device lock and recovery guard remain active.
A byte-correct SDK run alone does not establish exclusive NPU execution.

## Performance and readable output

`results.md` shows decode **tokens/s**, prefill tokens/s, model-ready TTFT, prompt
endings and generated text. `acceptance.json` and per-process `benchmark.json`
retain all measurements, tokens, checks and host RSS observations.

| Metric | Boundary |
|---|---|
| Initialization | Context/model construction, excluding host weight-file verification |
| Prefill | Sequential consumption of prompt IDs; excludes the first head invocation |
| TTFT | Reset completed and IDs ready through first prediction; includes prefill |
| Decode tokens/s | Number of post-first-token intervals divided by their summed seconds |
| TPOT | Mean post-first-token interval; inverse of decode tokens/s |
| Request output tokens/s | All output tokens divided by request time, including prefill |
| Cache time | Sum of slot writes, with revalidation time reported as a subset |

N output tokens give N−1 decode intervals; N=1 reports null decode throughput.
Tokenization, network/display, reset, initialization and warmups are excluded from
request timing. Prefill is currently sequential, not a batched implementation.
Measurements run without strace, streaming callbacks or tensor diagnostics and
with target printing disabled; active memory/kernel profiling is rejected.
Counters added for cache timing use the existing steady-clock measurements.

CPU outputs are prepared before device measurements. Failed or incomplete runs
retain raw timing and text but are excluded from accepted metrics. Medians/ranges
and sample counts accompany timing distributions; a few samples do not establish
robust tail latency or steady-state behavior. Host RSS includes allocator retention
and is not physical board occupancy. Known weight/KV payload and unconfirmed
physical peak/layout/workspace fields are kept separate.

## Reports and exit codes

- `suite.json`: requested matrix, policies, references and exact prompts.
- `selection/`, `teacher/`, `boundary/`, `kv-N/`, `reset/`: correctness and lifecycle evidence.
- `cycle-N-request-M/`: native outputs, every warmup/measured generation, host RSS,
  binary/library identity, process evidence and benchmark validation.
- `results.md`: human-readable throughput table and continuations.
- `acceptance.json`: numerical results, application timing and deferred proof gates.

Exit 0 means successful preparation or numerical success with `--diagnostic`.
Exit 1 means a failed/incomplete check. Without `--diagnostic`, numerical success
returns 2 / `hardware_pending`: backend, physical residency and SDK-internal memory
behavior have not been established. `application_timing_measured` and
`kv_and_reset_pass` are separate from the hardware-proof flags. Deferred profiler
work does not block valid host-wall observations. Long-running stability and
context compression remain future work; bounded longer-prefix observations do not
close those gates. See [the plan](../docs/remaining-acceptance-plan.md) and
[three separate memory diagnostics](np101-memory.md).

## Validation record

Earlier SDK numerical results remain in [the generation record](np101-generation.md); they do not replace
revalidation after a driver or library change.

On 2026-09-22 with package 1.0.6, the exact selection preflight and 24-layer,
capacity-512, eight-step teacher suite pass (23 and 491 checks respectively).
The new runner then passes two prompts × two fresh model processes, with one
warmup and two measured four-token continuations per process. All 48 returned
token IDs across 12 requests match CPU; those requests consume 120 tokens.
Application transfer totals, reset, release and process exit all pass. The suite
returns 2 / `hardware_pending`, as intended.

Unprofiled host-wall observations, with only four measured requests per prompt:

| Prompt | Prompt tokens | First-token median, two fresh processes | Decode-interval median, two fresh processes |
|---|---:|---|---|
| `The capital of France is` | 6 | 9.100 s / 9.210 s | 1.621 s / 1.614 s |
| `For dinner tonight, I will cook` | 8 | 12.198 s / 12.155 s | 1.605 s / 1.610 s |

Initialization ranges from 6.078 to 6.719 seconds. Observed decode throughput is
0.615–0.624 tokens/s. These short runs do not establish NPU throughput, long-context
performance or steady-state memory use. In fact, host RSS grows by about 3 MiB per
consumed token: 56,627,200 bytes over the two measured France requests and
69,214,208 bytes over the two dinner requests, repeated in both fresh processes.
This is tracked as `NP101-MEM-003` rather than declared a proven device leak.

A separate HAL accounting investigation used the declared/exported
`gcoOS_GetMemoryProfileInfo` from `HAL/gc_hal_base.h`. It was kept in local cached
diagnostic source, outside the production runtime. The need is allocation
observability, which the operator guide does not supply. The probe first checks a
small tensor, uses nonnull output storage, and never adopts the API as a default
inference path:

- With profiling disabled, status is success but the entire 80-byte output structure
  retains a sentinel pattern. Zero-initializing it and trusting success would report
  fabricated zero usage.
- With `VIV_MEMORY_PROFILE=1`, an 8 MiB constant tensor passes byte readback; SDK
  `gpu_memory.currentSize` rises by 16 MiB + 4 KiB and falls after graph release.
  Thus this count cannot be equated directly with unique tensor payload or physical
  resident bytes. The exit dump reaches zero in this control.
- In a separate full-model profiled run, all three nine-token requests return the
  four CPU-expected IDs. `gpu_memory.currentSize` stays at 1,454,438,928 bytes
  through the requests; its peak is 1,462,554,944 bytes and the final exit dump
  reports zero outstanding GPU-accounted bytes.
- `system_memory.currentSize` increases by 28,313,280 bytes per repeated request.
  Even the SDK exit dump reports 84,939,840 bytes and 1,296 allocations outstanding
  after 27 consumed tokens: 3,145,920 bytes and 48 allocations per token. This
  supports investigating retained/unreleased SDK host allocations; the originating
  call and ownership error are not yet identified. The per-slot KV graph
  re-verification path is a candidate, not an established cause.

Profiled observations are excluded from timing distributions. The benchmark runner
also rejects an active `VIV_MEMORY_PROFILE`. The chip team still needs to confirm
counter semantics, physical pool accounting and per-kernel backend evidence.

Local evidence (ignored by Git):

- `.cache/runs/acceptance-20260922-environment/`: host/SDK identity.
- `.cache/runs/acceptance-20260922-selection/` and
  `.cache/runs/acceptance-20260922-model/`: preliminary independent SDK health gates.
- `.cache/runs/acceptance-20260922-suite/`: complete suite, per-process snapshots,
  CPU references and raw timing/resource records. `acceptance-review.json`
  re-evaluates retained records with the final host-RSS-growth summary checks.
- `.cache/runs/acceptance-20260922-memory-profile/`: cached probe sources/binaries,
  8 MiB control, full-model counter observations and SDK exit dumps.

No sudo, driver modification or full-Qwen allocation retry was used. The separate
convolution SIGFPE and large-allocation data corruption remain unresolved; they
were not reproduced again during these OPT checks. Full formal acceptance stays
open for hardware evidence, memory-growth diagnosis and long-context validation.
Long-context/stress acceptance is deferred until the per-token allocation growth
is explained or corrected. The final build, 81 Python tests, three CTests and
Ruff/clang-format checks pass; these host checks do not close the hardware gates.

## Expanded short-prefix validation, 2026-09-23

The completed short suite is in `.cache/runs/acceptance-20260923-short/`:

- Exact selection gate passes. The natural eight-step, 24-layer teacher trajectory
  and capacity-eight boundary each pass 2,027 checks, now including every layer's
  valid K/V prefix and exact preservation of history. The natural trajectory
  includes six prompt tokens and two consumed CPU-predicted tokens.
- Shared KV tests pass at capacities 8 and 64, including fixed-reader visibility,
  final-only readback, overwrite and out-of-bounds rejection.
- A/B/A in one resident model returns the CPU-expected 16 predictions for each
  request, with 81 consumed tokens and 1,944 cache writes. Reset masks the old suffix.
- The exact 32-token natural prefix passes two fresh model processes, each with
  one warmup and three measured 16-token continuations. All 128 returned IDs across
  these eight requests match CPU, and all transfer/release checks pass.

| Fresh process | Decode tokens/s | Prefill tokens/s | TTFT median | Measured requests |
|---|---:|---:|---:|---:|
| 0 | 0.6648 | 0.7102 | 45.157 s | 3 |
| 1 | 0.6633 | 0.7106 | 45.117 s | 3 |

These preliminary unprofiled host-wall observations use sequential prefill. Some
CPU reference preparation for other prompt lengths overlapped this first run;
an uncontended short repeat was deferred when the user cancelled the later
2048-token case and the device recovery guard became active. The user subsequently
cold-started the board; the completed O0 repeat is recorded separately in
[SDK latency diagnostics](np101-sdk-timing.md), without rewriting this earlier evidence.
For process 0,
three measured requests take 203.168 s in total; KV writes account for 4.828 s,
including 4.489 s in revalidation (about 2.4% and 2.2% of request time). Thus the
known rebinding/reverification issue does not explain most of this observed latency.
The remaining time is elsewhere in the model execution path; execution-engine
attribution remains deferred. Do not infer a backend from these timings.

Example continuation (identical in every request):

```text
Prompt ending: Every morning, Anna
Output:  and her husband, John, would walk to the library, where they would read
```

This is readable English continuation, not an instruction-following/chat evaluation.
Host RSS still grows by about 3 MiB per consumed token (443,584,512 bytes over 141
measured consumed tokens in process 0). MEM-003 remains open. Numerical, KV/reset
and application timing results pass; strict status remains `hardware_pending`.
Longer-prefix observations are recorded separately below.

## Bounded natural-prefix observations, 2026-09-23

Evidence: `.cache/runs/acceptance-20260923-context-final/`. All CPU references were
prepared before these measurements. Each input uses a fresh full model, no warmup
and one measured request. These are exploratory single samples, not distributions.
The native application build is `RelWithDebInfo` (`-O2 -g -DNDEBUG`).
Cache capacity follows the requested input/output space, so these cases vary both
prompt length and static graph dimensions; they do not isolate valid-prefix length.
The same KV, teacher, boundary and A/B/A gates pass before timing; shared-cache
checks additionally pass at capacities 143, 527 and 2048.

| Input tokens | KV capacity | Output tokens | Decode tokens/s | TTFT | CPU token agreement |
|---|---:|---:|---:|---:|---|
| 128 | 143 | 16 | 0.6520 | 184.131 s | Exact |
| 512 | 527 | 16 | 0.5974 | 807.523 s | Exact |
| 2033 | 2048 | 16 | 0.5003 | 3847.554 s | Exact |

The user cancelled the 2048-token measurement during prefill after approximately
52 minutes. Its process group exited after SIGTERM; no completed prediction,
TTFT or decode throughput is claimed for that case. The raw suite status is
`failed` because it includes this interrupted request, not because the three
completed requests disagreed with CPU. Their independent reports and timings
remain valid. `cancellation.json` records the user-directed stop alongside the
unchanged raw evidence. The recovery marker remained in place until the user's
subsequent cold start; process exit alone did not establish recovery. O0's new-boot
checks then passed. The cancelled request remains incomplete.

OPT's 2048 learned positions
permit one prediction after consuming a full 2048-token prompt, but no additional
decode step. The separate 2033-token input leaves room for 16 predictions, subject
to EOS. The final predicted token does not enter KV until it is consumed.

Observed continuations:

```text
128-token prompt ending: why the old bridge had been built so far
Output:  from the village. Anna had been a teacher for many years, and she knew

512-token prompt ending: the weather, the birds, and the work of the mill
Output: .

The woman looked at the notebooks and said, “I�

2033-token prompt ending: Daniel wrote that point on the classroom board
Output: .

The next day, the students went to the riverbank to check
```

The 16-token cap may stop mid-sentence or within a UTF-8 character assembled from
multiple byte-level tokens. The replacement character above also appears in the
CPU continuation; the saved token IDs agree exactly. Preserve the actual output
rather than editing it into a more fluent response.

The 2033-token request consumes 2048 tokens and records 49,152 KV writes and
212,992 uploaded control bytes, plus 64 readback bytes for its 16 predicted IDs.
Transfer counts, byte bounds, normal release and process exit pass. Peak host RSS
is 6,697,212 KiB; growth across the request is 6,435,086,336 bytes, again about
3 MiB per consumed token. KV writes take 77.639 s out of 3877.539 s of request
time (about 2.0%), including 71.627 s in revalidation. The slow end-to-end model
path is not explained primarily by these measured KV writes.

Code inspection counts 50 compute-graph executions and 24 KV copy-graph executions
per consumed token, plus two graph executions per prediction (output projection
and vocabulary head). These are application API calls, not kernel or DMA counts.
The next latency investigation should separate graph execution, synchronization
and any SDK-internal transfers before selecting an optimization; these counts
alone do not establish which part is slow.

Final host checks pass: the complete CMake build, 91 Python unit tests, three
CPU-only CTests, Ruff and clang-format. Offline re-evaluation of the saved teacher
and boundary outputs also passes with the final stricter cache comparator; this
does not rerun the device or establish recovery.
