# OPT input/output and resident generation validation

The native implementation covers OPT-350M token lookup, learned positions,
24 decoder layers, single-buffer KV updates, output projection, tied LM head and
greedy token selection, with optional [categorical sampling](np101-sampling.md).
Python handles tokenization, independent references,
process supervision and text display. No model arithmetic or cache copy is
performed by the Python generation wrapper.

## Implementation

- `native/np101/ops/vocabulary.*` owns one FP16 table in bounded row blocks.
  Lookup and head graphs retain the same ordinary SDK tensor handles. The table
  is mutable in the SDK API but application-immutable after initialization.
- Each block has at most 4,096 rows; OPT has 12 full blocks and a 1,120-row tail.
  Host dispatch selects the lookup block and uploads its local index. GATHER
  performs the lookup. It never uploads an embedding vector or streams weights.
- `native/models/opt/model.*` adds `project_in` and learned position row
  `consumed + 2`, binds that output directly into the decoder, and applies
  `project_out` after layer 23. OPT-350M needs no additional decoder final norm.
- Each head block uses MatMul, ARGMAX and GATHER to produce a candidate. Ordered
  candidate reduction runs in the SDK graph and returns an INT32 token. Equal
  logits choose the smallest token ID, including across blocks. The final block
  contains only valid rows. TOPK and host argmax are not used.
- `native/inference/generation.*` owns model-independent request control. It contains
  no SDK or tensor arithmetic. `native/cli/opt_generate.cpp` runs native model
  initialization, prefill, decoding and ordered teardown.

All new operators are in the chip team's guide. The existing retained ordinary
OpenVX tensor ownership exception is reused: duplicating the 49.09 MiB token
matrix for the head would add a predictable memory cost. Lookup/head share the
same handles; this proves application allocation sharing, not absence of SDK
layout copies or physical board residency. No SDK-private node fields are changed.

The model holds all selected weights, graph objects and KV buffers until close.
At capacity 512, complete OPT weight payload is 662,392,832 bytes, KV payload is
50,331,648 bytes, and their sum is 712,724,480 bytes (679.71 MiB). This is a lower
bound, excluding activation tensors, graph data, SDK layouts and workspace.
Initialization and execution of all 24 layers are tested; physical memory-pool
accounting remains part of the hardware evidence gate. Host peak RSS in new
execution reports must not be presented as board memory use.

## Request and state contracts

Default token selection is resolved from the local checkpoint's
`generation_config.json` using the pinned Transformers 5.16.1 defaults. The current
OPT file specifies BOS/EOS/PAD IDs but no sampling or beam parameters; its effective
policy is `do_sample=false`, `num_beams=1`. The host adapter resolves this before
creating SDK graphs and records the source hash and effective configuration.
Unsupported beam search, active sampling filters, repetition penalties, forced
tokens and other nondefault processors are rejected rather than ignored. Special
token IDs must agree with the deployed native model. Explicit `--sample` retains
the validated temperature-one, unfiltered SDK sampling experiment.

The CPU generation comparison loads the same checkpoint policy and calls the
official `generate()` without overriding `do_sample`, `num_beams` or special IDs.
Generation length (`--max-new-tokens`, default 32) and KV capacity remain explicit
application controls; this selects the model's decoding policy, not every default
argument of the Transformers generation API.

Batch size is one, capacity is fixed in [1,512], and every supplied token is
attended. Padded batches are unsupported. Empty input inserts BOS once; ordinary
input is tokenized with the checkpoint's special-token policy. All IDs and the
complete prompt length are checked before state changes.

Prefill consumes the prompt sequentially and runs the head after its final token.
Decode returns the prediction, stops on EOS/limit/capacity, and only consumes that
prediction if another iteration is needed. Thus the final returned token is not
in KV yet. A prompt filling the cache can still produce one prediction. A zero
new-token limit resets state and performs no model computation.

Reset invalidates outputs and resets valid length without clearing/copying KV.
A failed SDK step invalidates the model; reset cannot recover partial execution.
Close releases head, output graph, layers in reverse order, input graph, table
and context. Context must outlive all users.

One consumed token uploads 26 INT32 controls (104 bytes: lookup index, position,
and 24 valid lengths). One prediction reads four bytes. Normal generation reads
no hidden state, full logits or cache. Diagnostic captures are separate from
these execution counters and intentionally read intermediate results.
Sampling additionally uploads four INT32 seed words per prediction; no full logits
are returned to the host. Request reset also resets the sampling draw counter.

## Commands

Use the `SpecFerry` Conda environment and fresh output directories:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4

# Exact tiny cases: ties, negative maxima, block boundary and final valid token.
python scripts/check_np101_generation.py --mode selection --diagnostic \
  --output .cache/runs/opt-selection

# Original full-vocabulary embedding and head, with official CPU expectations.
python scripts/check_np101_generation.py --mode io --diagnostic \
  --output .cache/runs/opt-io

# Integrate in this order; the small first case also exhausts its cache.
python scripts/check_np101_generation.py --layers 4 --capacity 8 --steps 8 \
  --diagnostic --output .cache/runs/opt-model-4
python scripts/check_np101_generation.py --layers 8 --steps 8 \
  --diagnostic --output .cache/runs/opt-model-8
python scripts/check_np101_generation.py --layers 24 --steps 8 \
  --diagnostic --output .cache/runs/opt-model-24

python scripts/generate_opt.py --prompt 'The capital of France is' \
  --max-new-tokens 32 --compare-cpu --output .cache/runs/opt-generation
```

`--prepare-only` on the checker creates CPU fixtures without opening the board.
Without `--diagnostic`, a numerical-only checker pass returns 2 because exclusive
NPU execution and physical residency remain unproven. `--compare-cpu` optionally
checks the generated token sequence after native execution using the official
CPU model; omit it for normal generation. The generation command
returns 0 for successful native execution, not hardware acceptance. Its report
retains both unresolved evidence flags. All commands use the existing device
lock, bounded timeout and recovery marker; they never retry an abnormal SDK exit.

## Validation and evidence

Independent numerical expectations come from the pinned official CPU FP16 model.
Original checkpoint bytes and the existing `atol=0.08, rtol=0.02` budget are
unchanged. Teacher-forced trajectories compare embedding, every layer output,
projected output, full vocabulary logits and exact predicted IDs. Reset and
fresh-instance repeats must be byte-identical to the first device trajectory.
Final-only runs omit intermediate predictions and captures until the last step.

Evidence under `.cache/runs/` on 2026-09-22:

| Check | Result | Directory |
|---|---|---|
| Original FP16 input/output, six cases | 20 checks passed | `opt-io-20260922` |
| Exact selection and shared-table cases | 23 exact checks passed | `opt-vocabulary-selection-20260922` |
| Four layers, capacity 8 | 131 checks passed, including capacity rejection | `opt-model-4-20260922` |
| Eight layers, capacity 512 | 203 checks passed | `opt-model-8-20260922` |
| All 24 layers, capacity 512 | 491 checks passed | `opt-model-24-20260922` |
| France prompt, 32 generated tokens | All IDs exactly match official CPU generation | `opt-generate-france-20260922` |
| Dinner prompt, 32 generated tokens | All IDs exactly match official CPU generation | `opt-generate-dinner-20260922` |
| Empty prompt, capacity 1 | BOS consumed once; one CPU-matching token, then capacity stop | `opt-generate-bos-capacity-20260922` |
| Checkpoint-default decoding, full 24-layer model | All 32 token IDs matched the official CPU model without overriding its decoding strategy | `opt-model-default-20260922` |
| Qwen3.5 real-weight layers 0–3, 32-step trajectory | 737 checks passed | `generation-qwen-regression-20260922` |

The complete-model suite executed 32 token steps across initial/reset/final-only/
fresh sequences, with 768 KV appends. Normal computation uploaded 3,328 bytes and
read 72 bytes across 18 predictions. All eight official next-token IDs matched.
Creation, release and process exit completed normally without a reboot or sudo.
Maximum absolute error across the full OPT teacher-forced boundaries was 0.046875;
all elementwise comparisons passed the original absolute/relative budget.
The France generation consumed six prompt tokens plus 31 predictions, returned
32 tokens, uploaded 3,848 control bytes and read 128 token bytes. Host peak RSS
was 460,352 KiB. Timings are traced host wall times, not NPU benchmarks.
The dinner prompt likewise matched all 32 CPU tokens, with 39 consumed tokens,
4,056 uploaded control bytes and 128 returned bytes. Its reproducible comparison
is recorded by `generate_opt.py --compare-cpu` in `result.json`. Empty input with
capacity one consumed only BOS, returned token 50118, uploaded 104 bytes, read
four bytes and stopped for capacity; the CPU comparison also passed.

The retained Qwen suite executed 80 steps across four sequences (32 initial,
8 reset, 32 final-only, 8 fresh). It preserved FP32 DeltaNet state and single-buffer
Attention cache, with zero explicit reads during normal steps and maximum
absolute error 0.03125. Its original references and tolerances were unchanged.

Host tests cover invalid model contracts, BOS/EOS, zero-token requests, exact
limits, cache exhaustion, final-token consumption semantics, invalid predictions,
exception propagation and rejection of incomplete/fallback acceptance reports.
All 63 Python unit tests, both native CTests, self-contained headers and both
format checks passed. Generation control tests run without SDK linkage; EOS,
zero-token requests and exception paths are covered there. The device runs above
exercise actual generated maximum-length and capacity stops; they did not produce
an EOS stop during their short natural-language continuations.

After checkpoint-default policy integration, all 67 Python tests passed, including
omitted generation flags, explicit sampling, rejection of unsupported checkpoint
settings and the CPU comparison's absence of decoding overrides. Ruff checks also
passed. The new default-policy device run returned 32 CPU-identical tokens,
consumed 37 tokens, uploaded 3,848 bytes, read 128 bytes and released normally.
Its `result.json` records `selection.origin=model_default`, `do_sample=false`,
`num_beams=1`, Transformers 5.16.1 and the checkpoint generation-config SHA256
`d37de99f9835d7a4d0916257f724504d12c4a668f36a50b2d42e09ba4ebd05d5`.

Numerical agreement and successful driver IO do not identify every kernel's
execution device or all internal transfers. Final backend/residency evidence,
long-context model-quality checks and formal performance acceptance remain open.
The Qwen full-weight allocation issue and deferred DeltaNet weight duplication
remain independent; this work does not close either item.
