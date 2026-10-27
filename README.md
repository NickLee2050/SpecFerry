# SpecFerry
A testbed for speculative decoding across heterogeneous edge devices and a local inference hub.

This DGX checkout deploys `facebook/opt-6.7b` as the target language model (TLM)
using vLLM 0.28.0. The NP101 DLM is developed in a separate checkout/session;
its active model is OPT-350M. This checkout's NP101 SDK smoke test and Qwen CPU
references do not represent that remote runtime. Real NP101 draft callbacks
have not yet been connected here. Project constraints are in [AGENTS.md](AGENTS.md).

## Set up the Conda environment

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate SpecFerry
python --version
```

The shared `SpecFerry` environment uses Python 3.12 for downloads, CPU references,
and tests. Conda manages the interpreter; the environment's pip installs the
pinned Python dependencies, including CPU-only PyTorch. Model weights and run
records stay in `.cache/`; project source stays in `scripts/` and `python/`.

To apply later dependency changes, run `conda env update -n SpecFerry -f environment.yml`.
For a shell without activation, prefix a command with
`conda run --no-capture-output -n SpecFerry`, for example:

```bash
conda run --no-capture-output -n SpecFerry python scripts/download_models.py --dry-run
```

## Download the active checkpoint

```bash
conda activate SpecFerry
python scripts/download_models.py --dry-run
python scripts/download_models.py
```

The default DGX TLM download remains `facebook/opt-6.7b`. On 2026-10-16 the user
explicitly selected `Qwen/Qwen3.5-27B` for standalone validation; select it with
`--model Qwen/Qwen3.5-27B`. Other Qwen download entries remain commented out.
The user subsequently authorized `facebook/opt-30b`; select it explicitly with
`--model facebook/opt-30b`. The default remains OPT-6.7B.
The script downloads
the original checkpoint and tokenizer under `.cache/models/facebook/opt-6.7b`, pins the revision, resumes
interrupted transfers, and writes `download-manifest.json` only after validating
the files. This does not convert or deploy the model. OPT uses PyTorch weight
shards and a BPE vocabulary/merges pair; duplicate TensorFlow and Flax weights
are excluded. The NP101 reference commands below still target Qwen3.5-0.8B and
require its separate checkpoint when that phase resumes.

Other model entries are commented out in `scripts/download_models.py`. After
enabling a required entry, `--parallel-models 2 --file-workers 4` allows two model
downloads concurrently with up to four file workers each. A single weight shard
does not become multiple shards through this option. Use `--output` to choose a
different storage root. Re-running a download retains its locked revision; use a
new output directory to intentionally select a different revision.

If the Xet transfer service stalls, stop that invocation and re-run with
`--http-only` to use the Hub's HTTP transfer path. Do not run two copies against
the same output directory simultaneously.

For repeated single-file HTTP interruptions, `--http-only --range-workers 12`
downloads the public weight file in bounded, resumable byte ranges. Each response
must match its requested range; the assembled file must match the Hub SHA256
before it is published. `--file-workers` also bounds simultaneous weight shards;
the total range concurrency is at most `file-workers * range-workers`. Range
fragments remain under the model's `.cache/`. A failed shard prevents publication
of the complete manifest, including when other shards have already succeeded.
For range transfers, each shard's download redirect is resolved without a Range
header and the signed CDN location is reused across byte ranges, then refreshed
before expiration or after HTTP 403. This avoids resolving the same Hub URL for
every fragment. HTTP 429 stops queued requests; honor the server's Retry-After
before re-running. Completed fragments are retained.

The download script works on both Windows and Linux. The current full reference
environment is validated on Linux x86_64; the native NP101 SDK is required to
build and run the SDK smoke test.

## Record the host and device environment

Run from the repository root, in a session that can see `/dev/galcore`:

```bash
python scripts/collect_environment.py --output .cache/runs/local
```

`--sdk-include` and `--sdk-lib` override `/usr/inc` and `/usr/lib/ljmicro`.
`environment.json` records toolchain versions, library fingerprints, PCI driver
binding and memory-pool parameters. `hardware-evidence.md` states the limits of
this evidence. A sandbox that hides device nodes cannot validate board access.
No driver or system configuration is changed by these tools.

## Run CPU reference inference and validation

The model download must finish and write its verified `download-manifest.json`
before running these commands.
Use the same activated `SpecFerry` Conda environment for the CPU reference.
CUDA and FLA are not required.

```bash
conda activate SpecFerry
python scripts/inspect_checkpoint.py --output .cache/runs/local/checkpoint-inventory.json
python scripts/reference_dlm.py --output .cache/runs/local --threads 8 --max-new-tokens 32
```

The loader checks the entire checkpoint and strictly maps all text weights,
excludes vision/MTP, and verifies the shared embedding/head. Both reference modes
run entirely on the host CPU. `original-fp32` expands the original BF16 weight
values losslessly to FP32. `deployment-fp16` uses FP16 projections and explicit
FP32 sensitive operations and recurrent states; native FP32 parameters stay FP32.
`reference-policy.json` records each boundary and the exact reference source hash.

Outputs include Chinese/English greedy generation, token IDs, step logits, cache
shapes, prefill-versus-sequential comparisons at 1/2/4/8 tokens, and actual layer
0/3 intermediates in `reference-trace/`. All model loading is local and offline.
Check `reference-summary.json` and `tolerances.json`; a failed numeric comparison
returns nonzero and does not silently widen the thresholds. These checks do not
prove NPU support, full DLM deployment, or general model quality.

## Build and run the NP101 SDK smoke test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j 4
python scripts/run_np101_smoke.py --output .cache/runs/local/smoke --repeats 10
```

CMake generates `build/compile_commands.json` by default during configuration,
providing clangd with the compiler flags and include paths for code completion
and diagnostics.

Configure `LJMICRO_SDK_INCLUDE_DIR` and `LJMICRO_SDK_LIBRARY_DIR` through CMake
when the SDK is elsewhere; also pass `--sdk-lib` to the runner. This builds
host C++17 code from `native/np101/`; the supplied `demo/` stays untouched.
The smoke uses the demo's FP16 NCHW CONV2D→RELU→POOL layout and CPU tolerance
0.1, checks allocations and NaN/Inf, reuses one graph while varying inputs, and
checks that release clears both handles. Release checks do not establish absence
of every internal SDK allocation leak.

`smoke.json` contains numeric results and phase timing; `smoke.log`,
`smoke.strace`, and `smoke-evidence.json` contain SDK output, successful driver
calls, the new binary hash and resolved runtime paths. The runner has a 1200 s
timeout (`--timeout` overrides it). A numeric pass is separate from proof of
individual node execution on NPU. Generic driver IO and a target label alone
do not meet the latter requirement; review the execution evidence before
concluding that the tested operations ran on the NPU.

## Format C++ code

The repository's `.clang-format` uses LLVM style with a 100-column line limit
and a blank line between function definitions. Control statements (`if`, `else`,
`for`, `while`, and `do`) require braces even for a single-statement body;
clang-format inserts missing braces automatically.
The Conda environment includes clang-format 18.1.8 for consistent formatting.

```bash
conda activate SpecFerry
python scripts/format_cpp.py
python scripts/format_cpp.py --check
```

The script formats C++ sources and headers listed by Git, including new,
untracked files while respecting Git ignore rules. `--check` reports formatting
differences and returns nonzero without modifying files. Use `--clang-format`
to select an executable at a custom path. The script works from any directory.

## Regression checks

```bash
python -m unittest discover -s tests -v
```

## DGX TLM service

Machine-local Docker deployment files live in the ignored `deploy/` directory.
Model weights and experiment records stay in `.cache/`.

The DGX Spark deployment runs `facebook/opt-6.7b` in FP16 using vLLM 0.28.0,
with a 2048-token context, concurrency one, and 2 GiB of KV cache. The container
is named `specferry-tlm`; its completion API is bound to `127.0.0.1:8000`.
Use `/v1/completions` with model name `specferry-tlm`. This checkpoint is a
text-completion model; no chat template is configured.

On this host, Miniconda is installed at `~/miniconda3`. Its Python 3.12
`SpecFerry` environment contains the download dependencies declared in
`environment.yml` / `requirements-download.txt`; the x86_64 CPU reference
dependencies are not installed on this ARM64 host. New Bash shells initialize
Conda without automatically activating base. For an existing shell:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SpecFerry
python scripts/benchmark_tlm.py
```

The 2026-10-15 baseline passed health, model-list, completion, and streaming
checks. Three warmed requests with distinct short prompts and 128 generated
tokens each measured a median client-observed decode rate of 15.41 tokens/s
and median TTFT of 72.7 ms. TTFT includes API and prefill overhead; these are
local single-request measurements, not distributed speculative decoding results.
The Chinese smoke-test output repeated text and does not demonstrate good
Chinese generation quality. Raw results, GPU utilization, checkpoint revision,
and startup logs are in `.cache/runs/tlm-opt/` and the checkpoint manifest.

To probe cached-chunk processing and simulate serial draft/verify timing:

```bash
python scripts/benchmark_verify_proxy.py
```

This uses the running completion API with isolated cache salts, 128/512-token
cached prefixes, and 1/2/3/5/9/17/33 fresh input positions. Each suffix starts with
a unique token so repeated trials cannot reuse its computed KV states. Three
warmups per shape precede seven shuffled measurements; aggregate cache-query
and cache-hit counts must exactly match the expected counts. Run without other
requests to this service, since concurrent traffic invalidates that accounting.
The 2026-10-15 measurements were approximately 69–76 ms per cached chunk.

This is an optimistic verification-cost proxy, not a speculative decoder:
the API computes final-position logits only, with no draft acceptance/rejection,
KV rollback, DLM catch-up, or network RTT. The simulation assumes a constant
conditional acceptance probability, counts the correction/bonus token, and
uses `sum(a**i for i in range(K+1)) / (K / draft_rate + chunk_time + RTT)`.
Acceptance rates are hypothetical. A draft rate of 2 tokens/s makes this serial
scheme slower than the measured TLM-only baseline, even with perfect acceptance.
Measurements, assumptions, cache evidence, and simulation results are saved in
`.cache/runs/verify-proxy/results.json`; the existing TLM service stays running.

## Full OPT candidate verification

`python/specferry/verification.py` implements single-sequence greedy verification:
compute predictions at all candidate positions, accept a matching prefix, emit
one correction or bonus token, and discard rejected KV states. The cache contains
all committed tokens except the last token, which starts the next verification
forward. EOS and output budgets are handled explicitly. Probabilistic sampling
and a remote verifier API are not implemented.

The GPU reference backend uses Transformers OPT with torch SDPA, independently
of the running vLLM service. It computes the LM head for **all K+1 positions**.
Do not compare its timing against the vLLM baseline as if both were one backend.
The native vLLM integration below is a separate implementation.

Run from the SpecFerry Conda environment on this DGX with the existing cached
image and checkpoint. Each output directory must be new:

```bash
conda activate SpecFerry
python -m unittest tests.test_verification -v
python scripts/run_opt_verifier.py \
  --output .cache/runs/opt-fp32-check --dtype float32 --phase correctness
python scripts/run_opt_verifier.py \
  --output .cache/runs/opt-fp16-timing --dtype float16 --phase performance
```

The runner starts a disposable container with no network, a 30-minute timeout,
and read-only project/Conda mounts. Only its results directory is writable.
It uses the **SpecFerry Python 3.12 interpreter** and GPU libraries from the pinned
image, declared in `requirements-tlm-runtime.txt`; it installs no packages.
Source snapshots/hashes, the exact command, GPU telemetry, and vLLM counters
are saved with each run. Avoid concurrent GPU workloads when measuring latency.

The 2026-10-15 run passed nine protocol unit tests and 21 real-weight FP32 checks
across three English prompts. Checks include full acceptance, first/middle
rejection with subsequent decoding, output budgets, synthetic stop IDs, and
48-token multi-round generation. Token IDs matched an independent ordinary
`OPTForCausalLM.forward` greedy loop. This is a finite test suite, not a universal
equivalence proof. The GPU stop-ID test is synthetic; actual EOS control flow
is covered by the deterministic unit backend.

**Known FP16 limitation:** the soup prompt diverged after a middle rejection.
At the first differing position, the ordinary path tied tokens 40992 and 44644
at 13.3203125; the verification path scored them 13.3125 and 13.3203125.
The observed maximum logit difference at that position was 0.0078125.
FP32 passes the same case with TF32 disabled, supporting a numerical explanation;
FP16 exact token equivalence remains unproven and its failed run is retained.
`--phase performance` explicitly skips the correctness gate and records reference
token agreement per trial. It must not be reported as an FP16 correctness pass.

Raw evidence is in `.cache/runs/overnight-20261015/`: `attempt-01/` retains the
FP16 failure, `fp16-divergence.json` contains the diagnostic logits/KV differences,
and `fp32-correctness/` contains the passing higher-precision suite.
`fp16-performance/` contains full-verifier timing and synthetic DLM experiments.
Performance trials use context lengths 128/512/1536, K=0/1/2/4/8/16/32,
accept/reject cases, two warmups per case and seven shuffled repetitions.
They exclude initial prefill. The sleep-based DLM loop uses oracle candidates
and synthetic RTT; it is neither an NP101 execution nor an actual network test.

Measured FP16 median wall time, including all-position LM head, argmax, CPU
accept/reject and cache crop (milliseconds; all-accept cases):

| Context tokens | K=0 | K=1 | K=4 | K=8 | K=16 | K=32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 66.72 | 63.08 | 62.39 | 62.09 | 63.57 | 66.44 |
| 512 | 68.02 | 66.82 | 65.85 | 65.33 | 66.72 | 69.45 |
| 1536 | 75.34 | 74.08 | 72.31 | 72.90 | 73.90 | 76.56 |

All 399 measured trials matched their ordinary reference token prefixes on this
benchmark workload. This does not resolve the separate FP16 failure above.
vLLM generation/request counters did not change during the measurement run.
Its post-run health, model list and a 16-token completion also passed.

An analytical example assumes a 128-token context, conditional acceptance 0.8,
10 ms RTT, serial drafting and no DLM resynchronization overhead. The baseline is
the **same Transformers backend's** K=0 measurement (14.99 tokens/s):

| Assumed DLM tokens/s | Best tested K | Estimated speedup over K=0 |
| ---: | ---: | ---: |
| 2 | 1 | 0.21x |
| 10 | 1 | 0.69x |
| 20 | 1 | 0.98x |
| 50 | 4 | 1.47x |
| 100 | 4 | 2.00x |

These are modeled scenarios, not hardware acceleration claims; TLM-only fallback
remains preferable below 1x. Eight additional sleep-based DLM experiments ran
the actual GPU verifier. With K=4 and a simulated DLM rate of 2 tokens/s, the
perfect-candidate case produced 2.43 output tokens/s; the assumed-0.8-acceptance
case produced 1.64 tokens/s over 24 output tokens. The small sample is illustrative.
The consolidated task record is `.cache/runs/overnight-20261015/report.json`;
`verification-timing.csv`, `analytical-simulation.csv`, and `timed-mock.csv`
provide tabular exports. Raw trial timings, candidate decisions and source
snapshots remain alongside them.

## SpecPIM reproducibility audit

The [author's H2-LLM artifact](https://github.com/leesou/H2-LLM-ISCA-2025/tree/c5708b5fc63fcb614d7f81ae464c8fc9fb4efa27)
contains SpecPIM's compute-centric dataflow search, selected by
`--is-dataflow-fixed specpim`. It is an architecture/dataflow evaluation framework,
not a ready-to-run distributed draft/verify server. The
[artifact evaluation notes](https://github.com/leesou/H2-LLM-ISCA-2025/blob/c5708b5fc63fcb614d7f81ae464c8fc9fb4efa27/ae/README.md#L72-L74)
state that the public evaluators replace an unavailable simulator with roofline
estimates and validate latency only. Both public processor evaluators return
zero energy; this artifact cannot establish the original paper's energy results.

The source audit is saved in `.cache/references/specpim/audit.json`, with 19
downloaded source/configuration/license files, SHA256 hashes and evidence links.
Twelve Python files pass Python 3.12 syntax parsing; imports and execution were
not tested. Upstream recommends Python 3.10, torch 2.0.1, numpy 1.25.2 and a native
Domino dependency. ARM64/Python 3.12 compatibility and GB10/NP101 cost calibration
remain unverified. The original ACM PDF returned HTTP 403, so its complete
evaluation setup has not been checked. No upstream DSE or native build was run.

For subsequent comparisons, retain a TLM-only baseline and a fixed-K serial
draft/verify baseline on identical hardware, precision, workloads and backend.
A transferred fixed heterogeneous mapping should be named **SpecPIM-inspired**
until its assumptions and original methodology are validated. Account for DLM
resynchronization, network RTT and the board's usable memory in actual runs.
Hardware/dataflow exploration remains deferred until the full NP101 inference
milestone is complete.

## Native vLLM external draft integration

The target now has an external draft adapter for **vLLM 0.28.0**. It uses the
engine's experimental `custom_class` proposer interface. vLLM performs candidate
verification, rejection sampling, and paged KV-cache state management within a
single generation request. The standard completion API on port 8000 remains the
ordinary target service. The experimental service uses loopback port 8011 and
model name `specferry-tlm-spec`.

Production transport and protocol logic is C++ in `native/draft/client.cpp`:
libcurl HTTP(S), absolute request deadlines, bounded replies, prefix/tokenizer
fingerprints, request identity checks, token validation, fallback and audit.
`python/specferry/vllm_proposer.py` is the thin Python hook required by vLLM; it
converts the framework's committed-prefix arrays and calls the native C ABI.
Python launchers, fixtures and validation programs are scripts. No upstream
vLLM source or model weights are patched.

The initial scope is OPT greedy drafting, TP/PP/DP=1, one concurrent sequence,
synchronous scheduling, and K between 1 and 32. The server launcher uses eager
execution to keep the initial integration inspectable. Its latency must not be
reported as a speedup over the graph-enabled ordinary service. FP16 batch-shape
numerical differences remain a separate correctness limitation.

Build the C++ library independently of the NP101 SDK:

```bash
cmake -S native/draft -B build/draft -DCMAKE_BUILD_TYPE=Release
cmake --build build/draft -j 4
```

Build dependencies are CMake, a C++17 compiler, libcurl, OpenSSL and json-c headers
and libraries. On this DGX, the libcurl headers were extracted from the official
Ubuntu `libcurl4-openssl-dev` 8.5.0-2ubuntu10.16 ARM64 package into project cache;
no system packages were installed. Use these overrides on this host:

```bash
cmake -S native/draft -B build/draft -DCMAKE_BUILD_TYPE=Release \
  -DCURL_INCLUDE_DIR="$PWD/.cache/dependencies/curl-8.5.0-2ubuntu10.16/root/usr/include/aarch64-linux-gnu" \
  -DCURL_LIBRARY=/usr/lib/aarch64-linux-gnu/libcurl.so.4
cmake --build build/draft -j 4
```

After a DLM implements the callback, launch the target with the SpecFerry Conda
Python interpreter. Use a new results directory:

```bash
python scripts/serve_vllm_spec.py \
  --dlm-url http://DEVICE_ADDRESS:DEVICE_PORT/v1/draft \
  --output .cache/runs/native-device-test --draft-tokens 4 --timeout 30
```

The launcher creates `specferry-tlm-spec` from the pinned cached image. It mounts
the same Conda Python 3.12 interpreter, native library and model, and records the
command, source snapshots, native library hash and checkpoint revision. It uses
host networking so a loopback callback or SSH tunnel is reachable, while binding
the inference API to `127.0.0.1`. It does not configure a public listening port.
Set `SPECFERRY_DLM_TOKEN` when the DLM callback requires bearer authentication;
the launcher forwards it without writing its value into the launch record.

The terminal device sends its usual generation request to the TLM completion
API. During generation, the TLM calls the device's draft endpoint, verifies the
returned candidates, and streams committed output back to the terminal:

```text
Terminal app -> TLM /v1/completions
                   -> DLM /v1/draft -> candidate IDs
                   -> native vLLM verify/accept/reject -> SSE -> terminal display
```

`POST /v1/draft` receives the entire authoritative committed prefix, including
the target's newly sampled token(s) exactly once. The device must synchronize
its draft cache to this prefix, discarding an uncommitted/rejected suffix before
drafting. It may return fewer than K candidates, including zero:

```json
{
  "protocol": "specferry.draft.v1",
  "request_id": "unique-request-id",
  "prefix_sha256": "sha256-of-compact-json-prefix-token-array",
  "tokenizer_sha256": "opt-bpe-fingerprint",
  "draft_mode": "greedy",
  "prefix_token_ids": [2, 10, 11],
  "max_draft_tokens": 4
}
```

The reply echoes `protocol`, `request_id`, `prefix_sha256`, `tokenizer_sha256`
and `draft_mode`, and adds `draft_token_ids`. Candidate IDs must be integers
within the target vocabulary, with count at most `max_draft_tokens`. The OPT BPE
fingerprint hashes, in order, each of `vocab.json` and `merges.txt` as filename,
NUL, eight-byte big-endian file length and raw file bytes. OPT BOS/EOS IDs are 2;
the device consumes the supplied IDs without adding special tokens. A different
tokenizer is rejected. Errors, timeouts and stale replies yield zero drafts so
the target continues ordinary decoding. The callback timeout must accommodate
actual draft latency; 0.2 seconds is used only for the local failure tests.

`scripts/mock_dlm_server.py` is a loopback-only **test fixture**, not a small
model. It supplies precomputed same-engine oracle tokens and controlled errors.
`scripts/validate_vllm_spec.py` compares against the same vLLM engine with empty
drafts, checks native acceptance counters, rejection recovery, output budgets,
stop tokens, HTTP failure fallback and SSE output. Fixtures and all raw requests
are saved under `.cache/runs/vllm-native*/`. No NP101 hardware or device NIC has
been tested through this adapter yet.

The 2026-10-16 C++ transport/binding run passed 24 relevant unit tests. Native
vLLM **FP32 passed all 17 integration cases**; FP16 passed 14/17. All three FP16
failures are the soup prompt in all/first/middle-rejection cases, diverging at
generated index 5 (token 44644 instead of 40992), the same near-tie location as
the independent reference failure. Strict FP16 equivalence is not established.
The FP32 run recorded 230 native draft rounds, 920 proposed tokens and 260
accepted tokens, demonstrating actual engine speculation. These counts include
extra accepted tokens subsequently truncated by output budgets. They are not
performance measurements or NP101 results.

Evidence is in `.cache/runs/vllm-native-cpp-20261016/` (FP16, expected nonzero
validation exit) and `.cache/runs/vllm-native-cpp-fp32-20261016/` (FP32). Each
retains its exact native binary, SHA256, source snapshots, validation JSON and
native per-request audit. `vllm-native-20261016/` and
`vllm-native-20261016-02/` retain earlier startup/numerical investigation records.
The test fixture resets to `no_draft` after each validation, so ordinary requests
to the experimental service use target-only decoding until fixtures are enabled.

For a fresh strict validation, first run the fixture in one terminal, then start
an FP32 experimental service in another terminal (ports/names must be unused):

```bash
conda activate SpecFerry
python scripts/mock_dlm_server.py --log .cache/runs/mock-dlm/requests.jsonl
```

```bash
conda activate SpecFerry
python -m unittest tests.test_vllm_proposer -v
python scripts/serve_vllm_spec.py \
  --dlm-url http://127.0.0.1:8012/v1/draft --dtype float32 \
  --name specferry-tlm-spec-check --port 8013 --timeout 0.2 \
  --output .cache/runs/native-check-server
# Wait for /health to return HTTP 200 before validation.
python scripts/validate_vllm_spec.py --base-url http://127.0.0.1:8013 \
  --dtype float32 --output .cache/runs/native-validation.json
```

Use `--continue-on-mismatch` for FP16 investigation; mismatches remain failed
cases and the command returns nonzero. The FP32 validation container is stopped
after the recorded run to free its GPU memory. The ordinary service and FP16
experimental service remain available.

## Execution decision and experiment baselines

The 2026-10-16 decision is to use pinned vLLM first, connect the real DLM, and
measure the unoptimized serial draft/verify cycle before optimizing it. The
cross-device method and draft contract stay outside framework-specific hooks.
A second TLM backend is a later portability check; framework independence has
not yet been demonstrated. Keep numerical differences documented, while retaining
protocol, rejection, cache recovery and output checks as acceptance gates.

Compare optimized cooperation with both (1) target-only inference and (2) the
unoptimized cross-device cycle. Report these speedups separately. Record actual
output tokens per generation wall time, TTFT, accepted prefix length, zero-accept
rounds, and drafting/communication/verification/synchronization costs. A callback's
HTTP wall time includes device execution and is not a measurement of network RTT
alone. Engine counters around a request require an otherwise idle service.

For a matching target-only baseline, use the same integration launcher, dtype,
K, eager execution and synchronous scheduling, with the proposer returning empty
drafts without making HTTP calls:

```bash
conda activate SpecFerry
python scripts/serve_vllm_spec.py --draft-source disabled \
  --name specferry-tlm-baseline --port 8010 \
  --output .cache/runs/baseline-server
```

Both integration services use model name `specferry-tlm-spec`. The existing
graph-enabled `specferry-tlm` on port 8000 remains a deployment reference with a
different configuration. Empty-draft mode retains the custom proposer machinery;
it is a controlled integration baseline, not a claim to maximum target-only speed.

## NP101 connection contract and remaining work

`POST /v1/draft` is a **SpecFerry-defined HTTP contract**, not an NP101 SDK API.
Its machine-readable shape is [protocol/draft-v1.schema.json](protocol/draft-v1.schema.json).
The DGX does not know the function names or cache ABI of the remote DLM runtime.
Implement its HTTP handler around the existing board inference code; no vLLM
installation is required on the DLM host.

| Interface or behavior | DGX status | Device-side requirement |
| --- | --- | --- |
| Terminal input and confirmed text output | Native `specferry generate` HTTP/SSE client | Build it on the terminal host; no GPU/SDK required |
| Target verification and target KV state | vLLM native verifier with thin hook | No target implementation on NP101 |
| Draft HTTP transport | C++ client, deadlines, identity/token checks and fallback | Implement `/v1/draft` around the actual DLM runtime |
| Token identity | OPT BPE fingerprint and integer ID contract | Check the same fingerprint; consume supplied IDs directly |
| Prefix and DLM cache synchronization | Full authoritative prefix sent every round | Discard rejected suffix; replay/catch up before drafting |
| Runtime capability and execution evidence | Not inferred from successful HTTP | Publish actual model/context limits, runtime revision and board trace |
| Real transport address | Undecided | Supply a reachable host/tunnel endpoint when the device service is ready |

The device handler must validate the request, synchronize its state to exactly
`prefix_token_ids`, generate at most K greedy candidate IDs, and echo all five
identity fields. The prefix includes BOS and the latest target-confirmed tokens;
never tokenize it again or insert another BOS. Hash the ASCII UTF-8 compact JSON
array, e.g. `[2,10,11]`, without spaces or a final newline. The tokenizer hash is
defined above; source tokenization policy is owned by the TLM. The native client
also checks that prefix length plus K fits the 2048-token target context.

For an initial implementation, rebuilding DLM state from the full prefix is
correct if cache truncation/reuse is unavailable; record its full cost. The
optimized implementation can reuse the longest common prefix, truncate invalid
state, and process missing confirmed tokens. Retain the exact relationship
between cached length and the last pending token in the device runtime's own
contract. Do not assume the cache ABI of this checkout's old SDK smoke test.
The request identity is per RPC, not a persistent session; current concurrency
is one. A change of request must still reconcile the entire supplied prefix.

An empty draft array is valid. Unknown tokenizer, unsupported context, invalid
prefix or device failure must produce an explicit error, or a documented empty
draft response; never silently truncate the prompt. On transport/protocol errors,
the DGX returns no candidates and lets the target continue. This fallback proves
availability, not successful device participation. Optional device timing fields
may be added to replies but are currently ignored by the verifier.

## Host tools and initial network path

`cmake --build build/draft -j 4` builds `build/draft/specferry` as well as the
library. The executable has no Python, GPU or NP101 dependency. Build it on each
host architecture instead of copying the DGX ARM64 binary to an x86 host.

```bash
build/draft/specferry fingerprint --model-dir .cache/models/facebook/opt-6.7b
build/draft/specferry probe --model-dir .cache/models/facebook/opt-6.7b \
  --dlm-url http://127.0.0.1:18012/v1/draft \
  --prefix-file /path/to/prefix-token-ids.json --draft-tokens 4 --timeout-ms 30000
build/draft/specferry generate --base-url http://127.0.0.1:8010 \
  --prompt-file /path/to/question.txt --max-tokens 128 \
  --output-dir .cache/runs/terminal-baseline
```

`probe` returns nonzero for transport or protocol failure; a valid empty reply
returns zero and reports no candidates. It does not establish real NPU execution.
`generate` prints only generated text to stdout and records input, request, raw
SSE events, output IDs, finish reason, actual output count, TTFT, decode rate and
end-to-end rate in a fresh directory. Incomplete/error streams return nonzero.

For initial host-mediated connectivity, keep both services on loopback and run
this SSH connection **from the DLM host** after it is allowed to run its server
on local port 8012:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:18011:127.0.0.1:8011 \
  -R 127.0.0.1:18012:127.0.0.1:8012 lishuhan@dgx01
```

The terminal calls `http://127.0.0.1:18011`; the DGX launcher uses
`--dlm-url http://127.0.0.1:18012/v1/draft`. Hostname resolution and existing SSH
key access must work from the DLM host. No container SSH daemon or public API
port is required. This is a host-mediated baseline; direct board NIC routing and
its performance remain undecided. Neither SSH nor the target-only service starts
or validates board inference. Follow device-side recovery gates before any NPU run.

After the real callback and the target service are ready, run on the DGX:

```bash
conda activate SpecFerry
python scripts/check_dlm_link.py \
  --dlm-url http://127.0.0.1:18012/v1/draft \
  --output .cache/runs/device-link-check
```

Use the same callback URL as the target's launch record. This probes the contract,
runs the native terminal and retains before/after engine metrics. It returns
nonzero if the generation only fell back without engine draft tokens. A positive
draft count with zero accepted tokens still establishes participation; it is not
a performance pass. Counter deltas can include accepted tokens later truncated
by the output budget. Pair these records with device logs for hardware evidence.

## Coordination between the two remote sessions

The DGX handoff is [handoff/dgx.json](handoff/dgx.json); dated service state must be
checked live. Use `git rev-parse HEAD`, the protocol file hash and launch source
hashes to identify the code actually running. Each session owns its local build,
services and hardware evidence. The NP101 session should publish its own handoff
with model/tokenizer, context limit, callback address, cache behavior, runtime
commit, device readiness and latest validation evidence.

When connected through the same Codex app, cross-host thread summaries can help
retrieve progress, but do not automatically merge conversation memory or source
trees. Prefer a shared Git contract and per-host handoffs. Read the other side's
handoff before integration changes; verify claims against code and run evidence.
Use separate branches and selective commits for shared protocol changes. The two
checkouts currently have substantially different model/runtime work: do not
overwrite their AGENTS, dependency files or NP101 recovery rules by copying the
entire DGX checkout. Do not synchronize build trees, model caches, credentials,
Docker state or live device state. Fetching a commit is not deploying its code.

Current remaining work is the actual device HTTP handler and cache adapter,
device capability/evidence publication, and real host-to-host validation. All
loopback/oracle tests remain synthetic and cannot supply NP101 performance data.
