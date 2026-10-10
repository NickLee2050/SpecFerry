# SpecFerry project constraints

## Code and tools

- Use descriptive names, focused functions, explicit ownership/errors, and comments for non-obvious decisions. Separate logical blocks; never reduce readability to cut lines.
- Prioritize the core implementation goal; reuse validated code, references and existing tools. Add tests/scripts only for a concrete gap. Expand checks only for an observed failure, uncertainty that affects a decision, or a required acceptance gate; do not routinely repeat completed validation or add intermediate probes and broad matrices.
- Measure performance over repeated operation collections with representative ordering and working sets; keep preparation, explicit IO and logging outside timed sections. Isolated-call timings are diagnostic and do not establish workload throughput.
- Report LLM throughput using actual generated output tokens: separate resident-request output count / generation wall time from post-first-token decode rate. Label input-token prefill processing and operator replay separately; never present them as output throughput. State output count and timing scope.
- Prioritize input preparation, output transfer and hidden data movement as performance hypotheses. Completion waits include outstanding execution and do not establish an IO bottleneck. Quantify added tracing overhead with the same workload and tracing off/on/off; prefer in-memory aggregation over per-call logging or syscall tracing.
- C++: repository LLVM clang-format, 100 columns, braces on all control statements. Python: Conda `SpecFerry` (3.12), Ruff at 100 columns; run `scripts/format_python.py` for safe fixes/imports and `--check` before delivery. Declare dependencies in `environment.yml` and requirements files; no venvs.
- Headers own declaration/definition dependencies. `.cpp` includes its matching header plus implementation dependencies, without repeating that header's direct includes. SDK callers include SDK headers unless their matching header already does. Remove unused includes. Quote project/SDK/third-party/platform headers before angle-bracket standard headers; separate groups and sort case-sensitively, without prioritizing the matching header.

## SDK and device

- Prefer fixed tensor bindings and buffer reuse; swap handles only when necessary.
- In the pinned v105 SDK, concurrently live Contexts share the process shader-slot table. Release the last old Context before rebuilding a model; a second Context does not isolate this budget. Parallel Verify remains unvalidated.
- Resolve API/operator semantics from `demo/ref_op_api_guide.md`, `demo/main.c` and installed headers. Operator listing alone neither specifies parameters nor proves hardware execution.
- Prefer documented interfaces. Undocumented alternatives require missing necessary semantics or substantial, predictable cost in documented interfaces; explain and validate before adoption. A passing diagnostic alone does not justify a new default.
- Preserve `vsi_nn_AddNode` initialization: set individual public parameters; never zero/replace `nn_param` or overwrite/free `pool.local`.
- Sudo requires explicit authorization for the action; passwordless access and consumed one-time grants do not authorize it.
- Before intentional selection-hang reproduction, remind the user to cold-restart host and NP101; warm restarts failed. Other explicitly authorized diagnostics may proceed.
- Before every NP101 run, consult `docs/np101-recovery-incidents.md` and apply its avoidance rules. For any event requiring a cold restart, create a separate incident entry with the boot ID, exact SDK-call and kernel timeline, evidence paths, established cause versus hypotheses, and a concrete rule for the next run. A correct output or driver "automatic recovery" does not clear the device gate. Preserve the recovery marker and stop board work; after confirmed host and NP101 cold restart, archive the marker and use a fixed repository convolution control followed by the frozen old OPT short request. Both must pass numerical, execution, release and kernel checks, with at least 45 seconds of delayed checks after each control. This permits continued testing within the checked workload scope; it does not prove fault-free hardware. The vendor demo is optional diagnosis, not a prerequisite after both controls pass. The repository convolution has prior hang evidence; its immutable archived binary/source are now pinned. `check_np101.py conv` runs that control, `recovery` then runs frozen old OPT, and `vendor-conv` is optional vendor diagnosis. Do not rebuild or replace pinned controls silently. If a control fails again under unchanged conditions, stop repeated board attempts and analyze the saved evidence before another recovery cycle.

## Memory

- Cap application const/nonconst tensor payload separately at 1 GiB; count shared backing once and temporary peak allocations. Never change `is_const` to bypass limits. Preserve larger-allocation evidence; no above-limit probes without a policy change, except the synthetic capacity probe below. Hidden SDK workspace/copies and physical addresses are unknown; logical offsets are not physical addresses.
- Authorized synthetic capacity/integrity probes near the v106 limit may use `--probe-limit` up to 4096 MiB. After SDK rejection retain/scan all successfully written blocks. Model inference and ordinary checks keep the 1 GiB limits.

## Models and milestone

- Active DLM: `facebook/opt-350m`, the only enabled download. Keep others commented until their phase. Preserve Qwen3.5 implementation, cached checkpoint and validation evidence for regression.
- Household endpoints primarily serve one request. OPT-350m is the draft model in the speculative-inference design; do not add another token-guessing stage to its initial implementation. Cross-endpoint request batching belongs to the household edge server and is deferred for endpoint optimization. Prioritize single-request decode execution, internal data movement and hardware utilization.
- Separate reusable operators/storage from model composition. Adapters own names, checkpoint paths and architecture ordering; shared NP101 code takes explicit dimensions, tensor contracts and precision policies. Do not claim unvalidated configurations.
- Default to checkpoint generation policy; resolve omitted fields with the pinned reference library. Sampling needs explicit overrides; record effective policy and reject unsupported settings without silent strategy substitution.
- Exclude every Llama generation and known Llama-derived checkpoint from recommendations/downloads/experiments unless explicitly reversed. Unrelated `llama.cpp` software is allowed.
- First milestone: NP101 executes complete DLM prompt processing, embedding, every decoder, state/cache updates, final normalization, LM head and greedy selection. CPU reference/partial offload are debugging stages. Tokenization, dispatch, initialization and text display may stay on host. Require hardware evidence; SDK success/software-model runs are insufficient.
- Exclude quantization research, training, MLIR, custom ISA and low-level mapping optimization from this milestone. Preserve necessary reference precision in sensitive state operations; validate NP101 support before claiming full deployment.

## Reports and plans

- Use descriptive functional names for source files, modules, fixtures, CLI cases, outputs and published documents; do not put conversational task codes in the repository. Update imports, tests and documentation together when renaming. Describe SDK issues by their failing operation. Preserve immutable original evidence under ignored cache paths; use descriptive aliases when referring to it.
- Report failing operation, expected/actual results and reproduction neutrally; explain terms without abstract summaries or conversation history.
- Plans specify steps, dependencies, deliverables, checks and acceptance; keep model selection in conversation. Current evidence and task context live in `docs/np101-status.md` and `TODO.md`.
