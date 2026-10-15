# SpecFerry project constraints

## Implementation

- Use descriptive names, focused functions, explicit ownership/errors and comments for non-obvious decisions. Separate logical blocks; never trade readability for fewer lines.
- Reuse validated code, references and tools. Add or repeat checks only for an observed failure, decision-relevant uncertainty or required acceptance gate; avoid broad matrices and intermediate probes without a concrete gap.
- C++: repository LLVM clang-format, 100 columns, braces on all control statements. Python: Conda `SpecFerry` (3.12), Ruff at 100 columns; run `scripts/format_python.py` for safe fixes/imports and `--check` before delivery. Declare dependencies in `environment.yml` and requirements files; no venvs.
- Headers own declaration/definition dependencies. `.cpp` includes its matching header and implementation dependencies, without repeating that header's direct includes. SDK callers include SDK headers unless their matching header does. Remove unused includes; sort quoted project/SDK/third-party/platform headers case-sensitively before a separate standard-header group, without prioritizing the matching header.

## Measurement

- Measure representative repeated operation collections. Keep preparation, explicit IO and logging outside graph/operator timings; isolated calls do not establish workload throughput.
- Report actual generated output count / generation wall time separately from post-first-token decode rate. State counts and scope; input prefill processing and operator replay are not output throughput. Definitions and accepted baselines live in `docs/np101-status.md`.
- Prioritize input/output and hidden movement as hypotheses, but completion waits include outstanding execution and do not prove an IO bottleneck. Quantify new tracing with the same workload off/on/off; prefer in-memory aggregation over per-call logs or syscall tracing.

## SDK, device and memory

- Prefer fixed tensor bindings and buffer reuse. Resolve semantics from `demo/ref_op_api_guide.md`, `demo/main.c` and installed headers; an operator listing does not specify parameters or prove hardware execution. Undocumented interfaces require missing necessary semantics or substantial predictable documented-interface cost, with explanation and validation before adoption; diagnostic success alone does not justify a default.
- Preserve `vsi_nn_AddNode` initialization: set public fields individually; never zero/replace `nn_param` or overwrite/free `pool.local`.
- Pinned v105 Contexts share the process shader-slot table. Release the last old Context before rebuilding; another live Context does not isolate this budget. Parallel Verify remains unvalidated.
- Before every NP101 run, read and follow `docs/np101-recovery-incidents.md`. Keep an independent record of each device fault requiring power-cycle recovery with host boot ID, exact SDK/kernel timeline, evidence, established cause versus hypotheses and next-run avoidance. Stop board work and preserve the recovery marker on device faults; follow the ledger for incomplete runs. Correct output or automatic recovery does not clear the gate.
- The former repository convolution implementation, cached copies and conv/recovery entry points were deleted by user instruction. Do not restore or rebuild them; historical successes do not revoke this prohibition. Incident evidence remains. Replacement controls require agreement; do not silently substitute vendor convolution. Preserve the recovery marker until agreed checks pass. Recovery requires a user-confirmed NP101 board power cycle, agreed controls, numerical/execution/release/kernel checks and at least 45 seconds of delayed checks; passing controls covers only the checked workload scope. Frozen old OPT and vendor convolution have also failed on confirmed cold boots; follow the latest ledger state before any retry.
- Host cold restart and a changed host boot ID are not required (user instruction, 2026-10-15). Preserve full kernel logs; acknowledge only previously recorded fault occurrences after a confirmed board power cycle, and reject all new faults. Before intentional selection-hang reproduction, remind the user to power-cycle NP101; a host warm restart alone did not establish recovery. Other explicitly authorized diagnostics may proceed. The user grants standing permission for read-only `sudo journalctl` log inspection; do not request it again. Any sudo command other than journalctl, or any journalctl operation that writes or changes state, requires explicit user approval. Passwordless access does not extend this permission.
- Cap application const/nonconst tensor payload separately at 1 GiB, counting shared backing once and temporary peaks. Never change `is_const` to bypass limits. Preserve larger-allocation evidence; no above-limit model/ordinary probes. Only authorized v106 synthetic capacity/integrity probes may use `--probe-limit` up to 4096 MiB; after rejection retain/scan all written blocks. SDK workspace/copies and physical addresses are unknown; logical offsets are not physical addresses.

## Model scope

- Active DLM: `facebook/opt-350m`, the only enabled download; keep others commented until their phase. Preserve Qwen3.5 implementation, cached checkpoint and regression evidence. Exclude all Llama generations and known derivatives from recommendations/downloads/experiments unless explicitly reversed; unrelated `llama.cpp` software is allowed.
- Household endpoints primarily serve one request. Prioritize single-request decode, internal movement and utilization; do not add token guessing inside this initial speculative-inference draft model. Cross-endpoint batching is deferred to the household edge server.
- Separate reusable operators/storage from model composition. Adapters own names, checkpoint paths and ordering; shared NP101 code takes explicit dimensions, tensor contracts and precision policies. Do not claim unvalidated configurations.
- Follow checkpoint generation policy; resolve omitted fields with the pinned reference library. Sampling requires explicit overrides, recorded effective policy and rejection of unsupported settings without silent substitution.
- The first milestone requires hardware evidence for complete DLM prompt processing, embedding, all decoders, state/KV updates, final normalization, LM head and greedy selection. CPU reference/partial offload are debugging stages; tokenization, dispatch, initialization and text display may stay on host. SDK success/software-model execution is insufficient.
- Exclude quantization research, training, MLIR, custom ISA and low-level mapping optimization from this milestone. Preserve reference precision in sensitive state operations and verify device support.

## Records and commits

- The EP exploration phase is closed for the documented accepted OPT and Qwen configurations. Preserve those baselines; reopen optional investigations only for a concrete requirement or new evidence.

- Use functional names in code, fixtures, CLI cases, outputs and published documents; never conversational task codes. Update imports/tests/docs together when renaming. Preserve immutable original evidence under ignored cache paths and refer to descriptive aliases.
- Keep stable conclusions in `docs/np101-status.md`, unfinished steps/dependencies/deliverables/acceptance in `TODO.md`, and device-recovery incidents in their ledger. Summarize completed work instead of appending duplicate run diaries. Keep model selection in conversation; report expected/actual behavior neutrally.
- Commit messages use a single subject starting with square-bracket tags, matching repository history: `[NP101] Optimize OPT inference`, `[Docs] Consolidate validation conclusions`. Use an imperative English summary; do not substitute `type(scope): ...`. Reserve `[WIP][Area] ...` for unfinished work.
