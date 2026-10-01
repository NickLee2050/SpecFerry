# SpecFerry project constraints

## Code and tools

- Use descriptive names, focused functions, explicit ownership/errors, and comments for non-obvious decisions. Separate logical blocks; never reduce readability to cut lines.
- C++: repository LLVM clang-format, 100 columns, braces on all control statements. Python: Conda `SpecFerry` (3.12), Ruff at 100 columns; run `scripts/format_python.py` for safe fixes/imports and `--check` before delivery. Declare dependencies in `environment.yml` and requirements files; no venvs.
- Headers own declaration/definition dependencies. `.cpp` includes its matching header plus implementation dependencies, without repeating that header's direct includes. SDK callers include SDK headers unless their matching header already does. Remove unused includes. Quote project/SDK/third-party/platform headers before angle-bracket standard headers; separate groups and sort case-sensitively, without prioritizing the matching header.

## SDK and device

- Prefer fixed tensor bindings and buffer reuse; swap handles only when necessary.
- Resolve API/operator semantics from `demo/ref_op_api_guide.md`, `demo/main.c` and installed headers. Operator listing alone neither specifies parameters nor proves hardware execution.
- Prefer documented interfaces. Undocumented alternatives require missing necessary semantics or substantial, predictable cost in documented interfaces; explain and validate before adoption. A passing diagnostic alone does not justify a new default.
- Preserve `vsi_nn_AddNode` initialization: set individual public parameters; never zero/replace `nn_param` or overwrite/free `pool.local`.
- Sudo requires explicit authorization for the action; passwordless access and consumed one-time grants do not authorize it.
- Before intentional P3/selection-hang reproduction, remind the user to cold-restart host and NP101; warm restarts failed. Other explicitly authorized diagnostics may proceed.

## Memory

- Cap application const/nonconst tensor payload separately at 1 GiB; count shared backing once and temporary peak allocations. Never change `is_const` to bypass limits. Preserve larger-allocation evidence; no above-limit probes without a policy change, except P5 below. Hidden SDK workspace/copies and physical addresses are unknown; logical offsets are not physical addresses.
- Authorized synthetic P5 capacity/integrity probes near the v106 limit may use `--probe-limit` up to 4096 MiB. After SDK rejection retain/scan all successfully written blocks. Model inference and ordinary checks keep the 1 GiB limits.

## Models and milestone

- Active DLM: `facebook/opt-350m`, the only enabled download. Keep others commented until their phase. Preserve Qwen3.5 implementation, cached checkpoint and validation evidence for regression.
- Separate reusable operators/storage from model composition. Adapters own names, checkpoint paths and architecture ordering; shared NP101 code takes explicit dimensions, tensor contracts and precision policies. Do not claim unvalidated configurations.
- Default to checkpoint generation policy; resolve omitted fields with the pinned reference library. Sampling needs explicit overrides; record effective policy and reject unsupported settings without silent strategy substitution.
- Exclude every Llama generation and known Llama-derived checkpoint from recommendations/downloads/experiments unless explicitly reversed. Unrelated `llama.cpp` software is allowed.
- First milestone: NP101 executes complete DLM prompt processing, embedding, every decoder, state/cache updates, final normalization, LM head and greedy selection. CPU reference/partial offload are debugging stages. Tokenization, dispatch, initialization and text display may stay on host. Require hardware evidence; SDK success/software-model runs are insufficient.
- Exclude quantization research, training, MLIR, custom ISA and low-level mapping optimization from this milestone. Preserve necessary reference precision in sensitive state operations; validate NP101 support before claiming full deployment.

## Reports and plans

- Use current P1–P10 IDs from `docs/np101-sdk-driver-validation-report.md`; preserve historical names/IDs, never infer current numbering from paths.
- Report failing operation, expected/actual results and reproduction neutrally; explain terms without abstract summaries or conversation history.
- Plans specify steps, dependencies, deliverables, checks and acceptance; keep model selection in conversation. Current evidence and task context live in `docs/np101-status.md` and `TODO.md`.
