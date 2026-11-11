# Configurable inference runtime plan

Updated: 2026-11-11. Directory migration is complete; stages 1–4 below are planned, not implemented.
SoC OPT/Qwen whole-model regressions passed. EP acceptance for this migration is compilation-only
at the user's request; hardware regression remains pending.

## Scope

Build a lightweight C++ inference runtime around the existing OPT and Qwen implementations.
Models supply explicit configuration and packages; shared code handles loading, generation,
timing and resource ownership. Keep model state semantics, NP101 operators and execution-mode
policies separate. This is not an arbitrary-model compiler.

Graph construction, Setup/Verify, RunGraph and token loops already run mainly in C++. Moving more
logic from Python improves structure, dependencies and consistency; it does not by itself improve
NPU throughput. Readability and acceptance matter more than line count.

## Organization and dependencies

The [README](../README.md#repository-layout) shows the current tree and the
[development guide](development.md) defines current responsibilities. Do not maintain a second
copy of that tree here. The following rules govern further changes:

- Root directories describe responsibilities: `src`, `tools`, `tests`, `docs`, `cmake`.
  Siblings use one classification axis; file suffixes identify languages.
- Product modules use nouns: `cli`, `inference`, `models`, `devices`; add `io` only for real
  shared file/package code. Avoid an unbounded `utils` or `common` module.
- Operational tools use short verbs: `deploy/{export,pack,execute}`, `benchmark`, `validate`,
  `format`. Keep `reference` for independent CPU implementations. There is no extra
  `specferry_tools` package layer or parallel `scripts` tree.
- Tests use `unit`, `integration`, `device`, with responsibility-based subdivisions;
  `fixtures` and `support` hold small data and test-only helpers. Unit/integration checks do not
  open the board; hardware tests remain explicitly enabled. Product code must not depend on helpers.
- `cmake/targets.cmake` defines build relationships; `cmake/devices/np101` owns SDK/target
  selection, and `cmake/toolchains` selects cross compilers and system-library roots.
  Build configuration need not mirror every source directory.
- Keep small command entries beside their tools. Update imports, build targets, formatting,
  editor settings, tests and examples together when moving code. Do not retain duplicate imports.

EP and SoC share model/operator code. CMake selects source files through `SPECFERRY_TARGET`,
without preprocessor branches duplicating computation. Current API-specific differences live in
`src/devices/np101/api/{ep,soc}`; there is no separate `platform` directory. Runtime profiles still
include model choices such as convolution coverage and embedding block size; move those into named
execution plans later, without changing their behavior during a structural move.

Future NP101 intermediate representation (IR) support starts separately under
`src/devices/np101/ir`, likely for SoC only. Do not create it before the vendor interface exists,
assume it reuses API tensor ownership, or advertise every API/IR × EP/SoC combination. Existing
Network Binary Graph (NBG) imports belong to the API implementation because they still execute
through that SDK. Extract common code only after actual semantics and ownership agree.

```text
CLI → inference runtime → model adapter → NP101 operators/resources → vendor SDK
                 ↘ file/package IO          API build selects EP or SoC

independent supervisor → inference child process
Python offline tools → exported packages
Python deployment tools → C++ command entries
```

Models own network structure, weight names, precision and state semantics. Shared operators take
explicit dimensions and tensor contracts. Do not force OPT attention KV and Qwen recurrent state
into a universal state class. No empty directories or `.gitkeep` architecture placeholders.

## Configuration boundaries

| Category | Contents | Contract |
|---|---|---|
| Model specification | Family, layers, dimensions, vocabulary, precision, generation defaults | Derived from checkpoint/model adapter; retain supported-range checks. |
| Package | Weight/NBG identities, ports, layout, shapes, source and SDK | Bound to real artifacts; configuration cannot change a fixed graph's meaning. |
| Execution plan | Context capacity, prefill width, operator choices, state commit | Only implemented combinations; distinguish valid configuration from hardware acceptance. |
| Request | Input, output limit, end-of-sequence behavior, supported generation policy, reset | Cannot restart a workload deadline; reject unsupported settings explicitly. |
| Runtime policy | Preparation/execution budgets, observation, retries, memory margins | Constrained by device policy and authorization, with no general bypass switch. |

Reuse manifests/JSON with versioned structures and explicit defaults. Model dimensions, SDK
constants and runtime-policy limits belong to their respective modules; equal numbers do not
justify a shared constant. Avoid a giant constants file. Separate options, artifact references
and resource ownership.

## Python and C++ responsibilities

| Work | Destination | Boundary |
|---|---|---|
| Request parsing, execution options, results | C++ | Thin CLI; reusable validation in the runtime. |
| Package structure, ports, budgets, SDK compatibility | C++ authoritative at execution | Exporters retain independent checks at their own trust boundary; share formats and tests. |
| Loading, generation, lifecycle | C++ | Reuse validated operators and ownership; unify orchestration. |
| Tokenization and text decoding | Existing implementation initially | Token-ID input is sufficient for the first step; assess native tokenizer compatibility separately. |
| Timeouts, kernel checks, recovery markers, release checks | Independent C++ supervisor, last | A stuck inference process must not supervise itself. |
| Checkpoint processing, model interchange/graph export, CPU reference | Python | Preserve existing tooling and the independent numerical oracle. |
| SSH deployment and experiment orchestration | Thin Python tools | Organize calls without duplicating model execution policy. |

The long-term aim is a board inference path without Python, while retaining Python offline.
Do not introduce another primary language without a concrete capability/toolchain benefit.
The current board supervisor uses Python 3.9 standard-library facilities; do not introduce host-only
heavy dependencies before replacing it.

## Refactoring targets

| Target | Change |
|---|---|
| Shared IO | Extract named atomic-JSON and file-digest helpers; preserve different rules for ordinary files and deployment files that forbid symbolic links. |
| Configuration | Follow Qwen `ModelOptions`; replace long positional OPT/layer arguments. |
| Thin commands | Separate parsing, validation, preparation, execution and result handling; share execution contracts. |
| Package loading | Reuse shape/type/port parsing; adapters retain layer order, tensor names and model-specific constraints. |
| Results | Share timing definitions, atomic publication and release records; preserve model fields and consumer compatibility. |
| Large modules | Separate OPT loading, prefill, decode and state management; keep model selection out of SDK profiles. |
| Build/tests | Keep source-based EP/SoC selection and explicit dependencies; gradually free independent CPU contract tests from SDK dependencies. |
| Comments/errors | Explain non-obvious SDK limits, commits and ownership. Errors identify stage/object and whether continued use is valid. |

## Delivery order

| Stage | Deliverable | Minimal acceptance |
|---|---|---|
| 0. Directory migration | ✅ Sources, tools, tests, imports, build, editor and documentation paths migrated; ignores updated | 142 Python tests, four C++ CPU tests, 18 command entries, both builds and SoC OPT/Qwen whole-model regressions passed. EP hardware pending. |
| 1. Configuration/contracts | Named options/constants, shared IO and package-version rules | Existing packages and invalid-input cases cover parsing, defaults and rejection; normally no board run. |
| 2. Loading/thin entries | Move execution parsing/results into C++; preserve CLI behavior | Stage 1 first; preserve SDK ordering and artifact contracts, run relevant CPU tests and both builds. |
| 3. Shared runtime | Common load/generate/close flow; split large model modules | Stage 2 first; reuse one OPT and one Qwen request per batch, checking numerics, TTFT, output throughput, memory and release. |
| 4. Supervisor/deployment | Separate board supervisor; document remaining runtime dependencies | Stable runtime first; fake-child tests for deadlines, exit, readiness and failure records, then minimal device acceptance. Native text support only as needed. |

Each stage must be independently reviewable and committable. Do not mix directory moves with
operator/performance changes. Audit/configuration/IO work can proceed independently once interfaces
are settled; do not simultaneously refactor all callers of an unsettled interface. Keep model/state
splits separate from supervisor migration to make regressions diagnosable.

## Main-branch and documentation policy

Include `docs/`, `AGENTS.md` and `TODO.md` in the main branch. Maintain project-authored guidance
in English; preserve vendor originals and immutable historical evidence in their original language.

| Material | Policy |
|---|---|
| README, usage, architecture, tests | Keep concise supported behavior, prerequisites and working entry points. |
| AGENTS.md | Keep shared coding, acceptance and device-safety rules. Separating personal authorization/machine details is future work; this plan does not revoke existing rules. |
| TODO.md | Keep unfinished deliverables, dependencies, acceptance and reopening conditions; do not present plans as capabilities. |
| This plan | Mark implementation status; fold completed architectural decisions into the development guide rather than maintaining duplicates forever. |
| Status and incident records | Keep decision-relevant conclusions, limits and traceable evidence. Original incident narratives may be archived unchanged with an English operational index. |
| Weights, SDKs, raw logs, deployment cache | Keep outside source control; document production/acquisition and identity checks. |
| Editor configuration | Team-shared settings only; local interpreter preferences are not runtime requirements. |
| Credentials and local assistant/proxy state | Keep out of main; use precise ignores rather than ignoring all dotfiles. |

`.cache`, `.ruff_cache`, `build` and `demo` are ignored; exact ignores were added for `.agents`,
`.codex` and `.aws`. The unused `tests/repro` directory was removed. The three protected empty
hidden directories were not forcibly deleted. Shared `.vscode` files remain tracked. Adding an
ignore does not untrack files or remove secrets from history. No license or secrets audit is claimed;
review personal paths, network details and vendor redistribution rights before external publication.

## Preservation requirements

- Keep accepted operator selection, construction order, tensor bindings, state commits and SDK release order.
- Preserve deployment hashes, runtime metadata, recovery gates and deadlines; AGENTS and the ledger remain authoritative.
- Rebuild deployment records when source/binary identities change; never edit frozen evidence to avoid verification.
- Do not prebuild thread pools, parallel Verify, RPC, graph languages or plugin loaders.
- Keep current model command behavior initially. Any CLI consolidation needs explicit migration instructions.
- Before each later stage, settle option granularity, native tokenizer scope, package compatibility and supervisor ownership; do not create speculative scaffolding.

The primary benefit is maintainability. Runtime speed claims still require separate whole-model evidence.
