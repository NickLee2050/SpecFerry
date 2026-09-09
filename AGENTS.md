# SpecFerry project constraints

- Write human-readable, high-quality code. Use descriptive names, focused functions, explicit ownership and error handling, and comments that explain non-obvious decisions. Separate logical blocks and keep formatting consistent; do not compress code at the expense of readability.
- Format C++ with the repository LLVM-based, 100-column clang-format rules, including braces for every control-statement body. Format Python with Ruff at 100 columns, apply safe lint fixes, and sort imports using `scripts/format_python.py`; verify with `--check` before delivery.
- Preserve SDK-owned node state initialized by `vsi_nn_AddNode`. Set individual public parameters; never zero or replace the entire `nn_param` structure or overwrite/free `pool.local`.
- Do not use sudo without explicit authorization for the particular action. The one-time authorization for `sudo journalctl -xe` on 2026-09-09 has been consumed; passwordless sudo is not continuing authorization.

- Use the Conda environment `SpecFerry` with Python 3.12 for project Python tools, downloads, references, and tests. Define its dependencies in `environment.yml` and the referenced requirements files; do not create or configure venv environments.
- Use Qwen3.5 for the active model roadmap. The first and only enabled download is `Qwen/Qwen3.5-0.8B`; keep other model downloads commented out until their phase starts.
- Permanently exclude all Llama model generations and known Llama-derived checkpoints from recommendations, downloads, and experiments in this project, unless the user explicitly reverses this preference. This does not prohibit the unrelated `llama.cpp` software project. OPT is a separate model family; discussion does not authorize adding it to the active download list.
- The first implementation milestone is the complete text-generation computation of the selected DLM on NP101, including prompt processing, token embedding, every decoder layer, state/cache updates, final normalization, LM head, and greedy token selection. CPU reference and partial offload are debugging stages, not completion of this milestone.
- Tokenization, command dispatch, initialization, and text display may remain on the host. Record hardware execution evidence; a successful SDK call or software-model run alone is insufficient.
- Do not introduce quantization research, training, MLIR, custom ISA generation, or low-level mapping optimization into the first milestone. Preserve necessary reference precision in sensitive state operations and verify NP101 support before claiming full deployment.
- Keep planning documents focused on concrete implementation steps, dependencies, deliverables, checks, and acceptance criteria. Keep model-selection discussion in the conversation.
