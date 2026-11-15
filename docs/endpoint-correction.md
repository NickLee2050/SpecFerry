# Endpoint correction

This local C++ interface accepts or revises a pending generated suffix. It does not call a
larger target model, estimate acceptance rates, add token guessing, or introduce a network service.
The caller supplies decisions; tests use fixed decisions. Ordinary model generation is unchanged.

## State and decisions

`CorrectionLoop` exclusively uses one model callback for its lifetime. Do not call that model
independently while a loop has a pending proposal. The model must outlive the callback. A loop is
not copyable or thread-safe. Its `CorrectionSession` is the session identity; revision numbers are
local to that object, not portable request IDs for a future transport protocol.

- `start(prompt, maximum)` resets the model and produces a pending proposal. An empty prompt uses
  the policy's beginning-of-sequence token; Qwen callers should supply a rendered chat prompt.
- `apply({revision, accepted_count, token}, maximum)` accepts the leading `accepted_count` proposal
  tokens. An optional token replaces the first rejected token, or follows a fully accepted proposal.
  Omitting it discards the rejected suffix without inserting a replacement.
- Logical accepted tokens are separate from model-consumed tokens. The final prediction has not
  entered the model state. Complete acceptance may therefore still require consuming that token.
- A decision must reference the current pending revision. Invalid tokens/counts, stale decisions,
  tokens after an accepted end-of-sequence (EOS), and capacity overflow fail before device work.
- Accepted EOS or one final unconsumed token beyond consumed capacity finishes the session without
  another model call. No physical rollback is claimed in this terminal case; a new `start` resets.
- An executor failure invalidates the session. Do not retry `apply`; recreate a failed model if its
  own contract requires it, then start again after the applicable device recovery checks.

## Execution strategy

Full acceptance retains state only when the corrected prefix extends the complete pending history.
Rejection that changes that history uses reset plus replay of the entire corrected prefix. This
reconstructs Qwen's FP32 recurrent matrices and convolution history as well as attention KV. Merely
shortening KV would leave the other state inconsistent. Reset changes KV visibility; masked stale
slots need not be cleared physically.

This baseline allocates no device snapshots, additional graph instances or duplicate weights.
It trades replay time for simple ownership and existing validated reset semantics. Checkpoint or
partial-recompute rollback is not implemented. Adapters without validated continuation support
must construct the loop with `supports_retention=false`; every nonterminal decision then replays.
In particular, do not enable OPT mapped-slot continuation, which its model explicitly rejects.

```cpp
using namespace specferry;
inference::CorrectionLoop session(
    policy, [&](const auto &tokens, unsigned maximum, bool retain) {
      return model.generate(tokens, maximum, np101::TensorReadback::MappedHost,
                            retain ? models::qwen3_5::HistoryMode::Retain
                                   : models::qwen3_5::HistoryMode::Reset);
    });
auto draft = session.start(prompt, 4);
// The caller supplies a revision-matched decision; no target inference happens here.
auto next = session.apply({draft.proposal->revision, 1, replacement_token}, 4);
```

`CorrectionStep::response_seconds` includes reset, replay, generation and local bookkeeping.
The optional `Generation` retains existing prefill, TTFT and decode timings; its wall time excludes
reset. `replayed_prefix_tokens` counts the whole corrected prefix on a replay, while
`retained_consumed` counts device positions reused. Initial prompt processing is not rollback replay.
Finished steps have no new `Generation` or `CorrectionProposal`. Accepted tokens and newly generated
pending tokens must not both be counted as newly produced model output.

CPU tests cover zero/partial/full acceptance, replacements/bonus tokens, EOS, capacity, stale and
invalid decisions, executor failure, and recurrence-aware replay equivalence. Device acceptance and
its scope are recorded in [status](np101-status.md); protocol tests alone do not establish hardware
state correctness or speculative-inference speedup.
