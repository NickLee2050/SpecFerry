#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace specferry::inference {
struct GenerationPolicy {
  unsigned vocabulary, capacity, bos, eos;
};

struct Generation {
  std::vector<std::int32_t> tokens;
  unsigned consumed = 0;
  unsigned replayed_prompt_tokens = 0;
  std::string stop_reason;
  double prefill_seconds = 0;
  double first_token_seconds = 0;
  std::vector<double> token_seconds;
  double total_seconds = 0;
  // Reset stays outside legacy generation timings; conversation response timings include it.
  double state_preparation_seconds = 0;
};

struct GenerationExecutor {
  std::function<void()> reset;
  std::function<void(std::int32_t)> consume;
  std::function<std::int32_t()> predict;
  // Optional block prefill; it must consume exactly the supplied IDs, without padding.
  std::function<void(const std::vector<std::int32_t> &)> prefill = {};
  // Called only after full-prefix validation, with history invalidated. May rewind
  // at most the consumed prefix; the caller replays those known IDs through prefill.
  std::function<unsigned(unsigned, unsigned)> prepare_prefill = {};
};

// Includes the last returned prediction, which has not yet entered the KV cache.
struct GenerationHistory {
  std::vector<std::int32_t> tokens;
  unsigned consumed = 0;
  bool valid = true;
};

struct PrefillChunk {
  unsigned begin, count;
};

enum class GenerationPhase { Reset, Prefill, Decode };
using GenerationPhaseObserver = std::function<void(GenerationPhase)>;

// Complete aligned blocks, with single-token leading and trailing steps.
std::vector<PrefillChunk> plan_prefill(unsigned tokens, unsigned block, unsigned position = 0);

// Recompute a partial historical block only when it adds a full block and the
// observed block cost is lower than the single-token steps it replaces.
unsigned prefill_replay_tokens(unsigned position, unsigned appended, unsigned block,
                               double single_seconds, double block_seconds);

// Batch one, no padding. All supplied prompt IDs participate in attention.
// The last prediction is returned without consuming it into the model state.
// Optional history requires the complete matching prefix and consumes only its pending suffix.
Generation generate_tokens(const GenerationPolicy &policy, const GenerationExecutor &executor,
                           const std::vector<std::int32_t> &prompt, unsigned maximum_new_tokens,
                           const std::function<void(std::int32_t)> &on_token = {},
                           const GenerationPhaseObserver &on_phase = {},
                           GenerationHistory *history = nullptr);
} // namespace specferry::inference
