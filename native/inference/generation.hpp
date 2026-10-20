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

// Batch one, no padding. All supplied prompt IDs participate in attention.
// The last prediction is returned without consuming it into the model state.
// Optional history requires the complete matching prefix and consumes only its pending suffix.
Generation generate_tokens(const GenerationPolicy &policy, const GenerationExecutor &executor,
                           const std::vector<std::int32_t> &prompt, unsigned maximum_new_tokens,
                           const std::function<void(std::int32_t)> &on_token = {},
                           const GenerationPhaseObserver &on_phase = {},
                           GenerationHistory *history = nullptr);
} // namespace specferry::inference
