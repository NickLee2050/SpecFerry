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
  double first_token_seconds = 0;
  std::vector<double> token_seconds;
};

struct GenerationExecutor {
  std::function<void()> reset;
  std::function<void(std::int32_t)> consume;
  std::function<std::int32_t()> predict;
};

// Batch one, no padding. All supplied prompt IDs participate in attention.
// The last prediction is returned without consuming it into the model state.
Generation generate_tokens(const GenerationPolicy &policy, const GenerationExecutor &executor,
                           const std::vector<std::int32_t> &prompt, unsigned maximum_new_tokens,
                           const std::function<void(std::int32_t)> &on_token = {});
} // namespace specferry::inference
