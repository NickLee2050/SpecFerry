#include "inference/generation.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>

namespace specferry::inference {
std::vector<PrefillChunk> plan_prefill(unsigned tokens, unsigned block) {
  if (!block || block > 8) {
    throw std::invalid_argument("prefill block must be in [1,8]");
  }
  std::vector<PrefillChunk> result;
  unsigned position = 0;
  while (position < tokens) {
    const auto count = tokens - position >= block ? block : 1;
    result.push_back({position, count});
    position += count;
  }
  return result;
}

Generation generate_tokens(const GenerationPolicy &policy, const GenerationExecutor &executor,
                           const std::vector<std::int32_t> &prompt, unsigned maximum_new_tokens,
                           const std::function<void(std::int32_t)> &on_token) {
  if (!policy.capacity || !policy.vocabulary ||
      policy.vocabulary > std::numeric_limits<std::int32_t>::max() ||
      policy.bos >= policy.vocabulary || policy.eos >= policy.vocabulary || !executor.reset ||
      !executor.consume || !executor.predict) {
    throw std::invalid_argument("invalid generation policy or executor");
  }
  const auto ids = prompt.empty() ? std::vector<std::int32_t>{std::int32_t(policy.bos)} : prompt;
  if (ids.size() > policy.capacity) {
    throw std::out_of_range("prompt exceeds capacity");
  }
  for (auto token : ids) {
    if (token < 0 || unsigned(token) >= policy.vocabulary) {
      throw std::invalid_argument("prompt token exceeds vocabulary");
    }
  }
  executor.reset();
  Generation result;
  result.stop_reason = "max_new_tokens";
  if (!maximum_new_tokens) {
    return result;
  }
  const auto start = std::chrono::steady_clock::now();
  if (executor.prefill) {
    executor.prefill(ids);
    result.consumed = ids.size();
  } else {
    for (auto token : ids) {
      executor.consume(token);
      ++result.consumed;
    }
  }
  result.prefill_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  auto previous = start;
  for (unsigned index = 0; index < maximum_new_tokens; ++index) {
    const auto token = executor.predict();
    if (token < 0 || unsigned(token) >= policy.vocabulary) {
      throw std::runtime_error("generation executor returned an invalid token");
    }
    const auto now = std::chrono::steady_clock::now();
    const auto seconds = std::chrono::duration<double>(now - previous).count();
    if (!index) {
      result.first_token_seconds = seconds;
    } else {
      result.token_seconds.push_back(seconds);
    }
    result.tokens.push_back(token);
    if (on_token) {
      on_token(token);
    }
    if (unsigned(token) == policy.eos) {
      result.stop_reason = "eos";
      break;
    }
    if (index + 1 == maximum_new_tokens) {
      break;
    }
    if (result.consumed == policy.capacity) {
      result.stop_reason = "capacity";
      break;
    }
    previous = now;
    executor.consume(token);
    ++result.consumed;
  }
  result.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}
} // namespace specferry::inference
