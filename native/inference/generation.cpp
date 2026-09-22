#include "inference/generation.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>

namespace specferry::inference {
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
  for (auto token : ids) {
    executor.consume(token);
    ++result.consumed;
  }
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
  return result;
}
} // namespace specferry::inference
