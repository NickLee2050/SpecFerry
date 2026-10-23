#include "inference/generation.hpp"
#include "np101/diagnostics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace specferry::inference {
std::vector<PrefillChunk> plan_prefill(unsigned tokens, unsigned block, unsigned position) {
  if (!block || block > 16) {
    throw std::invalid_argument("prefill block must be in [1,16]");
  }
  std::vector<PrefillChunk> result;
  unsigned offset = 0;
  while (offset < tokens) {
    const bool aligned = (position % block + offset % block) % block == 0;
    const auto count = aligned && tokens - offset >= block ? block : 1;
    result.push_back({offset, count});
    offset += count;
  }
  return result;
}

unsigned prefill_replay_tokens(unsigned position, unsigned appended, unsigned block,
                               double single_seconds, double block_seconds) {
  if (!block || block > 16) {
    throw std::invalid_argument("prefill block must be in [1,16]");
  }
  const auto replay = position % block;
  if (!replay || !appended || !std::isfinite(single_seconds) || !std::isfinite(block_seconds) ||
      single_seconds <= 0 || block_seconds <= 0) {
    return 0;
  }
  const auto leading = std::min(appended, block - replay);
  const auto original_blocks = (appended - leading) / block;
  const auto replay_blocks = (std::uint64_t(appended) + replay) / block;
  return replay_blocks > original_blocks && (block - replay) * single_seconds > block_seconds
             ? replay
             : 0;
}

Generation generate_tokens(const GenerationPolicy &policy, const GenerationExecutor &executor,
                           const std::vector<std::int32_t> &prompt, unsigned maximum_new_tokens,
                           const std::function<void(std::int32_t)> &on_token,
                           const GenerationPhaseObserver &on_phase, GenerationHistory *history) {
  if (!policy.capacity || !policy.vocabulary ||
      policy.vocabulary > std::numeric_limits<std::int32_t>::max() ||
      policy.bos >= policy.vocabulary || policy.eos >= policy.vocabulary || !executor.reset ||
      !executor.consume || !executor.predict || (executor.prepare_prefill && !executor.prefill)) {
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
  if (history &&
      (!history->valid || !maximum_new_tokens || history->consumed > history->tokens.size() ||
       (!history->tokens.empty() && history->consumed + 1 != history->tokens.size()) ||
       ids.size() < history->tokens.size() ||
       !std::equal(history->tokens.begin(), history->tokens.end(), ids.begin()))) {
    throw std::invalid_argument("generation history does not match the supplied prompt prefix");
  }
  unsigned prefix = history ? history->consumed : 0;
  const bool continuing = history && !history->tokens.empty();
  const auto preparation_start = std::chrono::steady_clock::now();
  if (history) {
    // A failed executor must never leave a resumable history attached to partial KV writes.
    history->valid = false;
  }
  Generation result;
  if (continuing && executor.prepare_prefill) {
    result.replayed_prompt_tokens = executor.prepare_prefill(prefix, ids.size() - prefix);
    if (result.replayed_prompt_tokens > prefix) {
      throw std::runtime_error("prefill replay exceeds the validated history prefix");
    }
    prefix -= result.replayed_prompt_tokens;
  } else if (!continuing) {
    if (on_phase) {
      on_phase(GenerationPhase::Reset);
    }
    np101::SdkScope phase("phase", "reset");
    executor.reset();
  }
  result.consumed = prefix;
  result.state_preparation_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - preparation_start).count();
  result.stop_reason = "max_new_tokens";
  if (!maximum_new_tokens) {
    return result;
  }
  if (on_phase) {
    on_phase(GenerationPhase::Prefill);
  }
  const auto start = std::chrono::steady_clock::now();
  {
    np101::SdkScope phase("phase", "prefill");
    if (executor.prefill) {
      executor.prefill({ids.begin() + prefix, ids.end()});
      result.consumed = ids.size();
    } else {
      for (auto index = prefix; index < ids.size(); ++index) {
        executor.consume(ids[index]);
        ++result.consumed;
      }
    }
  }
  result.prefill_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  auto previous = start;
  for (unsigned index = 0; index < maximum_new_tokens; ++index) {
    const auto token = [&] {
      np101::SdkScope phase("phase", index == 0 ? "prefill" : "decode");
      return executor.predict();
    }();
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
    // First prediction belongs to prefill; decode begins only if it is consumed.
    if (!index && on_phase) {
      on_phase(GenerationPhase::Decode);
    }
    {
      np101::SdkScope phase("phase", "decode");
      executor.consume(token);
    }
    ++result.consumed;
  }
  result.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (history) {
    history->tokens = ids;
    history->tokens.insert(history->tokens.end(), result.tokens.begin(), result.tokens.end());
    history->consumed = result.consumed;
    history->valid = true;
  }
  return result;
}
} // namespace specferry::inference
