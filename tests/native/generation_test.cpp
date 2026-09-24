#include "inference/generation.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("generation contract failed");
  }
}

template <class Action> void rejects(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected);
}

void contracts() {
  using namespace specferry::inference;
  std::vector<std::int32_t> consumed, predictions{3, 4, 2}, streamed;
  unsigned resets = 0, selected = 0;
  const GenerationExecutor executor{[&] {
                                      ++resets;
                                      consumed.clear();
                                      streamed.clear();
                                      selected = 0;
                                    },
                                    [&](std::int32_t token) { consumed.push_back(token); },
                                    [&] { return predictions.at(selected++); }};
  const auto notify = [&](std::int32_t token) { streamed.push_back(token); };
  auto result = generate_tokens({10, 8, 2, 2}, executor, {2, 7}, 8, notify);
  require(result.tokens == predictions && result.stop_reason == "eos");
  require(consumed == std::vector<std::int32_t>({2, 7, 3, 4}));
  require(result.consumed == 4 && streamed == predictions && result.token_seconds.size() == 2);
  require(result.prefill_seconds >= 0 && result.first_token_seconds >= result.prefill_seconds &&
          result.total_seconds >= result.first_token_seconds);

  result = generate_tokens({10, 8, 2, 2}, executor, {}, 1, notify);
  require(consumed == std::vector<std::int32_t>({2}) &&
          result.tokens == std::vector<std::int32_t>({3}));
  require(result.stop_reason == "max_new_tokens" && result.consumed == 1);

  result = generate_tokens({10, 2, 2, 2}, executor, {2, 7}, 8);
  require(result.stop_reason == "capacity" && result.consumed == 2 && result.tokens.size() == 1);
  result = generate_tokens({10, 2, 2, 2}, executor, {2, 7}, 1);
  require(result.stop_reason == "max_new_tokens" && result.consumed == 2);
  result = generate_tokens({10, 8, 2, 2}, executor, {}, 0);
  require(consumed.empty() && !selected && result.tokens.empty() && !result.consumed);
  require(result.prefill_seconds == 0 && result.first_token_seconds == 0 &&
          result.total_seconds == 0 && result.token_seconds.empty());

  const auto before = resets;
  rejects([&] { generate_tokens({10, 2, 2, 2}, executor, {2, 3, 4}, 1); });
  rejects([&] { generate_tokens({10, 8, 2, 2}, executor, {-1}, 1); });
  rejects([&] { generate_tokens({10, 8, 2, 2}, executor, {10}, 1); });
  require(resets == before);
  predictions = {2};
  result = generate_tokens({10, 1, 2, 2}, executor, {2}, 8);
  require(result.stop_reason == "eos" && result.consumed == 1 && selected == 1);
  predictions = {10};
  rejects([&] { generate_tokens({10, 8, 2, 2}, executor, {2}, 1); });
  predictions.clear();
  rejects([&] { generate_tokens({10, 8, 2, 2}, executor, {2}, 1); });

  // Block prefill must preserve BOS, EOS and consumed-token semantics, including
  // a final predicted token that has not yet entered the cache.
  auto blocked = executor;
  unsigned prefill_calls = 0;
  blocked.prefill = [&](const std::vector<std::int32_t> &tokens) {
    ++prefill_calls;
    consumed.insert(consumed.end(), tokens.begin(), tokens.end());
  };
  predictions = {3, 4, 2};
  result = generate_tokens({10, 8, 2, 2}, blocked, {2, 7, 8, 9, 1}, 3);
  require(prefill_calls == 1 && result.consumed == 7 && result.tokens == predictions);
  require(consumed == std::vector<std::int32_t>({2, 7, 8, 9, 1, 3, 4}));
  result = generate_tokens({10, 8, 2, 2}, blocked, {}, 1);
  require(consumed == std::vector<std::int32_t>({2}) && prefill_calls == 2);
  result = generate_tokens({10, 8, 2, 2}, blocked, {2}, 0);
  require(prefill_calls == 2 && consumed.empty());

  for (unsigned length = 0; length <= 128; ++length) {
    for (unsigned block : {1, 4, 8}) {
      unsigned end = 0;
      for (auto chunk : plan_prefill(length, block)) {
        require(chunk.begin == end && (chunk.count == block || chunk.count == 1));
        require(chunk.count == 1 || chunk.begin % block == 0);
        require(chunk.begin + chunk.count <= length);
        end += chunk.count;
      }
      require(end == length);
    }
  }
  const auto tail = plan_prefill(7, 4);
  require(tail.size() == 4 && tail[0].count == 4 && tail.back().begin == 6);
  rejects([&] { plan_prefill(8, 0); });
  rejects([&] { plan_prefill(8, 9); });
}
} // namespace

int main() {
  try {
    contracts();
    std::cout << "Generation BOS, EOS, capacity, limits and consumed-token contracts passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
