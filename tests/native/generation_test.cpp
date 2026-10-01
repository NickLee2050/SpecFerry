#include "inference/generation.hpp"
#include "inference/generation_io.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
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

void run_options_contracts() {
  namespace fs = std::filesystem;
  using specferry::inference::read_run_options;
  const auto directory =
      fs::temp_directory_path() /
      ("specferry-run-options-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directory(directory);

  struct Cleanup {
    fs::path directory;

    ~Cleanup() {
      std::error_code ignored;
      fs::remove_all(directory, ignored);
    }
  } cleanup{directory};

  const auto defaults = read_run_options(directory, true);
  require(defaults.maximum == 8 && defaults.block == 1 && defaults.readback == "none" &&
          defaults.repeats == 1 && !defaults.sampling);
  const auto parse = [&](const std::string &text, bool graph = true) {
    {
      std::ofstream output(directory / "run-options.txt");
      output << text;
      require(bool(output));
    }
    return read_run_options(directory, graph);
  };
  const auto benchmark = parse("block 4\nbenchmark 1\nwarmups 1\nrepeats 3\n");
  require(benchmark.block == 4 && benchmark.benchmark && benchmark.warmups == 1 &&
          benchmark.repeats == 3);
  const auto cache = parse("readback kv\ncache_first_layer 2\ncache_last_layer 3\n");
  require(cache.cache_first == 2 && cache.cache_last == 3);
  const auto sampling = parse("seed 42\nmax_new_tokens 0\n", false);
  require(sampling.sampling && sampling.seed == 42 && sampling.maximum == 0);
  // Reject malformed or unsupported requests before a caller can create an SDK Context.
  for (const auto *text :
       {"unknown 1\n", "block 1\nblock 4\n", "block 1 extra\n", "block\n",
        "max_new_tokens 4294967296\n", "max_new_tokens -1\n", "block 2.0\n", "seed 0\n",
        "readback prefix\n", "readback kv\ncache_first_layer 3\ncache_last_layer 2\n",
        "readback kv\ncache_last_layer 2\n", "block 4\ndecoder_group_size 2\n",
        "block 4\nprefill_head last-token\n", "warmups 1\n"}) {
    rejects([&] { parse(text); });
  }
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
    run_options_contracts();
    contracts();
    std::cout << "Generation BOS, EOS, capacity, limits and consumed-token contracts passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
