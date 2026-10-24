#include "inference/generation.hpp"
#include "inference/generation_io.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
          defaults.repeats == 1 && !defaults.sampling && defaults.kv_transfer == "device-copy");
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
  const auto head_skip = parse("block 4\nprefill_head last-token\n");
  require(head_skip.block == 4 && head_skip.prefill_head == "last-token");
  for (unsigned block : {8, 16}) {
    const auto larger = parse("block " + std::to_string(block) +
                              "\nvariant merged-kv-fc1-split2\nprefill_head last-token\n");
    require(larger.block == block && larger.prefill_head == "last-token");
  }
  const auto split_fc1 = parse("block 4\nvariant merged-kv-fc1-split2\n");
  require(split_fc1.block == 4 && split_fc1.variant == "merged-kv-fc1-split2");
  const auto compiled_prefill = parse("block 16\nvariant compiled-ffn-finish\n"
                                      "projection_backend nbg\nprefill_head last-token\n");
  require(compiled_prefill.block == 16 && compiled_prefill.projection_backend == "nbg");
  require(parse("history_mode reuse\n").history_mode == "reuse");
  require(parse("variant compiled-fullgraph\nkv_transfer mapped-slot\n").kv_transfer ==
          "mapped-slot");
  for (const auto *extra : {"variant compiled-decoder\n", "variant merged-kv\n",
                            "variant compiled-fullgraph\nhistory_mode reuse\n",
                            "variant compiled-fullgraph\nblock 16\n"}) {
    rejects([&] { parse(std::string(extra) + "kv_transfer mapped-slot\n"); });
  }
  rejects([&] { parse("kv_transfer unknown\n"); });
  rejects([&] { parse("history_mode compare\n"); });
  const auto cache = parse("readback kv\ncache_first_layer 2\ncache_last_layer 3\n");
  require(cache.cache_first == 2 && cache.cache_last == 3);
  const auto mapped = parse("readback kv\ncache_first_layer 23\ncache_last_layer 24\n"
                            "readback_transport mapped\n");
  require(mapped.readback_transport == "mapped" && mapped.cache_first == 23 &&
          mapped.cache_last == 24);
  const auto sampling = parse("seed 42\nmax_new_tokens 0\n", false);
  require(sampling.sampling && sampling.seed == 42 && sampling.maximum == 0);
  // Reject malformed or unsupported requests before a caller can create an SDK Context.
  for (const auto *text : {"unknown 1\n",
                           "block 1\nblock 4\n",
                           "block 1 extra\n",
                           "block\n",
                           "max_new_tokens 4294967296\n",
                           "max_new_tokens -1\n",
                           "block 2.0\n",
                           "seed 0\n",
                           "readback prefix\n",
                           "readback kv\ncache_first_layer 3\ncache_last_layer 2\n",
                           "readback kv\ncache_last_layer 2\n",
                           "block 4\ndecoder_group_size 2\n",
                           "prefill_head unknown\n",
                           "warmups 1\n",
                           "readback_transport unknown\n",
                           "readback_transport mapped\n",
                           "variant merged-kv-local-const\n",
                           "variant merged-kv-local-mutable\n",
                           "decoder_group_size 1\n",
                           "variant compiled-ffn-finish\nblock 4\n",
                           "variant compiled-decoder\nblock 16\n",
                           "variant compiled-fullgraph\nblock 16\n"}) {
    rejects([&] { parse(text); });
  }
  rejects([&] { parse("history_mode unknown\n"); });
  rejects([&] { parse("history_mode reuse\nbenchmark 1\n"); });
}

void conversation_contracts() {
  using namespace specferry::inference;
  std::vector<std::int32_t> consumed, appended;
  unsigned resets = 0;
  bool fail = false;
  const GenerationExecutor executor{[&] {
                                      ++resets;
                                      consumed.clear();
                                    },
                                    [&](std::int32_t token) { consumed.push_back(token); },
                                    [&] {
                                      if (fail) {
                                        throw std::runtime_error("executor failed");
                                      }
                                      return consumed.back() + 1;
                                    },
                                    [&](const std::vector<std::int32_t> &tokens) {
                                      appended = tokens;
                                      consumed.insert(consumed.end(), tokens.begin(), tokens.end());
                                    }};
  GenerationHistory history;
  const GenerationPolicy policy{100, 32, 2, 99};
  auto first = generate_tokens(policy, executor, {2, 5}, 2, {}, {}, &history);
  require(first.tokens == std::vector<std::int32_t>({6, 7}) && resets == 1 &&
          history.consumed == 3 && consumed == std::vector<std::int32_t>({2, 5, 6}));
  const std::vector<std::int32_t> second_prompt{2, 5, 6, 7, 10, 11};
  const auto second = generate_tokens(policy, executor, second_prompt, 2, {}, {}, &history);
  require(second.tokens == std::vector<std::int32_t>({12, 13}) && resets == 1 &&
          appended == std::vector<std::int32_t>({7, 10, 11}) && history.consumed == 7 &&
          consumed == std::vector<std::int32_t>({2, 5, 6, 7, 10, 11, 12}));
  const auto before = consumed;
  rejects([&] { generate_tokens(policy, executor, second_prompt, 2, {}, {}, &history); });
  rejects([&] { generate_tokens(policy, executor, history.tokens, 0, {}, {}, &history); });
  require(consumed == before && resets == 1 && history.valid);
  const auto replay = generate_tokens(policy, executor, second_prompt, 2);
  require(replay.tokens == second.tokens && consumed == before && resets == 2);
  fail = true;
  rejects([&] { generate_tokens(policy, executor, history.tokens, 1, {}, {}, &history); });
  require(!history.valid);
  rejects([&] { generate_tokens(policy, executor, history.tokens, 1, {}, {}, &history); });

  for (unsigned position = 0; position < 16; ++position) {
    unsigned offset = 0;
    for (auto chunk : plan_prefill(48, 16, position)) {
      require(chunk.begin == offset && (chunk.count == 1 || chunk.count == 16));
      require(chunk.count == 1 || (position + offset) % 16 == 0);
      offset += chunk.count;
    }
    require(offset == 48);
  }
  const auto aligned = plan_prefill(34, 16, 15);
  require(aligned.size() == 4 && aligned[0].count == 1 && aligned[1].count == 16 &&
          aligned[2].count == 16 && aligned[3].count == 1);
}

void prefill_replay_contracts() {
  using namespace specferry::inference;
  require(prefill_replay_tokens(51, 27, 16, 0.843, 1.438) == 3);
  require(prefill_replay_tokens(93, 28, 16, 0.843, 1.438) == 13);
  require(prefill_replay_tokens(15, 32, 16, 0.843, 1.438) == 0);
  require(prefill_replay_tokens(16, 32, 16, 0.843, 1.438) == 0);
  require(prefill_replay_tokens(3, 1, 16, 0.843, 1.438) == 0);
  require(prefill_replay_tokens(51, 27, 16, 0, 1.438) == 0);
  require(prefill_replay_tokens(51, 27, 16, 0.843, std::numeric_limits<double>::infinity()) == 0);
  rejects([&] { prefill_replay_tokens(51, 27, 0, 0.843, 1.438); });

  std::vector<std::int32_t> consumed, replayed;
  GenerationHistory history;
  unsigned preparations = 0;
  bool fail = false;
  GenerationExecutor executor{
      [&] { consumed.clear(); }, [&](std::int32_t token) { consumed.push_back(token); },
      [&] { return consumed.back() + 1; },
      [&](const std::vector<std::int32_t> &tokens) {
        if (fail) {
          throw std::runtime_error("prefill failed after rewind");
        }
        replayed = tokens;
        consumed.insert(consumed.end(), tokens.begin(), tokens.end());
      },
      [&](unsigned prefix, unsigned appended) {
        require(!history.valid && prefix == consumed.size() && appended > 0);
        ++preparations;
        consumed.resize(prefix - 3);
        return 3;
      }};
  const GenerationPolicy policy{100, 32, 2, 99};
  generate_tokens(policy, executor, {2, 3, 4, 5, 6, 7}, 2, {}, {}, &history);
  require(preparations == 0 && history.consumed == 7);
  auto prompt = history.tokens;
  prompt.insert(prompt.end(), {10, 11});
  auto invalid = prompt;
  invalid[0] = 0;
  const auto before = consumed;
  rejects([&] { generate_tokens(policy, executor, invalid, 1, {}, {}, &history); });
  require(history.valid && preparations == 0 && consumed == before);
  const auto result = generate_tokens(policy, executor, prompt, 1, {}, {}, &history);
  require(result.replayed_prompt_tokens == 3 && result.consumed == prompt.size() &&
          result.tokens == std::vector<std::int32_t>{12} && consumed == prompt &&
          replayed == std::vector<std::int32_t>({6, 7, 8, 9, 10, 11}) && history.valid);
  fail = true;
  rejects([&] { generate_tokens(policy, executor, history.tokens, 1, {}, {}, &history); });
  require(!history.valid);
  rejects([&] { generate_tokens(policy, executor, history.tokens, 1, {}, {}, &history); });
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

  std::vector<GenerationPhase> phases;
  const auto observe = [&](GenerationPhase phase) { phases.push_back(phase); };
  generate_tokens({10, 8, 2, 2}, executor, {2, 7}, 8, {}, observe);
  require(phases == std::vector<GenerationPhase>({GenerationPhase::Reset, GenerationPhase::Prefill,
                                                  GenerationPhase::Decode}));
  phases.clear();
  generate_tokens({10, 2, 2, 2}, executor, {2, 7}, 8, {}, observe);
  require(phases ==
          std::vector<GenerationPhase>({GenerationPhase::Reset, GenerationPhase::Prefill}));
  phases.clear();
  generate_tokens({10, 8, 2, 2}, executor, {2}, 0, {}, observe);
  require(phases == std::vector<GenerationPhase>({GenerationPhase::Reset}));

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
    for (unsigned block : {1, 4, 8, 16}) {
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
  rejects([&] { plan_prefill(8, 17); });
}
} // namespace

int main() {
  try {
    run_options_contracts();
    contracts();
    conversation_contracts();
    prefill_replay_contracts();
    std::cout << "Generation BOS, EOS, capacity, limits and consumed-token contracts passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
