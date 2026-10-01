#pragma once

#include "inference/generation.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace specferry::inference {
struct RunOptions {
  unsigned maximum = 8, warmups = 0, repeats = 1, seed = 0;
  unsigned block = 1, decoder_group_size = 1, cache_first = 0, cache_last = 0;
  bool sampling = false, benchmark = false, lifecycle = false;
  std::string variant = "merged-kv", prefill_head = "every-token", readback = "none";
};

// Each CLI consumes its own keys, then rejects leftovers before initializing the SDK.
class RunSettings {
public:
  explicit RunSettings(const std::filesystem::path &directory) {
    const auto path = directory / "run-options.txt";
    if (std::filesystem::exists(path)) {
      std::ifstream input(path);
      std::string line;
      while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key, value, extra;
        if (!(fields >> key)) {
          continue;
        }
        if (!(fields >> value) || fields >> extra || !values_.emplace(key, value).second) {
          throw std::invalid_argument("run-options.txt requires unique key/value lines");
        }
      }
      if (!input.eof()) {
        throw std::runtime_error("cannot read run-options.txt");
      }
    }
  }

  bool contains(const std::string &key) const { return values_.count(key); }

  std::string text(const std::string &key, const std::string &fallback) {
    const auto found = values_.find(key);
    const auto value = found == values_.end() ? fallback : found->second;
    values_.erase(key);
    return value;
  }

  unsigned number(const std::string &key, unsigned fallback, unsigned minimum, unsigned maximum) {
    const auto value = text(key, std::to_string(fallback));
    unsigned result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < minimum ||
        result > maximum) {
      throw std::invalid_argument("invalid run option: " + key);
    }
    return result;
  }

  void finish() const {
    if (!values_.empty()) {
      throw std::invalid_argument("unknown run option: " + values_.begin()->first);
    }
  }

private:
  std::map<std::string, std::string> values_;
};

inline RunOptions read_run_options(const std::filesystem::path &directory, bool graph) {
  RunSettings settings(directory);
  RunOptions result;
  result.maximum = settings.number("max_new_tokens", 8, graph ? 1 : 0,
                                   graph ? 32 : std::numeric_limits<unsigned>::max());
  result.warmups = settings.number("warmups", 0, 0, 5);
  result.repeats = settings.number("repeats", 1, 1, 20);
  result.sampling = settings.contains("seed");
  result.seed = settings.number("seed", 0, 0, std::numeric_limits<unsigned>::max());
  if (graph) {
    result.block = settings.number("block", 1, 1, 4);
    result.decoder_group_size = settings.number("decoder_group_size", 1, 1, 2);
    result.variant = settings.text("variant", "merged-kv");
    result.prefill_head = settings.text("prefill_head", "every-token");
    result.readback = settings.text("readback", "none");
    result.cache_first = settings.number("cache_first_layer", 0, 0, 32);
    result.cache_last = settings.number("cache_last_layer", 0, 0, 32);
    result.benchmark = settings.number("benchmark", 0, 0, 1);
    result.lifecycle = settings.number("lifecycle", 0, 0, 1);
    if (result.sampling || (result.block != 1 && result.block != 4) ||
        (result.prefill_head != "every-token" && result.prefill_head != "last-token") ||
        (result.readback != "none" && result.readback != "logits" && result.readback != "kv") ||
        ((result.readback == "kv") != (result.cache_first && result.cache_last)) ||
        (result.readback != "kv" && (result.cache_first || result.cache_last)) ||
        result.cache_first > result.cache_last ||
        (result.benchmark && (result.lifecycle || result.readback != "none")) ||
        (result.lifecycle && result.readback != "logits") ||
        (!result.benchmark && (result.warmups || result.repeats != 1)) ||
        (result.block == 4 && (result.variant != "merged-kv" || result.decoder_group_size != 1 ||
                               result.prefill_head != "every-token"))) {
      throw std::invalid_argument("unsupported graph run options");
    }
  }
  settings.finish();
  return result;
}

template <class T> void write_array(std::ostream &out, const std::vector<T> &values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    out << (i ? "," : "") << values[i];
  }
  out << ']';
}

inline void write_generation(std::ostream &out, const Generation &result) {
  out << std::setprecision(17) << "\"tokens\":";
  write_array(out, result.tokens);
  out << ",\"consumed\":" << result.consumed << ",\"stop_reason\":\"" << result.stop_reason
      << "\",\"prefill_seconds\":" << result.prefill_seconds
      << ",\"first_token_seconds\":" << result.first_token_seconds
      << ",\"total_seconds\":" << result.total_seconds << ",\"token_seconds\":";
  write_array(out, result.token_seconds);
}
} // namespace specferry::inference
