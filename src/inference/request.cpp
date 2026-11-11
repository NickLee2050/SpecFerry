#include "inference/request.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace specferry::inference {
std::vector<std::int32_t> read_prompt(std::istream &input, unsigned vocabulary, unsigned capacity) {
  std::vector<std::int32_t> tokens;
  std::int32_t token;
  while (input >> token) {
    if (token < 0 || unsigned(token) >= vocabulary) {
      throw std::invalid_argument("prompt token exceeds vocabulary");
    }
    tokens.push_back(token);
  }
  if (!input.eof() || tokens.empty() || tokens.size() > capacity) {
    throw std::invalid_argument("invalid prompt tokens or capacity");
  }
  return tokens;
}

std::vector<std::vector<std::int32_t>> read_requests(const std::filesystem::path &path,
                                                     unsigned vocabulary, unsigned capacity) {
  std::ifstream input(path);
  std::vector<std::vector<std::int32_t>> requests;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    requests.push_back(read_prompt(stream, vocabulary, capacity));
  }
  if (!input.eof() || requests.empty() || requests.size() > 64) {
    throw std::invalid_argument("expected one to 64 request lines");
  }
  return requests;
}

RunSettings::RunSettings(const std::filesystem::path &directory) {
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

bool RunSettings::contains(const std::string &key) const { return values_.count(key); }

std::string RunSettings::text(const std::string &key, const std::string &fallback) {
  const auto found = values_.find(key);
  const auto value = found == values_.end() ? fallback : found->second;
  values_.erase(key);
  return value;
}

unsigned RunSettings::number(const std::string &key, unsigned fallback, unsigned minimum,
                             unsigned maximum) {
  const auto value = text(key, std::to_string(fallback));
  unsigned result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < minimum ||
      result > maximum) {
    throw std::invalid_argument("invalid run option: " + key);
  }
  return result;
}

void RunSettings::finish() const {
  if (!values_.empty()) {
    throw std::invalid_argument("unknown run option: " + values_.begin()->first);
  }
}

RunOptions read_run_options(const std::filesystem::path &directory, bool graph) {
  RunSettings settings(directory);
  RunOptions result;
  // Context capacity and the external workload deadline bound graph generation.
  result.maximum =
      settings.number("max_new_tokens", 8, graph ? 1 : 0, std::numeric_limits<unsigned>::max());
  result.warmups = settings.number("warmups", 0, 0, 5);
  result.repeats = settings.number("repeats", 1, 1, 20);
  result.sampling = settings.contains("seed");
  result.seed = settings.number("seed", 0, 0, std::numeric_limits<unsigned>::max());
  if (graph) {
    result.block = settings.number("block", 1, 1, 16);
    result.variant = settings.text("variant", "merged-kv");
    result.prefill_head = settings.text("prefill_head", "every-token");
    result.readback = settings.text("readback", "none");
    result.readback_transport = settings.text("readback_transport", "sdk");
    result.projection_backend = settings.text("projection_backend", "auto");
    result.history_mode = settings.text("history_mode", "reset");
    result.kv_transfer = settings.text("kv_transfer", "device-copy");
    result.cache_first = settings.number("cache_first_layer", 0, 0, 32);
    result.cache_last = settings.number("cache_last_layer", 0, 0, 32);
    result.benchmark = settings.number("benchmark", 0, 0, 1);
    result.lifecycle = settings.number("lifecycle", 0, 0, 1);
    if ((result.kv_transfer != "device-copy" && result.kv_transfer != "mapped-slot") ||
        (result.kv_transfer == "mapped-slot" &&
         (result.variant != "compiled-fullgraph" || result.history_mode != "reset"))) {
      throw std::invalid_argument("mapped-slot KV requires compiled fullgraph with request reset");
    }
    if ((result.history_mode != "reset" && result.history_mode != "reuse") ||
        (result.history_mode != "reset" && (result.benchmark || result.lifecycle)) ||
        (result.projection_backend != "auto" && result.variant != "compiled-ffn-finish") ||
        (result.projection_backend != "auto" && result.projection_backend != "fcl" &&
         result.projection_backend != "nbg-qkv" && result.projection_backend != "nbg-output" &&
         result.projection_backend != "nbg") ||
        result.sampling ||
        (result.variant != "merged-kv" && result.variant != "merged-kv-fc1-split2" &&
         result.variant != "compiled-ffn-finish" && result.variant != "compiled-decoder" &&
         result.variant != "compiled-fullgraph") ||
        (result.variant == "compiled-ffn-finish" &&
         ((result.block != 1 && result.block != 16) || result.lifecycle)) ||
        (result.variant == "compiled-decoder" && (result.block != 1 || result.lifecycle)) ||
        (result.variant == "compiled-fullgraph" &&
         (result.lifecycle || !((result.block == 1 && result.prefill_head == "every-token") ||
                                (result.block == 16 && result.prefill_head == "last-token" &&
                                 result.kv_transfer == "mapped-slot")))) ||
        (result.block != 1 && result.block != 4 && result.block != 8 && result.block != 16) ||
        (result.prefill_head != "every-token" && result.prefill_head != "last-token") ||
        (result.readback != "none" && result.readback != "logits" && result.readback != "kv") ||
        (result.readback_transport != "sdk" && result.readback_transport != "mapped") ||
        (result.readback_transport == "mapped" && result.readback == "none") ||
        ((result.readback == "kv") != (result.cache_first && result.cache_last)) ||
        (result.readback != "kv" && (result.cache_first || result.cache_last)) ||
        result.cache_first > result.cache_last ||
        (result.benchmark && (result.lifecycle || result.readback != "none")) ||
        (result.lifecycle && result.readback != "logits") ||
        (!result.benchmark && (result.warmups || result.repeats != 1))) {
      throw std::invalid_argument("unsupported graph run options");
    }
  }
  settings.finish();
  return result;
}

} // namespace specferry::inference
