#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <map>
#include <string>
#include <vector>

namespace specferry::inference {
std::vector<std::int32_t> read_prompt(std::istream &input, unsigned vocabulary, unsigned capacity);
// Native requests remain token IDs; checkpoint tokenization belongs to the deployment tools.
std::vector<std::vector<std::int32_t>> read_requests(const std::filesystem::path &path,
                                                     unsigned vocabulary, unsigned capacity);

struct RunOptions {
  unsigned maximum = 8, warmups = 0, repeats = 1, seed = 0;
  unsigned block = 1, cache_first = 0, cache_last = 0;
  bool sampling = false, benchmark = false, lifecycle = false;
  std::string variant = "merged-kv", prefill_head = "every-token", readback = "none";
  std::string readback_transport = "sdk";
  std::string projection_backend = "auto";
  std::string history_mode = "reset";
  std::string kv_transfer = "device-copy";
};

// Consume recognized options, then reject leftovers before opening the device.
class RunSettings {
public:
  RunSettings(const std::filesystem::path &directory);
  bool contains(const std::string &key) const;
  std::string text(const std::string &key, const std::string &fallback);
  unsigned number(const std::string &key, unsigned fallback, unsigned minimum, unsigned maximum);
  void finish() const;

private:
  std::map<std::string, std::string> values_;
};

RunOptions read_run_options(const std::filesystem::path &directory, bool graph);
} // namespace specferry::inference
