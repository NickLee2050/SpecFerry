#pragma once

#include "np101/tensor_spec.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace specferry::np101 {
struct WeightRecord {
  std::string name;
  TensorSpec spec;
  std::uint64_t offset;
  std::uint64_t bytes;
  std::string sha256;
};

// Validates the native index before exposing bounded reads. The caller must also
// verify deployment-manifest.json to bind this pack to the fixed model revision.
class WeightStore {
public:
  explicit WeightStore(const std::filesystem::path &directory);

  const std::vector<WeightRecord> &records() const { return records_; }

  const WeightRecord &find(const std::string &name) const;

  void verify() const;

  std::vector<std::uint8_t> read(const WeightRecord &record, std::uint64_t relative_offset,
                                 std::size_t bytes) const;

private:
  std::filesystem::path data_path_;
  std::vector<WeightRecord> records_;
};
} // namespace specferry::np101
