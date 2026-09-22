#pragma once

#include "np101/component_spec.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace specferry::models::opt {
struct Config {
  unsigned hidden;
  unsigned intermediate;
  unsigned heads;
  unsigned layers;
  unsigned capacity;
  float epsilon;

  void validate() const;
  np101::TensorSpec hidden_spec() const;
  np101::KvSpec kv_spec() const;
  std::string prefix(unsigned layer) const;
};

Config read_config(const std::filesystem::path &path);
void validate_weights(const np101::WeightStore &weights, const Config &config,
                      const std::vector<unsigned> &selected);
} // namespace specferry::models::opt
