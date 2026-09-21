#pragma once

#include "np101/component_spec.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
enum class MixerKind { DeltaNet, Attention };

struct Config {
  unsigned hidden;
  unsigned intermediate;
  unsigned query_heads;
  unsigned rotary_dim;
  float rope_theta;
  float epsilon;
  np101::KvSpec kv;
  np101::DeltaSpec delta;
  std::vector<MixerKind> layer_types;

  void validate() const;
  np101::TensorSpec hidden_spec() const;
  std::string layer_prefix(unsigned layer) const;
  MixerKind mixer(unsigned layer) const;
};

void validate_layer_weights(const np101::WeightStore &weights, const Config &config, unsigned layer,
                            bool include_feed_forward);
// A small, versioned component fixture written from the validated model config.
Config read_config(const std::filesystem::path &path);
const np101::WeightRecord &resolve_weight(const np101::WeightStore &weights,
                                          const std::string &name);
} // namespace specferry::models::qwen3_5
