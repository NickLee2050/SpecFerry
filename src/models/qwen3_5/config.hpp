#pragma once

#include "devices/np101/api/component_spec.hpp"
#include "devices/np101/api/tensor_spec.hpp"
#include "devices/np101/api/weights.hpp"

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
} // namespace specferry::models::qwen3_5
