#pragma once

#include "models/qwen3_5/config.hpp"
#include "np101/ops/graph_builder.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
enum class CompiledBlockKind { DeltaNet, FeedForward };

struct CompiledBlock {
  CompiledBlockKind kind = CompiledBlockKind::DeltaNet;
  np101::ops::CompiledGraphSpec graph;
  unsigned layer = 0;
  bool includes_input_norm = false;
  std::vector<unsigned> input_order, output_order;
  std::map<std::string, std::string> weight_sha256;
};

void validate_compiled_mixer(const CompiledBlock &artifact, const np101::WeightStore &weights,
                             const Config &config, unsigned layer);
void validate_compiled_feed_forward(const CompiledBlock &artifact,
                                    const np101::WeightStore &weights, const Config &config,
                                    unsigned layer);
std::map<unsigned, CompiledBlock> load_compiled_feed_forward(const std::filesystem::path &package,
                                                             const np101::WeightStore &weights,
                                                             const Config &config);
// Pin every file/source/SDK before constructing a Context. A partial package is a diagnostic slice.
std::map<unsigned, CompiledBlock> load_compiled_mixers(const std::filesystem::path &package,
                                                       const np101::WeightStore &weights,
                                                       const Config &config);
} // namespace specferry::models::qwen3_5
