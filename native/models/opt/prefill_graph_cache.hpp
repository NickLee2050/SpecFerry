#pragma once

#include "models/opt/layer.hpp"

#include <filesystem>
#include <vector>

namespace specferry::models::opt {
// Optional package selection; validate all identities/files before creating a Context.
void attach_prefill_graph_caches(const std::filesystem::path &package, const ModelConfig &config,
                                 const np101::WeightStore &weights,
                                 std::vector<CompiledPrefillLayer> &layers);
} // namespace specferry::models::opt
