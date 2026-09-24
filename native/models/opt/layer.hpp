#pragma once

#include "models/opt/config.hpp"
#include "np101/ops/graph_builder.hpp"

#include <map>
#include <string>

namespace specferry::models::opt {
struct Projections {
  np101::ops::Tensor query, key, value;
};

// Construction only: callers choose graph boundaries, cache storage and execution.
// All tensors have SDK shape [hidden,tokens]; no checkpoint policy lives in shared ops.
Projections build_projections(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                              const Config &config, unsigned layer, np101::ops::Tensor hidden);
std::map<std::string, np101::ops::Tensor> build_decoder_tail(np101::ops::GraphBuilder &graph,
                                                             const np101::WeightStore &weights,
                                                             const Config &config, unsigned layer,
                                                             np101::ops::Tensor hidden,
                                                             np101::ops::Tensor attended);
} // namespace specferry::models::opt
