#pragma once

#include "models/opt/config.hpp"
#include "np101/ops/graph_builder.hpp"

#include <array>
#include <map>
#include <string>
#include <vector>

namespace specferry::models::opt {
struct Projections {
  np101::ops::Tensor query, key, value;
};

struct AttentionTail {
  np101::ops::Tensor mixer, residual, normalized;
};

struct FfnFinish {
  np101::ops::Tensor residual, output;
};

// A compiled FFN owns fc1/fc2 weights and biases. The source identities bind the
// artifact to one checkpoint layer; callers must also verify the artifact file.
struct CompiledFfn {
  np101::ops::CompiledGraphSpec graph;
  unsigned layer;
  // Fixed order: fc1.weight, fc1.bias, fc2.weight, fc2.bias.
  std::array<std::string, 4> weight_sha256;
};

// Full decoder artifacts use FP32 head-major KV and fixed M1 ports. The loader
// verifies each NBG file hash; these identities bind it to the deployment weights.
struct CompiledDecoder {
  np101::ops::CompiledGraphSpec graph;
  unsigned layer;
  std::map<std::string, std::string> weight_sha256;
};

std::vector<np101::TensorSpec> compiled_decoder_inputs(const Config &config);
void validate_compiled_decoder(const CompiledDecoder &compiled, const np101::WeightStore &weights,
                               const Config &config, unsigned layer);

void validate_compiled_ffn(const CompiledFfn &compiled, const np101::WeightStore &weights,
                           const Config &config, unsigned layer, unsigned tokens = 1);

// Construction only: callers choose graph boundaries, cache storage and execution.
// All tensors have SDK shape [hidden,tokens]; no checkpoint policy lives in shared ops.
Projections build_projections(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                              const Config &config, unsigned layer, np101::ops::Tensor hidden);
AttentionTail build_attention_tail(np101::ops::GraphBuilder &graph,
                                   const np101::WeightStore &weights, const Config &config,
                                   unsigned layer, np101::ops::Tensor hidden,
                                   np101::ops::Tensor attended);
FfnFinish build_ffn_finish(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                           const Config &config, unsigned layer, np101::ops::Tensor normalized,
                           np101::ops::Tensor mlp);
std::map<std::string, np101::ops::Tensor>
build_decoder_tail(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                   const Config &config, unsigned layer, np101::ops::Tensor hidden,
                   np101::ops::Tensor attended, bool split_fc1_output = false);
} // namespace specferry::models::opt
