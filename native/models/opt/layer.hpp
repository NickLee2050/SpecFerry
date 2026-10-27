#pragma once

#include "models/opt/config.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/graph_cache.hpp"

#include <array>
#include <filesystem>
#include <map>
#include <optional>
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
  // When present, the artifact also adds its input residual and applies FP32
  // final_layer_norm. Empty identities retain the original MLP-only contract.
  std::array<std::string, 2> final_norm_sha256;

  bool includes_final_norm() const { return !final_norm_sha256[0].empty(); }
};

// Full decoder artifacts use FP32 head-major KV and fixed M1 ports. The loader
// verifies each NBG file hash; these identities bind it to the deployment weights.
struct CompiledDecoder {
  np101::ops::CompiledGraphSpec graph;
  unsigned layer;
  std::map<std::string, std::string> weight_sha256;
  std::array<unsigned, 5> input_order{0, 1, 2, 3, 4};
  std::array<unsigned, 3> output_order{0, 1, 2};
};

// M16 NBG: attention plus normalization, or a complete decoder. FP32 cache
// ports use [width,capacity,heads]; ordinary FFNs retain their original weights.
struct CompiledPrefillLayer {
  np101::ops::CompiledGraphSpec graph;
  unsigned layer = 0;
  bool full_decoder = false;
  std::optional<np101::ops::VerifiedGraphCache> verify_cache;
};

std::vector<CompiledPrefillLayer>
load_compiled_block_prefill(const std::filesystem::path &package,
                            const std::filesystem::path &deployment, const ModelConfig &config,
                            const np101::WeightStore &weights);

struct CompiledFullModel {
  np101::ops::CompiledGraphSpec graph;
  std::vector<std::string> input_names, output_names;
  std::vector<unsigned> input_order, output_order;
  std::map<std::string, std::string> weight_sha256;

  // The package validates every cache output against this fixed port contract.
  bool current_cache_outputs() const;
};

struct CompiledModelPackage {
  std::vector<CompiledDecoder> decoders;
  CompiledFullModel full;
};

CompiledModelPackage load_compiled_model(const std::filesystem::path &package,
                                         const std::filesystem::path &deployment,
                                         const ModelConfig &config,
                                         const np101::WeightStore &weights);
void validate_compiled_full_model(const CompiledFullModel &compiled,
                                  const np101::WeightStore &weights, const ModelConfig &config);

struct CompiledHead {
  np101::ops::CompiledGraphSpec graph;
  // Fixed order: project_out.weight, tied embed_tokens.weight.
  std::array<std::string, 2> weight_sha256;
};

struct CompiledProjections {
  np101::ops::CompiledGraphSpec qkv, output;
  unsigned layer = 0;
  // Semantic Q/K/V order differs from SDK port order in some compiled exports.
  std::array<unsigned, 3> qkv_order{};
  // Q/K/V/output, each followed by its bias; checkpoint identities remain model-owned.
  std::array<std::string, 8> weight_sha256;
};

// The portable package pins deployment, source weights, NBGs and the accepted SDK.
// Call before Context construction; C64/C128 M1 fused FFNs have acceptance evidence.
std::vector<CompiledFfn> load_compiled_ffn(const std::filesystem::path &package,
                                           const std::filesystem::path &deployment,
                                           const ModelConfig &config,
                                           const np101::WeightStore &weights, CompiledHead &head,
                                           std::vector<CompiledProjections> *projections = nullptr);
// M16 manifests bind their M1 package and deployment; no SDK calls are made.
// Version 1: mixed M1, four ordinary layers. Version 2: full-model M1, six ordinary layers.
std::vector<CompiledFfn> load_compiled_prefill(const std::filesystem::path &package,
                                               const std::filesystem::path &deployment,
                                               const ModelConfig &config,
                                               const np101::WeightStore &weights);
void validate_compiled_projections(const CompiledProjections &projections,
                                   const np101::WeightStore &weights, const Config &config,
                                   unsigned layer);
void validate_compiled_head(const CompiledHead &head, const np101::WeightStore &weights,
                            const ModelConfig &config);

std::vector<np101::TensorSpec> compiled_decoder_inputs(const Config &config);
void validate_compiled_decoder(const CompiledDecoder &compiled, const np101::WeightStore &weights,
                               const Config &config, unsigned layer);

void validate_compiled_ffn(const CompiledFfn &compiled, const np101::WeightStore &weights,
                           const Config &config, unsigned layer, unsigned tokens = 1);

// Construction only: callers choose graph boundaries, cache storage and execution.
// All tensors have SDK shape [hidden,tokens]; no checkpoint policy lives in shared ops.
Projections
build_projections(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                  const Config &config, unsigned layer, np101::ops::Tensor hidden,
                  np101::ops::LinearBackend backend = np101::ops::LinearBackend::MatMul);
AttentionTail
build_attention_tail(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                     const Config &config, unsigned layer, np101::ops::Tensor hidden,
                     np101::ops::Tensor attended,
                     np101::ops::LinearBackend backend = np101::ops::LinearBackend::MatMul);
AttentionTail build_attention_finish(np101::ops::GraphBuilder &graph,
                                     const np101::WeightStore &weights, const Config &config,
                                     unsigned layer, np101::ops::Tensor hidden,
                                     np101::ops::Tensor projected);
FfnFinish build_ffn_finish(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                           const Config &config, unsigned layer, np101::ops::Tensor normalized,
                           np101::ops::Tensor mlp);
std::map<std::string, np101::ops::Tensor>
build_decoder_tail(np101::ops::GraphBuilder &graph, const np101::WeightStore &weights,
                   const Config &config, unsigned layer, np101::ops::Tensor hidden,
                   np101::ops::Tensor attended, bool split_fc1_output = false);
} // namespace specferry::models::opt
