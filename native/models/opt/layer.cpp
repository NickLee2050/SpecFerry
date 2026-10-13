#include "models/opt/layer.hpp"
#include "np101/memory_budget.hpp"
#include "vsi_nn_pub.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;

Tensor linear(GraphBuilder &g, const WeightStore &weights, Tensor input, const std::string &name,
              bool split_output_channels = false) {
  return g.linear(input, weights, weights.find(name + ".weight"), weights.find(name + ".bias"),
                  split_output_channels ? 2u : 1u);
}

Tensor norm(GraphBuilder &g, const WeightStore &weights, Tensor input, const std::string &name,
            float epsilon) {
  const auto width = input.spec.shape[0];
  const auto affine = [&](const std::string &suffix) {
    const auto &record = weights.find(name + suffix);
    if (record.spec.type != f16 || record.spec.shape != Shape{width}) {
      throw std::invalid_argument("unexpected OPT layer norm weight: " + record.name);
    }
    return g.weight_chunk(weights, record, {DataType::Float32, {width, 1}}, 0);
  };
  auto scale = affine(".weight");
  auto bias = affine(".bias");
  return g.layer_norm(input, scale, bias, epsilon);
}
} // namespace

std::vector<TensorSpec> compiled_decoder_inputs(const Config &config) {
  const auto dimension = config.kv_spec().head_dim;
  return {{f16, {config.hidden, 1}},
          {DataType::Float32, {dimension, config.capacity, config.heads}},
          {DataType::Float32, {dimension, config.capacity, config.heads}},
          {DataType::Int32, {2, config.heads}},
          {DataType::Float32, {config.capacity, 1}}};
}

void validate_compiled_decoder(const CompiledDecoder &compiled, const WeightStore &weights,
                               const Config &config, unsigned layer) {
  // Only this export geometry and precision policy have hardware evidence.
  if (config.hidden != 1024 || config.intermediate != 4096 || config.heads != 16 ||
      config.capacity != 64 || config.epsilon != 1e-5f || compiled.layer != layer ||
      compiled.weight_sha256.size() != 16 ||
      compiled.graph.graph_version != std::array<unsigned, 3>{1, 1, 34}) {
    throw std::invalid_argument("compiled OPT decoder differs from the accepted export contract");
  }
  const auto inputs = compiled_decoder_inputs(config);
  const std::vector<TensorSpec> outputs{inputs[0], inputs[1], inputs[2]};
  const auto matches = [](const auto &actual, const auto &expected) {
    if (actual.size() != expected.size()) {
      return false;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (actual[i].type != expected[i].type || actual[i].shape != expected[i].shape) {
        return false;
      }
    }
    return true;
  };
  if (!matches(compiled.graph.inputs, inputs) || !matches(compiled.graph.outputs, outputs)) {
    throw std::invalid_argument("compiled OPT decoder port contract mismatch");
  }
  std::size_t payload = 0;
  for (const std::string name :
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
        "self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"}) {
    for (const std::string part : {"weight", "bias"}) {
      const auto key = name + "." + part;
      const auto &record = weights.find(config.prefix(layer) + key);
      const auto identity = compiled.weight_sha256.find(key);
      if (identity == compiled.weight_sha256.end() || identity->second != record.sha256) {
        throw std::invalid_argument("compiled OPT decoder source weight differs: " + record.name);
      }
      // Export widens normalization affine parameters to FP32 once.
      payload += record.spec.bytes() * (name.find("layer_norm") != std::string::npos ? 2 : 1);
    }
  }
  if (compiled.graph.constant_bytes < payload ||
      compiled.graph.constant_bytes > segment_payload_limit) {
    throw std::invalid_argument("compiled OPT decoder does not account for its constant payload");
  }
}

void validate_compiled_ffn(const CompiledFfn &compiled, const WeightStore &weights,
                           const Config &config, unsigned layer, unsigned tokens) {
  const TensorSpec port{f16, {config.hidden, tokens}};
  const auto matches = [&](const std::vector<TensorSpec> &ports) {
    return ports.size() == 1 && ports.front().type == port.type &&
           ports.front().shape == port.shape;
  };
  if ((tokens != 1 && tokens != 16) || compiled.layer != layer || !matches(compiled.graph.inputs) ||
      !matches(compiled.graph.outputs)) {
    throw std::invalid_argument("compiled OPT FFN requires matching layer and FP16 block ports");
  }
  const std::array<std::string, 4> names{"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
  std::size_t payload = 0;
  for (std::size_t index = 0; index < names.size(); ++index) {
    const auto &weight = weights.find(config.prefix(layer) + names[index]);
    if (compiled.weight_sha256[index] != weight.sha256) {
      throw std::invalid_argument("compiled OPT FFN source weight differs: " + weight.name);
    }
    payload += weight.spec.bytes();
  }
  // Compiled packages can include padded or packed constants beyond the source weights.
  if (compiled.graph.constant_bytes < payload ||
      compiled.graph.constant_bytes > segment_payload_limit) {
    throw std::invalid_argument("compiled OPT FFN constant payload does not cover source weights");
  }
}

Projections build_projections(GraphBuilder &g, const WeightStore &weights, const Config &config,
                              unsigned layer, Tensor hidden) {
  const auto prefix = config.prefix(layer);
  auto query = linear(g, weights, hidden, prefix + "self_attn.q_proj");
  query = g.binary(VSI_NN_OP_MULTIPLY, query,
                   g.scalar(1.0f / std::sqrt(float(config.kv_spec().head_dim)), f16));
  auto key = linear(g, weights, hidden, prefix + "self_attn.k_proj");
  auto value = linear(g, weights, hidden, prefix + "self_attn.v_proj");
  return {query, key, value};
}

AttentionTail build_attention_tail(GraphBuilder &g, const WeightStore &weights,
                                   const Config &config, unsigned layer, Tensor hidden,
                                   Tensor attended) {
  const auto prefix = config.prefix(layer);
  auto mixer = linear(g, weights, attended, prefix + "self_attn.out_proj");
  auto residual = g.binary(VSI_NN_OP_ADD, hidden, mixer);
  auto normalized = norm(g, weights, residual, prefix + "self_attn_layer_norm", config.epsilon);
  return {mixer, residual, normalized};
}

FfnFinish build_ffn_finish(GraphBuilder &g, const WeightStore &weights, const Config &config,
                           unsigned layer, Tensor normalized, Tensor mlp) {
  auto residual = g.binary(VSI_NN_OP_ADD, normalized, mlp);
  auto output =
      norm(g, weights, residual, config.prefix(layer) + "final_layer_norm", config.epsilon);
  return {residual, output};
}

std::map<std::string, Tensor> build_decoder_tail(GraphBuilder &g, const WeightStore &weights,
                                                 const Config &config, unsigned layer,
                                                 Tensor hidden, Tensor attended,
                                                 bool split_fc1_output) {
  const auto prefix = config.prefix(layer);
  const auto attention = build_attention_tail(g, weights, config, layer, hidden, attended);
  auto fc1 = linear(g, weights, attention.normalized, prefix + "fc1", split_fc1_output);
  auto activation = g.unary(VSI_NN_OP_RELU, fc1, f16);
  auto mlp = linear(g, weights, activation, prefix + "fc2");
  const auto finish = build_ffn_finish(g, weights, config, layer, attention.normalized, mlp);
  return {{"mixer", attention.mixer},
          {"attention_residual", attention.residual},
          {"attention_norm", attention.normalized},
          {"fc1", fc1},
          {"activation", activation},
          {"mlp", mlp},
          {"ffn_residual", finish.residual},
          {"output", finish.output}};
}
} // namespace specferry::models::opt
