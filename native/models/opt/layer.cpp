#include "models/opt/layer.hpp"
#include "vsi_nn_pub.h"

#include <cmath>
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

std::map<std::string, Tensor> build_decoder_tail(GraphBuilder &g, const WeightStore &weights,
                                                 const Config &config, unsigned layer,
                                                 Tensor hidden, Tensor attended,
                                                 bool split_fc1_output) {
  const auto prefix = config.prefix(layer);
  auto mixer = linear(g, weights, attended, prefix + "self_attn.out_proj");
  auto residual = g.binary(VSI_NN_OP_ADD, hidden, mixer);
  auto normalized = norm(g, weights, residual, prefix + "self_attn_layer_norm", config.epsilon);
  auto fc1 = linear(g, weights, normalized, prefix + "fc1", split_fc1_output);
  auto activation = g.unary(VSI_NN_OP_RELU, fc1, f16);
  auto mlp = linear(g, weights, activation, prefix + "fc2");
  auto final_residual = g.binary(VSI_NN_OP_ADD, normalized, mlp);
  auto output = norm(g, weights, final_residual, prefix + "final_layer_norm", config.epsilon);
  return {{"mixer", mixer},
          {"attention_residual", residual},
          {"attention_norm", normalized},
          {"fc1", fc1},
          {"activation", activation},
          {"mlp", mlp},
          {"ffn_residual", final_residual},
          {"output", output}};
}
} // namespace specferry::models::opt
