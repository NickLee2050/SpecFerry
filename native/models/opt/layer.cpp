#include "models/opt/layer.hpp"
#include "vsi_nn_pub.h"

#include <cmath>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;

Tensor linear(GraphBuilder &g, const WeightStore &weights, Tensor input, const std::string &name) {
  return g.linear(input, weights, weights.find(name + ".weight"), weights.find(name + ".bias"));
}

Tensor norm(GraphBuilder &g, const WeightStore &weights, Tensor input, const std::string &name,
            float epsilon) {
  const TensorSpec spec{f16, {input.spec.shape[0]}};
  auto scale = g.weight(weights, weights.find(name + ".weight"), spec);
  auto bias = g.weight(weights, weights.find(name + ".bias"), spec);
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
                                                 Tensor hidden, Tensor attended) {
  const auto prefix = config.prefix(layer);
  auto mixer = linear(g, weights, attended, prefix + "self_attn.out_proj");
  auto residual = g.binary(VSI_NN_OP_ADD, hidden, mixer);
  auto normalized = norm(g, weights, residual, prefix + "self_attn_layer_norm", config.epsilon);
  auto fc1 = linear(g, weights, normalized, prefix + "fc1");
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
