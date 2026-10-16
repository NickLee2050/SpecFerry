#include "np101/ops/mixers.hpp"
#include "vsi_nn_pub.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace specferry::np101::ops {
namespace {
constexpr auto f16 = DataType::Float16;
constexpr auto f32 = DataType::Float32;
} // namespace

AttentionResult attention_core(GraphBuilder &g, Tensor query, Tensor keys, Tensor values,
                               Tensor valid_length, float scale) {
  if (query.spec.shape.size() != 3 || keys.spec.shape.size() != 3 ||
      keys.spec.shape != values.spec.shape || query.spec.shape[0] != keys.spec.shape[0] ||
      query.spec.shape[2] != keys.spec.shape[2] || !std::isfinite(scale) || scale <= 0 ||
      query.spec.type != f16 || keys.spec.type != f16 || values.spec.type != f16 ||
      valid_length.spec.type != DataType::Int32 ||
      (valid_length.spec.elements() != 1 && valid_length.spec.elements() != query.spec.shape[1])) {
    throw std::invalid_argument("attention Q/K/V contract mismatch");
  }
  const auto dimension = keys.spec.shape[0], capacity = keys.spec.shape[1];
  const auto groups = query.spec.shape[1], kv_heads = keys.spec.shape[2];
  auto scores = g.matmul(query, keys, {capacity, groups, kv_heads}, false, true);
  scores = g.convert(g.binary(VSI_NN_OP_MULTIPLY, scores, g.scalar(scale, f16)), f32);
  // The installed SDK's rank-three softmax did not honor the token axis.
  scores = g.reshape(scores, {capacity, groups * kv_heads});
  std::vector<std::uint8_t> bytes(capacity * sizeof(std::int32_t));
  for (unsigned position = 0; position < capacity; ++position) {
    auto value = static_cast<std::int32_t>(position);
    std::memcpy(bytes.data() + position * sizeof(value), &value, sizeof(value));
  }
  auto positions = g.constant({DataType::Int32, {capacity, 1}}, bytes);
  const bool per_query = valid_length.spec.elements() != 1;
  auto limits = per_query ? g.reshape(valid_length, {1, groups}) : valid_length;
  if (per_query) {
    // A complete query sequence is repeated for each head. Doubling keeps the
    // number of concat nodes logarithmic without relying on rank-three masking
    // or a separate attention graph for every head.
    unsigned copied_heads = 1;
    while (copied_heads < kv_heads) {
      const auto remaining = kv_heads - copied_heads;
      const auto added_heads = remaining < copied_heads ? remaining : copied_heads;
      auto repeated =
          added_heads == copied_heads ? limits : g.slice(limits, {0, 0}, {1, groups * added_heads});
      limits = g.concat(limits, repeated, 1);
      copied_heads += added_heads;
    }
  }
  auto valid = g.tensor({DataType::Bool8, {capacity, per_query ? groups * kv_heads : 1}});
  g.node(VSI_NN_OP_RELATIONAL_OPS, {positions, limits}, valid)->nn_param.relational_ops.op =
      VSI_NN_RELATIONAL_OPS_LESS;
  auto masked = g.tensor(scores.spec);
  g.node(VSI_NN_OP_SELECT, {valid, scores, g.scalar(-1e9f)}, masked);
  auto probability_fp32 = g.tensor(masked.spec);
  auto *softmax = g.node(VSI_NN_OP_SOFTMAX, {masked}, probability_fp32);
  softmax->nn_param.softmax.axis = 0;
  softmax->nn_param.softmax.beta = 1;
  auto probabilities = g.reshape(g.convert(probability_fp32, f16), {capacity, groups, kv_heads});
  auto attended = g.reshape(g.matmul(probabilities, values, {dimension, groups, kv_heads}),
                            {dimension, groups * kv_heads});
  return {attended, probabilities};
}

Tensor swiglu(GraphBuilder &g, Tensor input, const WeightStore &weights, const WeightRecord &gate,
              const WeightRecord &up, const WeightRecord &down, LinearBackend backend) {
  auto gated = g.project(input, weights, gate, backend);
  auto values = g.project(input, weights, up, backend);
  auto activated = g.unary(VSI_NN_OP_SWISH, gated, DataType::Float16);
  return g.project(g.binary(VSI_NN_OP_MULTIPLY, activated, values), weights, down, backend);
}
} // namespace specferry::np101::ops
