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
      (valid_length.spec.elements() != 1 &&
       (keys.spec.shape[2] != 1 || valid_length.spec.elements() != query.spec.shape[1]))) {
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
  auto valid = g.tensor({DataType::Bool8, {capacity, per_query ? groups : 1}});
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

Tensor delta_recurrence(GraphBuilder &g, Tensor query, Tensor key, Tensor value, Tensor decay,
                        Tensor beta, Tensor state, Tensor updated_state, float epsilon) {
  if (query.spec.shape != key.spec.shape || query.spec.shape.size() != 3 ||
      value.spec.shape.size() != 3 || query.spec.shape[1] != 1 || value.spec.shape[1] != 1 ||
      value.spec.shape[2] != query.spec.shape[2] || state.spec.type != f32 ||
      query.spec.type != f16 || key.spec.type != f16 || value.spec.type != f16 ||
      decay.spec.type != f32 || beta.spec.type != f32 || updated_state.spec.type != f32 ||
      state.spec.shape != updated_state.spec.shape ||
      state.spec.shape != Shape{value.spec.shape[0], key.spec.shape[0], key.spec.shape[2]}) {
    throw std::invalid_argument("DeltaNet recurrence contract mismatch");
  }
  const auto key_dim = key.spec.shape[0], value_dim = value.spec.shape[0],
             heads = key.spec.shape[2];
  // Match the official L2 normalization and its FP16 boundary before recurrence.
  query = g.convert(g.convert(g.normalize(g.convert(query, f32), false, epsilon), f16), f32);
  key = g.convert(g.convert(g.normalize(g.convert(key, f32), false, epsilon), f16), f32);
  query = g.binary(VSI_NN_OP_MULTIPLY, query, g.scalar(1.0f / std::sqrt(float(key_dim))));
  value = g.convert(value, f32);
  auto decayed = g.binary(VSI_NN_OP_MULTIPLY, state, decay);
  auto prediction = g.matmul(key, decayed, {value_dim, 1, heads});
  auto error = g.binary(VSI_NN_OP_SUBTRACT, value, prediction);
  auto correction = g.binary(VSI_NN_OP_MULTIPLY, error, beta);
  auto update = g.matmul(key, correction, {value_dim, key_dim, heads}, true);
  g.node(VSI_NN_OP_ADD, {decayed, update}, updated_state);
  // Preserve small FP32 core values until the caller's RMS reduction.
  return g.matmul(query, updated_state, {value_dim, 1, heads});
}

Tensor short_convolution(GraphBuilder &g, Tensor history, Tensor packed, Tensor kernel,
                         Tensor updated_history) {
  if (history.spec.shape.size() != 2 || history.spec.shape[0] < 2 || history.spec.type != f16 ||
      packed.spec.type != f16 || kernel.spec.type != f16 || updated_history.spec.type != f16 ||
      history.spec.shape != updated_history.spec.shape ||
      packed.spec.elements() != history.spec.shape[1] ||
      kernel.spec.elements() != history.spec.elements()) {
    throw std::invalid_argument("short convolution contract mismatch");
  }
  auto width = history.spec.shape[0], channels = history.spec.shape[1];
  auto retained = g.slice(history, {1, 0}, {width - 1, channels});
  auto current = g.reshape(packed, {1, channels});
  g.node(VSI_NN_OP_CONCAT, {retained, current}, updated_history)->nn_param.concat.axis = 0;
  kernel = g.reshape(g.convert(kernel, f32), {width, channels});
  auto products = g.binary(VSI_NN_OP_MULTIPLY, g.convert(updated_history, f32), kernel);
  auto result = g.unary(VSI_NN_OP_SWISH, g.convert(g.reduce(products, false), f16), f16);
  return g.reshape(result, {channels, 1});
}

Tensor swiglu(GraphBuilder &g, Tensor input, const WeightStore &weights, const WeightRecord &gate,
              const WeightRecord &up, const WeightRecord &down) {
  auto gated = g.project(input, weights, gate);
  auto values = g.project(input, weights, up);
  auto activated = g.unary(VSI_NN_OP_SWISH, gated, DataType::Float16);
  return g.project(g.binary(VSI_NN_OP_MULTIPLY, activated, values), weights, down);
}
} // namespace specferry::np101::ops
