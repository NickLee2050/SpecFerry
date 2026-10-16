#pragma once

#include "np101/ops/graph_builder.hpp"
#include "np101/weights.hpp"

namespace specferry::np101::ops {
struct AttentionResult {
  Tensor attended;
  Tensor probabilities;
};

// SDK axes: Q [head_dim, query_groups, kv_heads], K/V [head_dim, capacity, kv_heads].
// FP16 storage, FP32 masked softmax, then FP16 probabilities and output.
// valid_length is scalar or one causal limit per query column, shared across heads.
// Flattened softmax columns are head * query_groups + query; attended has shape
// [head_dim, query_groups * kv_heads] in that same order.
AttentionResult attention_core(GraphBuilder &builder, Tensor query, Tensor keys, Tensor values,
                               Tensor valid_length, float scale);
Tensor swiglu(GraphBuilder &builder, Tensor input, const WeightStore &weights,
              const WeightRecord &gate, const WeightRecord &up, const WeightRecord &down,
              LinearBackend backend = LinearBackend::MatMul);
} // namespace specferry::np101::ops
