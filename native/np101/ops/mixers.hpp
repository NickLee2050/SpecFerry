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
AttentionResult attention_core(GraphBuilder &builder, Tensor query, Tensor keys, Tensor values,
                               Tensor valid_length, float scale);
// Q/K/V are FP16 [dimension,1,heads]; state is FP32 [value_dim,key_dim,heads].
// The caller supplies independent input/output banks and the L2 epsilon.
Tensor delta_recurrence(GraphBuilder &builder, Tensor query, Tensor key, Tensor value, Tensor decay,
                        Tensor beta, Tensor state, Tensor updated_state, float epsilon);
Tensor short_convolution(GraphBuilder &builder, Tensor history, Tensor packed, Tensor kernel,
                         Tensor updated_history);
Tensor swiglu(GraphBuilder &builder, Tensor input, const WeightStore &weights,
              const WeightRecord &gate, const WeightRecord &up, const WeightRecord &down);
} // namespace specferry::np101::ops
