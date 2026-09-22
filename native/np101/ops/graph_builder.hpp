#pragma once

#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"
#include "vsi_nn_pub.h"

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <vector>

namespace specferry::np101::ops {
using Shape = std::vector<std::uint32_t>;

struct Tensor {
  vsi_nn_tensor_id_t id;
  TensorSpec spec;
};

// Adds nodes to one graph. Parameter arrays outlive the graph that borrows them.
// This is a construction helper, not an execution engine or model registry.
class GraphBuilder {
  std::deque<Shape> parameters_;

public:
  Graph graph;

  GraphBuilder(Context &context, unsigned tensors, unsigned nodes);
  Tensor tensor(TensorSpec spec);
  Tensor constant(TensorSpec spec, const std::vector<std::uint8_t> &bytes);
  Tensor floats(const std::vector<float> &values, Shape shape, DataType type);
  Tensor scalar(float value, DataType type = DataType::Float32);
  Tensor weight(const WeightStore &store, const WeightRecord &record, TensorSpec expected);
  Tensor share(GraphBuilder &owner, Tensor source);
  Tensor bind(TensorBinding source, TensorSpec expected);
  vsi_nn_node_t *node(vsi_nn_op_t op, std::initializer_list<Tensor> inputs, Tensor output);
  Tensor unary(vsi_nn_op_t op, Tensor input, DataType type);
  Tensor convert(Tensor input, DataType type);
  Tensor binary(vsi_nn_op_t op, Tensor left, Tensor right);
  Tensor reshape(Tensor input, Shape shape);
  Tensor slice(Tensor input, Shape start, Shape length);
  Tensor concat(Tensor left, Tensor right);
  Tensor reduce(Tensor input, bool mean, unsigned axis = 0);
  Tensor matmul(Tensor left, Tensor right, Shape output, bool transpose_left = false,
                bool transpose_right = false);
  Tensor project(Tensor input, const WeightStore &store, const WeightRecord &record);
  // Fused FP16 fully connected operation, including bias before output rounding.
  Tensor linear(Tensor input, const WeightStore &store, const WeightRecord &matrix,
                const WeightRecord &bias);
  Tensor layer_norm(Tensor input, Tensor scale, Tensor bias, float epsilon);
  // FP32 square/reduce/rsqrt: mean=true is RMS; false is L2 normalization.
  Tensor normalize(Tensor input, bool mean, float epsilon, unsigned axis = 0);
  Tensor rms_norm(Tensor input, Tensor scale, float epsilon, float scale_offset,
                  DataType output_type);
  void compile(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs);
};
} // namespace specferry::np101::ops
