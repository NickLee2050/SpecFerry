#pragma once

#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"
#include "vsi_nn_pub.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <string>
#include <vector>

namespace specferry::np101 {
class WeightBank;
}

namespace specferry::np101::ops {
using Shape = std::vector<std::uint32_t>;

struct Tensor {
  vsi_nn_tensor_id_t id;
  TensorSpec spec;
};

struct CompiledGraphSpec {
  std::string path;
  // Caller-validated compiled ports, in SDK order (contiguous dimension first).
  std::vector<TensorSpec> inputs, outputs;
  // Known embedded tensor payload, excluding unknown SDK copies and workspaces.
  std::size_t constant_bytes = 0;
  std::array<unsigned, 3> graph_version{};
};

// Adds nodes to one graph. Parameter arrays outlive the graph that borrows them.
// This is a construction helper, not an execution engine or model registry.
class GraphBuilder {
  std::deque<Shape> parameters_;
  std::deque<std::string> compiled_paths_;
  std::array<unsigned, 3> compiled_version_{};
  WeightBank *weights_;
  std::vector<vsi_nn_tensor_id_t> mutable_weight_inputs_;
  std::vector<vsi_nn_tensor_id_t> input_ids_, output_ids_;

public:
  Graph graph;

  // Use the supplied shared bank, or local const weights when no bank is supplied.
  GraphBuilder(Context &context, unsigned tensors, unsigned nodes, WeightBank *weights = nullptr);
  Tensor tensor(TensorSpec spec);
  Tensor constant(TensorSpec spec, const std::vector<std::uint8_t> &bytes);
  Tensor floats(const std::vector<float> &values, Shape shape, DataType type);
  Tensor scalar(float value, DataType type = DataType::Float32);
  Tensor weight(const WeightStore &store, const WeightRecord &record, TensorSpec expected);
  // An explicit FP32 target widens FP16 checkpoint data once during initialization.
  // Offsets refer to source bytes; shared backing is keyed by target dtype and shape.
  Tensor weight_chunk(const WeightStore &store, const WeightRecord &record, TensorSpec spec,
                      std::size_t offset);
  Tensor share(GraphBuilder &owner, Tensor source);
  Tensor bind(TensorBinding source, TensorSpec expected);
  vsi_nn_node_t *node(vsi_nn_op_t op, std::initializer_list<Tensor> inputs, Tensor output);
  Tensor unary(vsi_nn_op_t op, Tensor input, DataType type);
  Tensor convert(Tensor input, DataType type);
  Tensor binary(vsi_nn_op_t op, Tensor left, Tensor right);
  Tensor reshape(Tensor input, Shape shape);
  Tensor permute(Tensor input, Shape order);
  Tensor slice(Tensor input, Shape start, Shape length);
  Tensor concat(Tensor left, Tensor right, unsigned axis = 0);
  Tensor reduce(Tensor input, bool mean, unsigned axis = 0);
  Tensor matmul(Tensor left, Tensor right, Shape output, bool transpose_left = false,
                bool transpose_right = false);
  Tensor gather(Tensor input, Tensor indices, unsigned axis);
  Tensor argmax(Tensor input, unsigned axis);
  Tensor integer(std::int32_t value);
  Tensor project(Tensor input, const WeightStore &store, const WeightRecord &record);
  // FP16 projection followed by bias addition, with an FP16 intermediate boundary.
  Tensor linear(Tensor input, const WeightStore &store, const WeightRecord &matrix,
                const WeightRecord &bias, unsigned output_parts = 1);
  // Creates fixed NBG ports; mixing NBG with ordinary nodes needs workload validation.
  // The builder owns the path until graph destruction; the graph owns payload charges.
  std::vector<Tensor> compiled(const CompiledGraphSpec &spec, const std::vector<Tensor> &inputs);
  Tensor layer_norm(Tensor input, Tensor scale, Tensor bias, float epsilon);
  // FP32 square/reduce/rsqrt: mean=true is RMS; false is L2 normalization.
  Tensor normalize(Tensor input, bool mean, float epsilon, unsigned axis = 0);
  Tensor rms_norm(Tensor input, Tensor scale, float epsilon, float scale_offset,
                  DataType output_type);
  void compile(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs);
};
} // namespace specferry::np101::ops
