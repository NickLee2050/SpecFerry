#include "np101/ops/graph_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace specferry::np101::ops {
namespace {
constexpr auto f16 = DataType::Float16;
constexpr auto f32 = DataType::Float32;
} // namespace

GraphBuilder::GraphBuilder(Context &context, unsigned tensors, unsigned nodes)
    : graph(context, tensors, nodes) {}

Tensor GraphBuilder::tensor(TensorSpec spec) { return {add_tensor(graph, spec), std::move(spec)}; }

Tensor GraphBuilder::constant(TensorSpec spec, const std::vector<std::uint8_t> &bytes) {
  return {add_tensor(graph, spec, true, bytes), std::move(spec)};
}

Tensor GraphBuilder::floats(const std::vector<float> &values, Shape shape, DataType type) {
  if (type != f16 && type != f32) {
    throw std::invalid_argument("floating constants require FP16 or FP32");
  }
  TensorSpec spec{type, std::move(shape)};
  if (values.size() != spec.elements()) {
    throw std::invalid_argument("constant size mismatch");
  }
  vsi_nn_dtype_t dtype{};
  dtype.vx_type = type == f16 ? VSI_NN_TYPE_FLOAT16 : VSI_NN_TYPE_FLOAT32;
  dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
  std::vector<std::uint8_t> bytes(spec.bytes());
  for (std::size_t index = 0; index < values.size(); ++index) {
    check(vsi_nn_Float32ToDtype(values[index], bytes.data() + index * dtype_bytes(type), &dtype),
          "encode constant");
  }
  return constant(spec, bytes);
}

Tensor GraphBuilder::scalar(float value, DataType type) { return floats({value}, {1}, type); }

Tensor GraphBuilder::weight(const WeightStore &store, const WeightRecord &record,
                            TensorSpec expected) {
  if (record.spec.type != expected.type || record.spec.shape != expected.shape) {
    throw std::invalid_argument("unexpected weight contract: " + record.name);
  }
  return constant(expected, store.read(record, 0, record.bytes));
}

Tensor GraphBuilder::share(GraphBuilder &owner, Tensor source) {
  return {retain_tensor(owner.graph, source.id, graph), source.spec};
}

Tensor GraphBuilder::bind(TensorBinding source, TensorSpec expected) {
  return {bind_tensor(source, graph, expected), std::move(expected)};
}

vsi_nn_node_t *GraphBuilder::node(vsi_nn_op_t op, std::initializer_list<Tensor> inputs,
                                  Tensor output) {
  auto *result = vsi_nn_AddNode(graph.get(), op, 0, 0, nullptr);
  if (!result || !result->input.tensors || !result->output.tensors ||
      result->input.num < inputs.size() || result->output.num < 1) {
    throw std::runtime_error("AddNode failed");
  }
  std::fill_n(result->input.tensors, result->input.num, VSI_NN_TENSOR_ID_NA);
  unsigned slot = 0;
  for (const auto &input : inputs) {
    result->input.tensors[slot++] = input.id;
  }
  result->output.tensors[0] = output.id;
  if (op == VSI_NN_OP_MULTIPLY) {
    result->nn_param.multiply.scale = 1;
  } else if (op == VSI_NN_OP_SWISH) {
    result->nn_param.swish.beta = 1;
    result->nn_param.swish.type = VSI_NN_SWISH;
  }
  return result;
}

Tensor GraphBuilder::unary(vsi_nn_op_t op, Tensor input, DataType type) {
  auto output = tensor({type, input.spec.shape});
  node(op, {input}, output);
  return output;
}

Tensor GraphBuilder::convert(Tensor input, DataType type) {
  return unary(VSI_NN_OP_DATACONVERT, input, type);
}

Tensor GraphBuilder::binary(vsi_nn_op_t op, Tensor left, Tensor right) {
  if (right.spec.shape.size() > left.spec.shape.size()) {
    throw std::invalid_argument("binary operand rank exceeds output rank");
  }
  for (std::size_t axis = 0; axis < right.spec.shape.size(); ++axis) {
    if (right.spec.shape[axis] != 1 && right.spec.shape[axis] != left.spec.shape[axis]) {
      throw std::invalid_argument("invalid binary broadcast");
    }
  }
  auto output = tensor(left.spec);
  node(op, {left, right}, output);
  return output;
}

Tensor GraphBuilder::reshape(Tensor input, Shape shape) {
  TensorSpec spec{input.spec.type, shape};
  if (spec.elements() != input.spec.elements()) {
    throw std::invalid_argument("reshape changes element count");
  }
  auto output = tensor(spec);
  parameters_.push_back(std::move(shape));
  auto *operation = node(VSI_NN_OP_RESHAPE, {input}, output);
  operation->nn_param.reshape.size = parameters_.back().data();
  operation->nn_param.reshape.dim_num = parameters_.back().size();
  return output;
}

Tensor GraphBuilder::slice(Tensor input, Shape start, Shape length) {
  if (start.size() != input.spec.shape.size() || length.size() != start.size()) {
    throw std::invalid_argument("slice rank mismatch");
  }
  for (std::size_t axis = 0; axis < start.size(); ++axis) {
    if (start[axis] > input.spec.shape[axis] ||
        length[axis] > input.spec.shape[axis] - start[axis]) {
      throw std::out_of_range("slice exceeds input");
    }
  }
  auto output = tensor({input.spec.type, length});
  auto *operation = node(VSI_NN_OP_SLICE, {input}, output);
  operation->nn_param.slice.dims = start.size();
  parameters_.push_back(std::move(start));
  operation->nn_param.slice.start = parameters_.back().data();
  parameters_.push_back(std::move(length));
  operation->nn_param.slice.length = parameters_.back().data();
  return output;
}

Tensor GraphBuilder::concat(Tensor left, Tensor right) {
  auto shape = left.spec.shape;
  if (left.spec.type != right.spec.type || shape.size() != right.spec.shape.size() ||
      !std::equal(shape.begin() + 1, shape.end(), right.spec.shape.begin() + 1) ||
      right.spec.shape[0] > std::numeric_limits<std::uint32_t>::max() - shape[0]) {
    throw std::invalid_argument("concat shape/type mismatch");
  }
  shape[0] += right.spec.shape[0];
  auto output = tensor({left.spec.type, shape});
  node(VSI_NN_OP_CONCAT, {left, right}, output)->nn_param.concat.axis = 0;
  return output;
}

Tensor GraphBuilder::reduce(Tensor input, bool mean, unsigned axis) {
  auto shape = input.spec.shape;
  if (axis >= shape.size()) {
    throw std::invalid_argument("reduction axis exceeds rank");
  }
  shape[axis] = 1;
  auto output = tensor({input.spec.type, shape});
  parameters_.push_back({axis});
  auto *operation = node(VSI_NN_OP_REDUCE, {input}, output);
  operation->nn_param.reduce.type = mean ? VSI_NN_REDUCE_MEAN : VSI_NN_REDUCE_SUM;
  operation->nn_param.reduce.axis = reinterpret_cast<const int32_t *>(parameters_.back().data());
  operation->nn_param.reduce.axis_num = 1;
  operation->nn_param.reduce.keep_dim = TRUE;
  return output;
}

Tensor GraphBuilder::matmul(Tensor left, Tensor right, Shape shape, bool transpose_left,
                            bool transpose_right) {
  if (left.spec.shape.size() < 2 || right.spec.shape.size() < 2 ||
      left.spec.type != right.spec.type ||
      left.spec.shape[transpose_left ? 1 : 0] != right.spec.shape[transpose_right ? 0 : 1]) {
    throw std::invalid_argument("matmul reduction dimensions disagree");
  }
  // SDK axis zero is contiguous: [K,M] x [N,K] produces [N,M].
  const auto rank = std::max(left.spec.shape.size(), right.spec.shape.size());
  Shape expected(rank, 1);
  expected[0] = right.spec.shape[transpose_right ? 1 : 0];
  expected[1] = left.spec.shape[transpose_left ? 0 : 1];
  for (std::size_t axis = 2; axis < rank; ++axis) {
    const auto a = axis < left.spec.shape.size() ? left.spec.shape[axis] : 1;
    const auto b = axis < right.spec.shape.size() ? right.spec.shape[axis] : 1;
    if (a != b && a != 1 && b != 1) {
      throw std::invalid_argument("matmul batch dimensions disagree");
    }
    expected[axis] = std::max(a, b);
  }
  if (shape != expected) {
    throw std::invalid_argument("matmul output shape disagrees with operands");
  }
  auto output = tensor({left.spec.type, std::move(shape)});
  auto *operation = node(VSI_NN_OP_MATRIXMUL, {left, right}, output);
  operation->nn_param.matrixmul.transpose[0] = transpose_left;
  operation->nn_param.matrixmul.transpose[1] = transpose_right;
  return output;
}

Tensor GraphBuilder::project(Tensor input, const WeightStore &store, const WeightRecord &record) {
  if (input.spec.shape.size() != 2 || input.spec.shape[1] != 1 || record.spec.shape.size() != 2 ||
      record.spec.shape[0] != input.spec.shape[0] || input.spec.type != f16 ||
      record.spec.type != f16) {
    throw std::invalid_argument("projection requires FP16 [K,1] and [K,N]");
  }
  const auto columns = input.spec.shape[0], rows = record.spec.shape[1];
  // Preserve the validated at-most-two-block projection strategy and bounded IO.
  constexpr unsigned block_rows = 4096;
  if (rows > 2 * block_rows) {
    throw std::invalid_argument("projection exceeds validated two-block width");
  }
  if (rows <= block_rows) {
    return matmul(input, weight(store, record, record.spec), {rows, 1}, false, true);
  }
  const auto first_bytes = std::size_t(block_rows) * columns * dtype_bytes(f16);
  auto first = constant({f16, {columns, block_rows}}, store.read(record, 0, first_bytes));
  auto second = constant({f16, {columns, rows - block_rows}},
                         store.read(record, first_bytes, record.bytes - first_bytes));
  auto first_output = matmul(input, first, {block_rows, 1}, false, true);
  auto second_output = matmul(input, second, {rows - block_rows, 1}, false, true);
  return concat(first_output, second_output);
}

Tensor GraphBuilder::normalize(Tensor input, bool mean, float epsilon, unsigned axis) {
  if (input.spec.type != f32 || !std::isfinite(epsilon) || epsilon <= 0) {
    throw std::invalid_argument("normalization requires FP32 and positive finite epsilon");
  }
  auto squares = binary(VSI_NN_OP_MULTIPLY, input, input);
  auto reduced = reduce(squares, mean, axis);
  auto inverse = unary(VSI_NN_OP_RSQRT, binary(VSI_NN_OP_ADD, reduced, scalar(epsilon)), f32);
  return binary(VSI_NN_OP_MULTIPLY, input, inverse);
}

Tensor GraphBuilder::linear(Tensor input, const WeightStore &store, const WeightRecord &matrix,
                            const WeightRecord &bias) {
  if (input.spec.type != f16 || input.spec.shape.size() != 2 || input.spec.shape[1] != 1 ||
      matrix.spec.type != f16 || matrix.spec.shape.size() != 2 ||
      matrix.spec.shape[0] != input.spec.shape[0] || bias.spec.type != f16 ||
      bias.spec.shape != Shape{matrix.spec.shape[1]}) {
    throw std::invalid_argument("linear requires FP16 input [K,1], weight [K,N], bias [N]");
  }
  // The installed SDK crashes while verifying even an isolated biased FCL.
  // Use documented MatMul/Add; FP16 projection rounds before the bias addition.
  // Keep this boundary visible in the independent model-level comparisons.
  auto projected = project(input, store, matrix);
  return binary(VSI_NN_OP_ADD, projected, weight(store, bias, bias.spec));
}

Tensor GraphBuilder::layer_norm(Tensor input, Tensor scale, Tensor bias, float epsilon) {
  if (input.spec.type != f16 || input.spec.shape.size() != 2 || input.spec.shape[1] != 1 ||
      scale.spec.type != f16 || bias.spec.type != f16 ||
      scale.spec.shape != Shape{input.spec.shape[0]} || bias.spec.shape != scale.spec.shape) {
    throw std::invalid_argument("layer norm requires FP16 [width,1] and affine vectors");
  }
  auto values = convert(input, f32);
  auto centered = binary(VSI_NN_OP_SUBTRACT, values, reduce(values, true));
  auto normalized = normalize(centered, true, epsilon);
  auto multiplier = reshape(convert(scale, f32), input.spec.shape);
  auto offset = reshape(convert(bias, f32), input.spec.shape);
  return convert(binary(VSI_NN_OP_ADD, binary(VSI_NN_OP_MULTIPLY, normalized, multiplier), offset),
                 f16);
}

Tensor GraphBuilder::rms_norm(Tensor input, Tensor scale, float epsilon, float scale_offset,
                              DataType output_type) {
  auto normalized = normalize(convert(input, f32), true, epsilon);
  auto multiplier = convert(scale, f32);
  multiplier = binary(VSI_NN_OP_ADD, multiplier, scalar(scale_offset));
  Shape shape(input.spec.shape.size(), 1);
  shape[0] = input.spec.shape[0];
  return convert(binary(VSI_NN_OP_MULTIPLY, normalized, reshape(multiplier, shape)), output_type);
}

void GraphBuilder::compile(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs) {
  std::vector<vsi_nn_tensor_id_t> input_ids, output_ids;
  for (const auto &input : inputs) {
    input_ids.push_back(input.id);
  }
  for (const auto &output : outputs) {
    output_ids.push_back(output.id);
  }
  if (!vsi_nn_SetGraphInputs(graph.get(), input_ids.data(), input_ids.size()) ||
      !vsi_nn_SetGraphOutputs(graph.get(), output_ids.data(), output_ids.size())) {
    throw std::runtime_error("graph IO declaration failed");
  }
  check(vsi_nn_SetupGraph(graph.get(), FALSE), "SetupGraph");
  check(vsi_nn_VerifyGraph(graph.get()), "VerifyGraph");
}
} // namespace specferry::np101::ops
