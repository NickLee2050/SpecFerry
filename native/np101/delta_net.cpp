#include "np101/delta_net.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>

namespace specferry::np101 {
namespace {
using Id = vsi_nn_tensor_id_t;
using Shape = std::vector<std::uint32_t>;
constexpr auto f16 = DataType::Float16;
constexpr auto f32 = DataType::Float32;
const std::string prefix = "model.layers.0.linear_attn.";

struct Tensor {
  Id id;
  TensorSpec spec;
};

// One narrowly scoped exception to ref_op_api_guide.md: retain a fixed OpenVX
// tensor in a second SDK wrapper. AddTensor only creates graph-owned tensors;
// the installed SDK does not export AttachTensorToGraph. Autoregressive state
// must cross executions without host copies. No handle is swapped or rebound
// after graph compilation. Each wrapper owns exactly one retained reference.
Tensor share_state(Graph &source, Tensor tensor, Graph &destination) {
  auto *original = vsi_nn_GetTensor(source.get(), tensor.id);
  if (!original || !original->t || original->attr.vtl || original->attr.is_const ||
      original->attr.is_created_from_handle) {
    throw std::runtime_error("state sharing requires a materialized ordinary mutable tensor");
  }
  Tensor shared{add_tensor(destination, tensor.spec), tensor.spec};
  auto *wrapper = vsi_nn_GetTensor(destination.get(), shared.id);
  check(vxRetainReference(reinterpret_cast<vx_reference>(original->t)), "retain shared state");
  if (wrapper->t) {
    auto status = vxReleaseTensor(&wrapper->t);
    if (status != VX_SUCCESS) {
      auto retained = original->t;
      vxReleaseTensor(&retained);
      check(status, "release unused state tensor");
    }
  }
  wrapper->t = original->t;
  return shared;
}

class StepGraph {
public:
  // Parameter arrays must outlive graph destruction, hence declared first.
  std::deque<Shape> parameters;
  Graph graph;
  Tensor input;
  Tensor recurrent_input;
  Tensor convolution_input;
  Tensor recurrent_output;
  Tensor convolution_output;
  std::map<std::string, Tensor> outputs;

  StepGraph(Context &context, const WeightStore &weights, StepGraph *previous = nullptr)
      : graph(context, 160, 128), input(make({f16, {1024, 1}})) {
    if (previous) {
      recurrent_input = share_state(previous->graph, previous->recurrent_output, graph);
      recurrent_output = share_state(previous->graph, previous->recurrent_input, graph);
      convolution_input = share_state(previous->graph, previous->convolution_output, graph);
      convolution_output = share_state(previous->graph, previous->convolution_input, graph);
    } else {
      recurrent_input = make(DeltaNet::recurrent_spec());
      recurrent_output = make(DeltaNet::recurrent_spec());
      convolution_input = make(DeltaNet::convolution_spec());
      convolution_output = make(DeltaNet::convolution_spec());
    }
    build(weights);
    std::array<Id, 3> inputs{input.id, recurrent_input.id, convolution_input.id};
    std::vector<Id> output_ids;
    for (const auto &entry : outputs) {
      output_ids.push_back(entry.second.id);
    }
    if (!vsi_nn_SetGraphInputs(graph.get(), inputs.data(), inputs.size()) ||
        !vsi_nn_SetGraphOutputs(graph.get(), output_ids.data(), output_ids.size())) {
      throw std::runtime_error("DeltaNet graph IO declaration failed");
    }
    check(vsi_nn_SetupGraph(graph.get(), FALSE), "DeltaNet SetupGraph");
    check(vsi_nn_VerifyGraph(graph.get()), "DeltaNet VerifyGraph");
  }

private:
  Tensor make(TensorSpec spec) { return {add_tensor(graph, spec), std::move(spec)}; }

  Tensor constant(TensorSpec spec, const std::vector<std::uint8_t> &bytes) {
    return {add_tensor(graph, spec, true, bytes), std::move(spec)};
  }

  Tensor scalar(float value) {
    std::vector<std::uint8_t> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return constant({f32, {1}}, bytes);
  }

  Tensor weight(const WeightStore &weights, const std::string &name, TensorSpec expected) {
    const auto &record = weights.find(prefix + name);
    if (record.spec.type != expected.type || record.spec.shape != expected.shape) {
      throw std::invalid_argument("unexpected DeltaNet weight contract: " + name);
    }
    return constant(expected, weights.read(record, 0, record.bytes));
  }

  vsi_nn_node_t *node(vsi_nn_op_t op, std::initializer_list<Tensor> inputs, Tensor output) {
    auto *result = vsi_nn_AddNode(graph.get(), op, 0, 0, nullptr);
    if (!result || result->input.num < inputs.size() || result->output.num < 1) {
      throw std::runtime_error("DeltaNet AddNode failed");
    }
    std::fill_n(result->input.tensors, result->input.num, VSI_NN_TENSOR_ID_NA);
    unsigned slot = 0;
    for (auto input_tensor : inputs) {
      result->input.tensors[slot++] = input_tensor.id;
    }
    result->output.tensors[0] = output.id;
    if (op == VSI_NN_OP_MULTIPLY) {
      result->nn_param.multiply.scale = 1.0f;
    } else if (op == VSI_NN_OP_SWISH) {
      result->nn_param.swish.beta = 1.0f;
      result->nn_param.swish.type = VSI_NN_SWISH;
    }
    return result;
  }

  Tensor unary(vsi_nn_op_t op, Tensor input_tensor, DataType type) {
    auto output = make({type, input_tensor.spec.shape});
    node(op, {input_tensor}, output);
    return output;
  }

  Tensor convert(Tensor input_tensor, DataType type) {
    return unary(VSI_NN_OP_DATACONVERT, input_tensor, type);
  }

  Tensor binary(vsi_nn_op_t op, Tensor left, Tensor right) {
    auto output = make(left.spec);
    node(op, {left, right}, output);
    return output;
  }

  Tensor reshape(Tensor input_tensor, Shape shape) {
    auto output = make({input_tensor.spec.type, shape});
    parameters.push_back(std::move(shape));
    auto *operation = node(VSI_NN_OP_RESHAPE, {input_tensor}, output);
    operation->nn_param.reshape.size = parameters.back().data();
    operation->nn_param.reshape.dim_num = parameters.back().size();
    return output;
  }

  Tensor slice(Tensor input_tensor, Shape start, Shape length) {
    auto output = make({input_tensor.spec.type, length});
    auto *operation = node(VSI_NN_OP_SLICE, {input_tensor}, output);
    operation->nn_param.slice.dims = start.size();
    parameters.push_back(std::move(start));
    operation->nn_param.slice.start = parameters.back().data();
    parameters.push_back(std::move(length));
    operation->nn_param.slice.length = parameters.back().data();
    return output;
  }

  Tensor reduce(Tensor input_tensor, bool mean) {
    auto shape = input_tensor.spec.shape;
    shape[0] = 1;
    auto output = make({input_tensor.spec.type, shape});
    parameters.push_back({0});
    auto *operation = node(VSI_NN_OP_REDUCE, {input_tensor}, output);
    operation->nn_param.reduce.type = mean ? VSI_NN_REDUCE_MEAN : VSI_NN_REDUCE_SUM;
    operation->nn_param.reduce.axis = reinterpret_cast<const int32_t *>(parameters.back().data());
    operation->nn_param.reduce.axis_num = 1;
    operation->nn_param.reduce.keep_dim = TRUE;
    return output;
  }

  Tensor matmul(Tensor left, Tensor right, Shape shape, bool transpose_left = false,
                bool transpose_right = false) {
    auto output = make({left.spec.type, std::move(shape)});
    auto *operation = node(VSI_NN_OP_MATRIXMUL, {left, right}, output);
    operation->nn_param.matrixmul.transpose[0] = transpose_left;
    operation->nn_param.matrixmul.transpose[1] = transpose_right;
    return output;
  }

  Tensor project(const WeightStore &weights, Tensor value, const std::string &name,
                 unsigned width) {
    if (width > 4096) {
      const auto &record = weights.find(prefix + name + ".weight");
      const unsigned columns = value.spec.shape[0];
      if (record.spec.type != f16 || record.spec.shape != Shape{columns, width} || width > 8192) {
        throw std::invalid_argument("unexpected blocked projection contract: " + name);
      }
      const std::size_t first_bytes = std::size_t(4096) * columns * 2;
      auto first = constant({f16, {columns, 4096}}, weights.read(record, 0, first_bytes));
      auto second = constant({f16, {columns, width - 4096}},
                             weights.read(record, first_bytes, record.bytes - first_bytes));
      auto left = matmul(value, first, {4096, 1}, false, true);
      auto right = matmul(value, second, {width - 4096, 1}, false, true);
      auto result = make({f16, {width, 1}});
      node(VSI_NN_OP_CONCAT, {left, right}, result)->nn_param.concat.axis = 0;
      return result;
    }
    auto matrix = weight(weights, name + ".weight", {f16, {value.spec.shape[0], width}});
    return matmul(value, matrix, {width, 1}, false, true);
  }

  Tensor normalize(Tensor value, bool mean) {
    auto squares = binary(VSI_NN_OP_MULTIPLY, value, value);
    auto sum = reduce(squares, mean);
    auto shifted = binary(VSI_NN_OP_ADD, sum, scalar(1e-6f));
    return binary(VSI_NN_OP_MULTIPLY, value, unary(VSI_NN_OP_RSQRT, shifted, f32));
  }

  Tensor convolve(const WeightStore &weights, Tensor packed) {
    auto retained = slice(convolution_input, {1, 0}, {3, 6144});
    auto current = reshape(packed, {1, 6144});
    node(VSI_NN_OP_CONCAT, {retained, current}, convolution_output)->nn_param.concat.axis = 0;
    auto kernel = weight(weights, "conv1d.weight", {f16, {4, 1, 6144}});
    kernel = reshape(convert(kernel, f32), {4, 6144});
    // Accumulate the four-tap depthwise convolution in FP32, round once to FP16
    // before SiLU, matching the CPU reference's convolution precision boundary.
    auto products = binary(VSI_NN_OP_MULTIPLY, convert(convolution_output, f32), kernel);
    auto result = unary(VSI_NN_OP_SWISH, convert(reduce(products, false), f16), f16);
    return reshape(result, {6144, 1});
  }

  Tensor recurrence(Tensor convolved, Tensor decay, Tensor beta) {
    auto query = reshape(slice(convolved, {0, 0}, {2048, 1}), {128, 1, 16});
    auto key = reshape(slice(convolved, {2048, 0}, {2048, 1}), {128, 1, 16});
    auto value = reshape(slice(convolved, {4096, 0}, {2048, 1}), {128, 1, 16});
    // Preserve the reference's FP32 normalization followed by an FP16 boundary.
    query = convert(convert(normalize(convert(query, f32), false), f16), f32);
    key = convert(convert(normalize(convert(key, f32), false), f16), f32);
    query = binary(VSI_NN_OP_MULTIPLY, query, scalar(1.0f / std::sqrt(128.0f)));
    value = convert(value, f32);
    auto decayed = binary(VSI_NN_OP_MULTIPLY, recurrent_input, decay);
    auto prediction = matmul(key, decayed, {128, 1, 16});
    auto error = binary(VSI_NN_OP_SUBTRACT, value, prediction);
    auto correction = binary(VSI_NN_OP_MULTIPLY, error, beta);
    auto update = matmul(key, correction, {128, 128, 16}, true);
    node(VSI_NN_OP_ADD, {decayed, update}, recurrent_output);
    return convert(matmul(query, recurrent_output, {128, 1, 16}), f16);
  }

  void build(const WeightStore &weights) {
    auto packed = project(weights, input, "in_proj_qkv", 6144);
    auto gate = project(weights, input, "in_proj_z", 2048);
    auto a = project(weights, input, "in_proj_a", 16);
    auto b = project(weights, input, "in_proj_b", 16);
    auto convolved = convolve(weights, packed);

    auto bias = reshape(convert(weight(weights, "dt_bias", {f16, {16}}), f32), {16, 1});
    auto time = unary(VSI_NN_OP_SOFTRELU, binary(VSI_NN_OP_ADD, convert(a, f32), bias), f32);
    auto rate = unary(VSI_NN_OP_EXP, weight(weights, "A_log", {f32, {16}}), f32);
    rate = reshape(binary(VSI_NN_OP_MULTIPLY, rate, scalar(-1.0f)), {16, 1});
    auto decay = unary(VSI_NN_OP_EXP, binary(VSI_NN_OP_MULTIPLY, time, rate), f32);
    auto beta = convert(unary(VSI_NN_OP_SIGMOID, b, f16), f32);
    auto core = recurrence(convolved, reshape(decay, {1, 1, 16}), reshape(beta, {1, 1, 16}));

    auto normalized = normalize(convert(reshape(core, {128, 16}), f32), true);
    normalized = convert(convert(normalized, f16), f32);
    auto scale = reshape(weight(weights, "norm.weight", {f32, {128}}), {128, 1});
    auto scaled = binary(VSI_NN_OP_MULTIPLY, normalized, scale);
    auto activated_gate = unary(VSI_NN_OP_SWISH, convert(reshape(gate, {128, 16}), f32), f32);
    auto gated = convert(binary(VSI_NN_OP_MULTIPLY, scaled, activated_gate), f16);
    auto result = project(weights, reshape(gated, {2048, 1}), "out_proj", 1024);
    outputs = {{"output", result},
               {"recurrent", recurrent_output},
               {"convolution", convolution_output},
               {"qkv", packed},
               {"convolved", convolved},
               {"decay", decay},
               {"beta", beta},
               {"core", core}};
  }
};
} // namespace

struct DeltaNet::Impl {
  std::unique_ptr<StepGraph> forward;
  std::unique_ptr<StepGraph> backward;
  std::size_t completed_steps = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights) {
    forward = std::make_unique<StepGraph>(context, weights);
    backward = std::make_unique<StepGraph>(context, weights, forward.get());
  }
};

DeltaNet::DeltaNet(Context &context, const WeightStore &weights)
    : impl_(std::make_unique<Impl>(context, weights)) {
  reset();
}

DeltaNet::~DeltaNet() = default;

TensorSpec DeltaNet::recurrent_spec() { return {f32, {128, 128, 16}}; }

TensorSpec DeltaNet::convolution_spec() { return {f16, {4, 6144}}; }

TensorSpec DeltaNet::output_spec(const std::string &name) {
  const std::map<std::string, TensorSpec> specs{{"output", {f16, {1024, 1}}},
                                                {"recurrent", recurrent_spec()},
                                                {"convolution", convolution_spec()},
                                                {"qkv", {f16, {6144, 1}}},
                                                {"convolved", {f16, {6144, 1}}},
                                                {"decay", {f32, {16, 1}}},
                                                {"beta", {f32, {16, 1}}},
                                                {"core", {f16, {128, 1, 16}}}};
  return specs.at(name);
}

void DeltaNet::reset(const std::vector<std::uint8_t> &recurrent,
                     const std::vector<std::uint8_t> &convolution) {
  if (!impl_) {
    throw std::logic_error("reset on closed DeltaNet");
  }
  if ((!recurrent.empty() && recurrent.size() != recurrent_spec().bytes()) ||
      (!convolution.empty() && convolution.size() != convolution_spec().bytes())) {
    throw std::invalid_argument("DeltaNet initial state size mismatch");
  }
  impl_->failed = true;
  auto &step = *impl_->forward;
  const auto state =
      recurrent.empty() ? std::vector<std::uint8_t>(recurrent_spec().bytes()) : recurrent;
  const auto history =
      convolution.empty() ? std::vector<std::uint8_t>(convolution_spec().bytes()) : convolution;
  for (auto tensor : {step.recurrent_input, step.recurrent_output}) {
    upload_tensor(step.graph, tensor.id, state);
  }
  for (auto tensor : {step.convolution_input, step.convolution_output}) {
    upload_tensor(step.graph, tensor.id, history);
  }
  impl_->completed_steps = 0;
  impl_->failed = false;
}

void DeltaNet::step(const std::vector<std::uint8_t> &hidden_fp16) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("DeltaNet must be initialized/reset before stepping");
  }
  if (hidden_fp16.size() != hidden_size * sizeof(std::uint16_t)) {
    throw std::invalid_argument("DeltaNet expects one FP16 hidden vector");
  }
  auto &step = impl_->completed_steps % 2 == 0 ? *impl_->forward : *impl_->backward;
  impl_->failed = true;
  upload_tensor(step.graph, step.input.id, hidden_fp16);
  check(vsi_nn_RunGraph(step.graph.get()), "DeltaNet RunGraph");
  ++impl_->completed_steps;
  impl_->failed = false;
}

std::vector<std::uint8_t> DeltaNet::read(const std::string &name) {
  if (!impl_ || impl_->failed || impl_->completed_steps == 0) {
    throw std::logic_error("DeltaNet has no completed output");
  }
  auto &last = impl_->completed_steps % 2 == 1 ? *impl_->forward : *impl_->backward;
  return read_tensor(last.graph, last.outputs.at(name).id);
}

std::size_t DeltaNet::steps() const { return impl_ ? impl_->completed_steps : 0; }

void DeltaNet::close() {
  if (impl_) {
    impl_->backward->graph.close();
    impl_->forward->graph.close();
    impl_.reset();
  }
}
} // namespace specferry::np101
