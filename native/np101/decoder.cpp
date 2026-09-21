#include "np101/attention.hpp"
#include "np101/decoder.hpp"
#include "np101/delta_net.hpp"
#include "np101/kv_cache.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <utility>

namespace specferry::np101 {
namespace {
using Clock = std::chrono::steady_clock;
using Shape = std::vector<std::uint32_t>;
constexpr auto f16 = DataType::Float16;
constexpr auto f32 = DataType::Float32;
const TensorSpec hidden_spec{f16, {1024, 1}};

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// Only the two outer decoder graphs use these small construction helpers.
// Parameters remain alive until graph destruction; all arithmetic is in the guide.
class DecoderGraph {
public:
  std::deque<Shape> parameters;
  Graph graph;
  std::map<std::string, vsi_nn_tensor_id_t> outputs;

  explicit DecoderGraph(Context &context) : graph(context, 64, 48) {}

  vsi_nn_tensor_id_t tensor(DataType type = f16, Shape shape = {1024, 1}) {
    return add_tensor(graph, {type, std::move(shape)});
  }

  vsi_nn_tensor_id_t scalar(float value) {
    std::vector<std::uint8_t> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return add_tensor(graph, {f32, {1}}, true, bytes);
  }

  vsi_nn_tensor_id_t weight(const WeightStore &weights, const std::string &name, Shape shape) {
    const auto &record = weights.find(name);
    if (record.spec.type != f16 || record.spec.shape != shape) {
      throw std::invalid_argument("unexpected decoder weight contract: " + name);
    }
    return add_tensor(graph, record.spec, true, weights.read(record, 0, record.bytes));
  }

  vsi_nn_node_t *node(vsi_nn_op_t op, std::initializer_list<vsi_nn_tensor_id_t> inputs,
                      vsi_nn_tensor_id_t output) {
    auto *result = vsi_nn_AddNode(graph.get(), op, 0, 0, nullptr);
    if (!result || result->input.num < inputs.size() || result->output.num < 1) {
      throw std::runtime_error("decoder AddNode failed");
    }
    std::fill_n(result->input.tensors, result->input.num, VSI_NN_TENSOR_ID_NA);
    std::copy(inputs.begin(), inputs.end(), result->input.tensors);
    result->output.tensors[0] = output;
    if (op == VSI_NN_OP_MULTIPLY) {
      result->nn_param.multiply.scale = 1;
    } else if (op == VSI_NN_OP_SWISH) {
      result->nn_param.swish.beta = 1;
      result->nn_param.swish.type = VSI_NN_SWISH;
    }
    return result;
  }

  vsi_nn_tensor_id_t operation(vsi_nn_op_t op, std::initializer_list<vsi_nn_tensor_id_t> inputs,
                               DataType type = f16, Shape shape = {1024, 1}) {
    auto output = tensor(type, std::move(shape));
    node(op, inputs, output);
    return output;
  }

  vsi_nn_tensor_id_t norm(const WeightStore &weights, const std::string &name,
                          vsi_nn_tensor_id_t input) {
    auto value = operation(VSI_NN_OP_DATACONVERT, {input}, f32);
    auto squares = operation(VSI_NN_OP_MULTIPLY, {value, value}, f32);
    auto mean = tensor(f32, {1, 1});
    parameters.push_back({0});
    auto *reduce = node(VSI_NN_OP_REDUCE, {squares}, mean);
    reduce->nn_param.reduce.type = VSI_NN_REDUCE_MEAN;
    reduce->nn_param.reduce.axis = reinterpret_cast<const int32_t *>(parameters.back().data());
    reduce->nn_param.reduce.axis_num = 1;
    reduce->nn_param.reduce.keep_dim = TRUE;
    auto shifted = operation(VSI_NN_OP_ADD, {mean, scalar(1e-6f)}, f32, {1, 1});
    auto inverse = operation(VSI_NN_OP_RSQRT, {shifted}, f32, {1, 1});
    auto normalized = operation(VSI_NN_OP_MULTIPLY, {value, inverse}, f32);
    auto scale = operation(VSI_NN_OP_DATACONVERT, {weight(weights, name, {1024})}, f32, {1024});
    scale = operation(VSI_NN_OP_ADD, {scale, scalar(1)}, f32, {1024});
    auto scaled = operation(VSI_NN_OP_MULTIPLY, {normalized, scale}, f32);
    return operation(VSI_NN_OP_DATACONVERT, {scaled});
  }

  vsi_nn_tensor_id_t project(const WeightStore &weights, const std::string &name,
                             vsi_nn_tensor_id_t input, unsigned columns, unsigned rows) {
    auto result = tensor(f16, {rows, 1});
    auto *projection =
        node(VSI_NN_OP_MATRIXMUL, {input, weight(weights, name, {columns, rows})}, result);
    projection->nn_param.matrixmul.transpose[0] = false;
    projection->nn_param.matrixmul.transpose[1] = true;
    return result;
  }

  void compile(std::initializer_list<vsi_nn_tensor_id_t> inputs) {
    std::vector<vsi_nn_tensor_id_t> input_ids(inputs), output_ids;
    for (const auto &entry : outputs) {
      output_ids.push_back(entry.second);
    }
    if (!vsi_nn_SetGraphInputs(graph.get(), input_ids.data(), input_ids.size()) ||
        !vsi_nn_SetGraphOutputs(graph.get(), output_ids.data(), output_ids.size())) {
      throw std::runtime_error("decoder IO declaration failed");
    }
    check(vsi_nn_SetupGraph(graph.get(), FALSE), "decoder SetupGraph");
    check(vsi_nn_VerifyGraph(graph.get()), "decoder VerifyGraph");
  }
};
} // namespace

struct DecoderLayer::Impl {
  // Destruction reverses this order: consumers, mixers, then their storage owners.
  Graph storage;
  vsi_nn_tensor_id_t mixed;
  DecoderGraph normalization;
  std::unique_ptr<DeltaNet> delta;
  std::unique_ptr<Attention> attention;
  DecoderGraph feed_forward;
  DecoderMetrics timings;
  std::size_t completed = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, unsigned layer, TensorBinding input)
      : storage(context, 1, 0),
        mixed(add_tensor(storage, hidden_spec, false, std::vector<std::uint8_t>(2048))),
        normalization(context), feed_forward(context) {
    if (layer > 3) {
      throw std::invalid_argument("decoder currently supports layers 0-3");
    }
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    const auto incoming = bind_hidden(input, normalization.graph);
    const auto normalized =
        normalization.norm(weights, prefix + "input_layernorm.weight", incoming);
    normalization.outputs = {{"normalized", normalized}};
    normalization.compile({incoming});

    TensorBinding mixer_input{normalization.graph, normalized}, mixer_output{storage, mixed};
    if (layer == 3) {
      attention = std::make_unique<Attention>(context, weights, mixer_input, mixer_output);
    } else {
      delta = std::make_unique<DeltaNet>(context, weights, layer, mixer_input, mixer_output);
    }

    // The original input stays alive until the residual is consumed. No in-place
    // activation writes or alternating public output buffers are involved.
    const auto residual = bind_hidden(input, feed_forward.graph);
    const auto mixer = bind_hidden(mixer_output, feed_forward.graph);
    auto &tail = feed_forward;
    const auto sum = tail.operation(VSI_NN_OP_ADD, {residual, mixer});
    const auto norm = tail.norm(weights, prefix + "post_attention_layernorm.weight", sum);
    const auto gate = tail.project(weights, prefix + "mlp.gate_proj.weight", norm, 1024, 3584);
    const auto up = tail.project(weights, prefix + "mlp.up_proj.weight", norm, 1024, 3584);
    const auto activation = tail.operation(VSI_NN_OP_SWISH, {gate}, f16, {3584, 1});
    const auto product = tail.operation(VSI_NN_OP_MULTIPLY, {activation, up}, f16, {3584, 1});
    const auto mlp = tail.project(weights, prefix + "mlp.down_proj.weight", product, 3584, 1024);
    const auto output = tail.operation(VSI_NN_OP_ADD, {sum, mlp});
    tail.outputs = {{"residual", sum}, {"post_norm", norm}, {"mlp", mlp}, {"output", output}};
    tail.compile({residual, mixer});
  }
};

DecoderLayer::DecoderLayer(Context &context, const WeightStore &weights, unsigned layer,
                           TensorBinding input)
    : impl_(std::make_unique<Impl>(context, weights, layer, input)) {}

DecoderLayer::~DecoderLayer() = default;

void DecoderLayer::step() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder layer must be recreated");
  }
  if (impl_->completed >= KvCache::maximum_capacity) {
    throw std::out_of_range("decoder capacity exhausted");
  }
  auto &state = *impl_;
  if ((state.delta && state.delta->steps() != state.completed) ||
      (state.attention && state.attention->length() != state.completed)) {
    state.failed = true;
    throw std::logic_error("decoder mixer position is inconsistent");
  }
  state.failed = true;
  auto start = Clock::now();
  check(vsi_nn_RunGraph(state.normalization.graph.get()), "decoder input normalization");
  state.timings.normalization_seconds += elapsed(start);
  start = Clock::now();
  if (state.delta) {
    state.delta->step();
  } else {
    state.attention->step();
  }
  state.timings.mixer_seconds += elapsed(start);
  start = Clock::now();
  check(vsi_nn_RunGraph(state.feed_forward.graph.get()), "decoder residual and MLP");
  state.timings.feed_forward_seconds += elapsed(start);
  ++state.timings.steps;
  ++state.completed;
  state.failed = false;
}

void DecoderLayer::reset() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder layer must be recreated");
  }
  impl_->failed = true;
  if (impl_->delta) {
    impl_->delta->reset();
  } else {
    impl_->attention->reset();
  }
  impl_->completed = 0;
  impl_->failed = false;
}

std::size_t DecoderLayer::length() const { return impl_ ? impl_->completed : 0; }

TensorBinding DecoderLayer::output() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("invalid decoder output binding");
  }
  return {impl_->feed_forward.graph, impl_->feed_forward.outputs.at("output")};
}

std::vector<std::uint8_t> DecoderLayer::read(const std::string &name) {
  if (!impl_ || impl_->failed || !impl_->completed) {
    throw std::logic_error("decoder layer has no completed output");
  }
  if (name == "normalized") {
    return read_tensor(impl_->normalization.graph, impl_->normalization.outputs.at(name));
  }
  if (name == "mixer") {
    return read_tensor(impl_->storage, impl_->mixed);
  }
  const auto &outputs = impl_->feed_forward.outputs;
  if (outputs.count(name)) {
    return read_tensor(impl_->feed_forward.graph, outputs.at(name));
  }
  return impl_->delta ? impl_->delta->read(name) : impl_->attention->read(name);
}

DecoderMetrics DecoderLayer::metrics() const {
  if (!impl_) {
    return {};
  }
  auto result = impl_->timings;
  if (impl_->attention) {
    result.cache_writes = impl_->attention->cache_writes();
    result.cache_revalidations = impl_->attention->cache_revalidations();
    result.cache_write_seconds = impl_->attention->cache_write_seconds();
    result.cache_revalidation_seconds = impl_->attention->cache_revalidation_seconds();
  }
  return result;
}

void DecoderLayer::close() {
  if (impl_) {
    impl_->feed_forward.graph.close();
    if (impl_->attention) {
      impl_->attention->close();
    }
    if (impl_->delta) {
      impl_->delta->close();
    }
    impl_->normalization.graph.close();
    impl_->storage.close();
    impl_.reset();
  }
}

struct DecoderGroup::Impl {
  Graph storage;
  vsi_nn_tensor_id_t input;
  unsigned first;
  std::vector<std::unique_ptr<DecoderLayer>> layers;
  std::size_t completed = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, unsigned first_layer)
      : storage(context, 1, 0),
        input(add_tensor(storage, hidden_spec, false, std::vector<std::uint8_t>(2048))),
        first(first_layer) {
    if (first != 0 && first != 3) {
      throw std::invalid_argument("decoder group starts at layer 0 or 3");
    }
    try {
      for (unsigned layer = first; layer < 4; ++layer) {
        auto binding = layers.empty() ? TensorBinding{storage, input} : layers.back()->output();
        layers.push_back(std::make_unique<DecoderLayer>(context, weights, layer, binding));
      }
    } catch (...) {
      while (!layers.empty()) {
        layers.pop_back();
      }
      throw;
    }
  }

  ~Impl() {
    // Explicit reverse destruction also covers partial construction/exceptions.
    while (!layers.empty()) {
      layers.pop_back();
    }
  }
};

DecoderGroup::DecoderGroup(Context &context, const WeightStore &weights, unsigned first_layer)
    : impl_(std::make_unique<Impl>(context, weights, first_layer)) {}

DecoderGroup::~DecoderGroup() = default;

void DecoderGroup::step(const std::vector<std::uint8_t> &hidden_fp16) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder group must be recreated");
  }
  if (hidden_fp16.size() != hidden_spec.bytes()) {
    throw std::invalid_argument("decoder expects one FP16 hidden vector");
  }
  if (impl_->completed >= KvCache::maximum_capacity) {
    throw std::out_of_range("decoder group capacity exhausted");
  }
  // Reject before any layer advances. A later SDK failure instead leaves the
  // whole group invalid; changing just its public length cannot undo recurrence.
  for (const auto &layer : impl_->layers) {
    if (layer->length() != impl_->completed) {
      impl_->failed = true;
      throw std::logic_error("decoder layer positions disagree");
    }
  }
  impl_->failed = true;
  upload_tensor(impl_->storage, impl_->input, hidden_fp16);
  for (const auto &layer : impl_->layers) {
    layer->step();
  }
  ++impl_->completed;
  impl_->failed = false;
}

void DecoderGroup::reset() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder group must be recreated");
  }
  impl_->failed = true;
  for (const auto &layer : impl_->layers) {
    layer->reset();
  }
  impl_->completed = 0;
  impl_->failed = false;
}

std::size_t DecoderGroup::length() const { return impl_ ? impl_->completed : 0; }

std::vector<std::uint8_t> DecoderGroup::read(unsigned layer, const std::string &name) {
  if (!impl_ || impl_->failed || !impl_->completed || layer < impl_->first || layer >= 4) {
    throw std::logic_error("decoder group has no valid output for this layer");
  }
  return impl_->layers.at(layer - impl_->first)->read(name);
}

std::vector<DecoderMetrics> DecoderGroup::metrics() const {
  std::vector<DecoderMetrics> result;
  if (impl_) {
    for (const auto &layer : impl_->layers) {
      result.push_back(layer->metrics());
    }
  }
  return result;
}

void DecoderGroup::close() {
  if (impl_) {
    while (!impl_->layers.empty()) {
      impl_->layers.back()->close();
      impl_->layers.pop_back();
    }
    impl_->storage.close();
    impl_.reset();
  }
}
} // namespace specferry::np101
