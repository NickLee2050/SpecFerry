#include "models/qwen3_5/delta_net.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "vsi_nn_pub.h"

#include <map>
#include <optional>
#include <stdexcept>

namespace specferry::models::qwen3_5 {
using namespace np101;
using namespace np101::ops;

namespace {
using Shape = std::vector<std::uint32_t>;
constexpr auto f16 = DataType::Float16;
constexpr auto f32 = DataType::Float32;

Tensor share_state(Graph &source, Tensor tensor, Graph &destination) {
  return {retain_tensor(source, tensor.id, destination), tensor.spec};
}

class StepGraph : public GraphBuilder {
public:
  Config config;
  std::string prefix;
  Tensor input;
  Tensor result;
  Tensor recurrent_input;
  Tensor convolution_input;
  Tensor recurrent_output;
  Tensor convolution_output;
  std::map<std::string, Tensor> outputs;

  StepGraph(Context &context, const WeightStore &weights, const Config &configuration,
            unsigned layer, std::optional<TensorBinding> input_binding,
            std::optional<TensorBinding> output_binding, StepGraph *previous = nullptr)
      : GraphBuilder(context, 160, 128), config(configuration),
        prefix(config.layer_prefix(layer) + "linear_attn.") {
    const auto hidden = config.hidden_spec();
    if (input_binding && output_binding) {
      input = {bind_tensor(*input_binding, graph, hidden), hidden};
      result = {bind_tensor(*output_binding, graph, hidden), hidden};
    } else if (previous) {
      input = share_state(previous->graph, previous->input, graph);
      result = share_state(previous->graph, previous->result, graph);
    } else {
      input = tensor(hidden);
      result = tensor(hidden);
    }
    if (previous) {
      recurrent_input = share_state(previous->graph, previous->recurrent_output, graph);
      recurrent_output = share_state(previous->graph, previous->recurrent_input, graph);
      convolution_input = share_state(previous->graph, previous->convolution_output, graph);
      convolution_output = share_state(previous->graph, previous->convolution_input, graph);
    } else {
      recurrent_input = tensor(config.delta.recurrent());
      recurrent_output = tensor(config.delta.recurrent());
      convolution_input = tensor(config.delta.convolution());
      convolution_output = tensor(config.delta.convolution());
    }
    build(weights);
    std::vector<Tensor> snapshots;
    for (const auto &entry : outputs) {
      snapshots.push_back(entry.second);
    }
    compile({input, recurrent_input, convolution_input}, snapshots);
  }

private:
  Tensor weight(const WeightStore &weights, const std::string &name, TensorSpec expected) {
    return GraphBuilder::weight(weights, weights.find(prefix + name), expected);
  }

  Tensor project(const WeightStore &weights, Tensor value, const std::string &name,
                 unsigned width) {
    const auto &record = weights.find(prefix + name + ".weight");
    if (record.spec.shape != Shape{value.spec.shape[0], width}) {
      throw std::invalid_argument("unexpected Qwen projection dimensions");
    }
    return GraphBuilder::project(value, weights, record);
  }

  Tensor recurrence(Tensor convolved, Tensor decay, Tensor beta) {
    const auto &spec = config.delta;
    const auto kw = spec.key_width(), vw = spec.value_width();
    auto query = reshape(slice(convolved, {0, 0}, {kw, 1}), {spec.key_dim, 1, spec.heads});
    auto key = reshape(slice(convolved, {kw, 0}, {kw, 1}), {spec.key_dim, 1, spec.heads});
    auto value = reshape(slice(convolved, {2 * kw, 0}, {vw, 1}), {spec.value_dim, 1, spec.heads});
    // The official L2 epsilon is independent of the model RMSNorm epsilon.
    return delta_recurrence(*this, query, key, value, decay, beta, recurrent_input,
                            recurrent_output, 1e-6f);
  }

  void build(const WeightStore &weights) {
    const auto &spec = config.delta;
    const auto channels = spec.channels(), width = spec.value_width(), heads = spec.heads;
    auto packed = project(weights, input, "in_proj_qkv", channels);
    auto gate = project(weights, input, "in_proj_z", width);
    auto a = project(weights, input, "in_proj_a", heads);
    auto b = project(weights, input, "in_proj_b", heads);
    auto kernel = weight(weights, "conv1d.weight", {f16, {spec.convolution_width, 1, channels}});
    auto convolved =
        short_convolution(*this, convolution_input, packed, kernel, convolution_output);

    auto bias = reshape(convert(weight(weights, "dt_bias", {f16, {heads}}), f32), {heads, 1});
    auto time = unary(VSI_NN_OP_SOFTRELU, binary(VSI_NN_OP_ADD, convert(a, f32), bias), f32);
    auto rate = unary(VSI_NN_OP_EXP, weight(weights, "A_log", {f32, {heads}}), f32);
    rate = reshape(binary(VSI_NN_OP_MULTIPLY, rate, scalar(-1.0f)), {heads, 1});
    auto decay = unary(VSI_NN_OP_EXP, binary(VSI_NN_OP_MULTIPLY, time, rate), f32);
    auto beta = convert(unary(VSI_NN_OP_SIGMOID, b, f16), f32);
    auto core = recurrence(convolved, reshape(decay, {1, 1, heads}), reshape(beta, {1, 1, heads}));

    auto normalized = normalize(reshape(core, {spec.value_dim, heads}), true, spec.epsilon);
    normalized = convert(convert(normalized, f16), f32);
    auto scale =
        reshape(weight(weights, "norm.weight", {f32, {spec.value_dim}}), {spec.value_dim, 1});
    auto scaled = binary(VSI_NN_OP_MULTIPLY, normalized, scale);
    auto activated_gate =
        unary(VSI_NN_OP_SWISH, convert(reshape(gate, {spec.value_dim, heads}), f32), f32);
    auto gated = convert(binary(VSI_NN_OP_MULTIPLY, scaled, activated_gate), f16);
    auto matrix = weight(weights, "out_proj.weight", {f16, {width, config.hidden}});
    auto *projection = node(VSI_NN_OP_MATRIXMUL, {reshape(gated, {width, 1}), matrix}, result);
    projection->nn_param.matrixmul.transpose[0] = false;
    projection->nn_param.matrixmul.transpose[1] = true;
    outputs = {{"output", result},
               {"recurrent", recurrent_output},
               {"convolution", convolution_output},
               {"qkv", packed},
               {"convolved", convolved},
               {"decay", decay},
               {"beta", beta},
               {"core", core},
               {"gate", gate},
               {"gated", gated}};
  }
};
} // namespace

struct DeltaNet::Impl {
  std::unique_ptr<StepGraph> forward;
  std::unique_ptr<StepGraph> backward;
  std::size_t completed_steps = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
       std::optional<TensorBinding> input = {}, std::optional<TensorBinding> output = {}) {
    forward = std::make_unique<StepGraph>(context, weights, config, layer, input, output);
    backward =
        std::make_unique<StepGraph>(context, weights, config, layer, input, output, forward.get());
  }
};

DeltaNet::DeltaNet(Context &context, const WeightStore &weights, const Config &config,
                   unsigned layer) {
  validate_layer_weights(weights, config, layer, false);
  if (config.mixer(layer) != MixerKind::DeltaNet) {
    throw std::invalid_argument("layer is not DeltaNet");
  }
  impl_ = std::make_unique<Impl>(context, weights, config, layer);
  reset();
}

DeltaNet::DeltaNet(Context &context, const WeightStore &weights, const Config &config,
                   unsigned layer, TensorBinding input, TensorBinding output) {
  validate_layer_weights(weights, config, layer, false);
  if (config.mixer(layer) != MixerKind::DeltaNet) {
    throw std::invalid_argument("layer is not DeltaNet");
  }
  impl_ = std::make_unique<Impl>(context, weights, config, layer, input, output);
  reset();
}

DeltaNet::~DeltaNet() = default;

TensorSpec DeltaNet::recurrent_spec() const {
  if (!impl_) {
    throw std::logic_error("closed DeltaNet");
  }
  return impl_->forward->config.delta.recurrent();
}

TensorSpec DeltaNet::convolution_spec() const {
  if (!impl_) {
    throw std::logic_error("closed DeltaNet");
  }
  return impl_->forward->config.delta.convolution();
}

TensorSpec DeltaNet::output_spec(const std::string &name) const {
  if (!impl_) {
    throw std::logic_error("closed DeltaNet");
  }
  return impl_->forward->outputs.at(name).spec;
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
  if (hidden_fp16.size() != impl_->forward->config.hidden_spec().bytes()) {
    throw std::invalid_argument("DeltaNet expects one FP16 hidden vector");
  }
  auto &step = impl_->completed_steps % 2 == 0 ? *impl_->forward : *impl_->backward;
  impl_->failed = true;
  upload_tensor(step.graph, step.input.id, hidden_fp16);
  impl_->failed = false;
  this->step();
}

void DeltaNet::step() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("DeltaNet must be initialized/reset before stepping");
  }
  auto &step = impl_->completed_steps % 2 == 0 ? *impl_->forward : *impl_->backward;
  impl_->failed = true;
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
} // namespace specferry::models::qwen3_5
