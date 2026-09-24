#include "models/qwen3_5/attention.hpp"
#include "models/qwen3_5/decoder.hpp"
#include "models/qwen3_5/delta_net.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace specferry::models::qwen3_5 {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;

// Qwen chooses the norm convention; the arithmetic is shared with other callers.
class DecoderGraph : public GraphBuilder {
public:
  Config config;
  std::map<std::string, Tensor> outputs;

  DecoderGraph(Context &context, const Config &configuration)
      : GraphBuilder(context, 64, 48), config(configuration) {}

  Tensor norm(const WeightStore &weights, const std::string &name, Tensor input) {
    auto scale = weight(weights, weights.find(name), {f16, {config.hidden}});
    return rms_norm(input, scale, config.epsilon, 1.0f, f16);
  }

  void compile_outputs(const std::vector<Tensor> &inputs) {
    std::vector<Tensor> tensors;
    for (const auto &entry : outputs) {
      tensors.push_back(entry.second);
    }
    compile(inputs, tensors);
  }
};
} // namespace

struct DecoderLayer::Impl {
  // Destruction reverses this order: consumers, mixers, then their storage owners.
  Config config;
  Graph storage;
  vsi_nn_tensor_id_t mixed;
  DecoderGraph normalization;
  std::unique_ptr<DeltaNet> delta;
  std::unique_ptr<Attention> attention;
  DecoderGraph feed_forward;
  std::size_t completed = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, const Config &configuration, unsigned layer,
       TensorBinding input)
      : config(configuration), storage(context, 1, 0),
        mixed(add_tensor(storage, config.hidden_spec(), false,
                         std::vector<std::uint8_t>(config.hidden_spec().bytes()))),
        normalization(context, config), feed_forward(context, config) {
    const auto prefix = config.layer_prefix(layer);
    const auto incoming = normalization.bind(input, config.hidden_spec());
    const auto normalized =
        normalization.norm(weights, prefix + "input_layernorm.weight", incoming);
    normalization.outputs = {{"normalized", normalized}};
    normalization.compile_outputs({incoming});

    TensorBinding mixer_input{normalization.graph, normalized.id}, mixer_output{storage, mixed};
    if (config.mixer(layer) == MixerKind::Attention) {
      attention =
          std::make_unique<Attention>(context, weights, config, layer, mixer_input, mixer_output);
    } else {
      delta =
          std::make_unique<DeltaNet>(context, weights, config, layer, mixer_input, mixer_output);
    }

    // The original input stays alive until the residual is consumed. No in-place
    // activation writes or alternating public output buffers are involved.
    const auto residual = feed_forward.bind(input, config.hidden_spec());
    const auto mixer = feed_forward.bind(mixer_output, config.hidden_spec());
    auto &tail = feed_forward;
    const auto sum = tail.binary(VSI_NN_OP_ADD, residual, mixer);
    const auto norm = tail.norm(weights, prefix + "post_attention_layernorm.weight", sum);
    const auto mlp = swiglu(tail, norm, weights, weights.find(prefix + "mlp.gate_proj.weight"),
                            weights.find(prefix + "mlp.up_proj.weight"),
                            weights.find(prefix + "mlp.down_proj.weight"));
    const auto output = tail.binary(VSI_NN_OP_ADD, sum, mlp);
    tail.outputs = {{"residual", sum}, {"post_norm", norm}, {"mlp", mlp}, {"output", output}};
    tail.compile_outputs({residual, mixer});
  }
};

DecoderLayer::DecoderLayer(Context &context, const WeightStore &weights, const Config &config,
                           unsigned layer, TensorBinding input)
    : impl_(nullptr) {
  validate_layer_weights(weights, config, layer, true);
  impl_ = std::make_unique<Impl>(context, weights, config, layer, input);
}

DecoderLayer::~DecoderLayer() = default;

void DecoderLayer::step() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder layer must be recreated");
  }
  if (impl_->completed >= impl_->config.kv.capacity) {
    throw std::out_of_range("decoder capacity exhausted");
  }
  auto &state = *impl_;
  if ((state.delta && state.delta->steps() != state.completed) ||
      (state.attention && state.attention->length() != state.completed)) {
    state.failed = true;
    throw std::logic_error("decoder mixer position is inconsistent");
  }
  state.failed = true;
  check(vsi_nn_RunGraph(state.normalization.graph.get()), "decoder input normalization");
  if (state.delta) {
    state.delta->step();
  } else {
    state.attention->step();
  }
  check(vsi_nn_RunGraph(state.feed_forward.graph.get()), "decoder residual and MLP");
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
  return {impl_->feed_forward.graph, impl_->feed_forward.outputs.at("output").id};
}

std::vector<std::uint8_t> DecoderLayer::read(const std::string &name) {
  if (!impl_ || impl_->failed || !impl_->completed) {
    throw std::logic_error("decoder layer has no completed output");
  }
  if (name == "normalized") {
    return read_tensor(impl_->normalization.graph, impl_->normalization.outputs.at(name).id);
  }
  if (name == "mixer") {
    return read_tensor(impl_->storage, impl_->mixed);
  }
  const auto &outputs = impl_->feed_forward.outputs;
  if (outputs.count(name)) {
    return read_tensor(impl_->feed_forward.graph, outputs.at(name).id);
  }
  return impl_->delta ? impl_->delta->read(name) : impl_->attention->read(name);
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
  Config config;
  Graph storage;
  vsi_nn_tensor_id_t input;
  std::vector<unsigned> selected;
  std::vector<std::unique_ptr<DecoderLayer>> layers;
  std::size_t completed = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, const Config &configuration,
       const std::vector<unsigned> &indices)
      : config(configuration), storage(context, 1, 0),
        input(add_tensor(storage, config.hidden_spec(), false,
                         std::vector<std::uint8_t>(config.hidden_spec().bytes()))),
        selected(indices) {
    try {
      for (unsigned layer : selected) {
        auto binding = layers.empty() ? TensorBinding{storage, input} : layers.back()->output();
        layers.push_back(std::make_unique<DecoderLayer>(context, weights, config, layer, binding));
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

DecoderGroup::DecoderGroup(Context &context, const WeightStore &weights, const Config &config,
                           const std::vector<unsigned> &layers) {
  config.validate();
  if (layers.empty()) {
    throw std::invalid_argument("decoder slice is empty");
  }
  std::vector<unsigned> seen;
  for (auto layer : layers) {
    validate_layer_weights(weights, config, layer, true);
    if (std::find(seen.begin(), seen.end(), layer) != seen.end()) {
      throw std::invalid_argument("decoder slice repeats a layer");
    }
    seen.push_back(layer);
  }
  impl_ = std::make_unique<Impl>(context, weights, config, layers);
}

DecoderGroup::~DecoderGroup() = default;

void DecoderGroup::step(const std::vector<std::uint8_t> &hidden_fp16) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder group must be recreated");
  }
  if (hidden_fp16.size() != impl_->config.hidden_spec().bytes()) {
    throw std::invalid_argument("decoder expects one FP16 hidden vector");
  }
  if (impl_->completed >= impl_->config.kv.capacity) {
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
  if (!impl_ || impl_->failed || !impl_->completed) {
    throw std::logic_error("decoder group has no valid output");
  }
  auto found = std::find(impl_->selected.begin(), impl_->selected.end(), layer);
  if (found == impl_->selected.end()) {
    throw std::out_of_range("layer not in this decoder slice");
  }
  return impl_->layers.at(found - impl_->selected.begin())->read(name);
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
} // namespace specferry::models::qwen3_5
