#include "models/qwen3_5/attention.hpp"
#include "models/qwen3_5/decoder.hpp"
#include "models/qwen3_5/delta_net.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_pub.h"

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

  DecoderGraph(Context &context, const Config &configuration, WeightBank *bank = nullptr)
      : GraphBuilder(context, 64, 48, bank), config(configuration) {}

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
  std::unique_ptr<DecoderGraph> normalization;
  std::unique_ptr<DeltaNet> delta;
  std::unique_ptr<Attention> attention;
  DecoderGraph feed_forward;
  std::size_t completed = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, const Config &configuration, unsigned layer,
       TensorBinding input, WeightBank *bank, LinearBackend backend, const CompiledBlock *compiled,
       const CompiledBlock *compiled_feed_forward, bool deferred_delta_commit)
      : config(configuration), storage(context, 1, 0),
        mixed(add_tensor(storage, config.hidden_spec(), false,
                         std::vector<std::uint8_t>(config.hidden_spec().bytes()))),
        feed_forward(context, config, bank) {
    const auto prefix = config.layer_prefix(layer);
    TensorBinding mixer_output{storage, mixed};
    if (!compiled || !compiled->includes_input_norm) {
      normalization = std::make_unique<DecoderGraph>(context, config, bank);
      const auto incoming = normalization->bind(input, config.hidden_spec());
      const auto normalized =
          normalization->norm(weights, prefix + "input_layernorm.weight", incoming);
      normalization->outputs = {{"normalized", normalized}};
      normalization->compile_outputs({incoming});
    }
    const auto mixer_input =
        normalization
            ? TensorBinding{normalization->graph, normalization->outputs.at("normalized").id}
            : input;
    if (config.mixer(layer) == MixerKind::Attention) {
      if (compiled) {
        throw std::invalid_argument("DeltaNet compiled mixer cannot replace attention");
      }
      attention = std::make_unique<Attention>(context, weights, config, layer, mixer_input,
                                              mixer_output, bank);
    } else {
      if (!compiled) {
        throw std::invalid_argument("Qwen DeltaNet requires its compiled mixer");
      }
      delta = std::make_unique<DeltaNet>(context, weights, config, layer, mixer_input, mixer_output,
                                         *compiled, deferred_delta_commit);
    }

    // The original input stays alive until the residual is consumed. No in-place
    // activation writes or alternating public output buffers are involved.
    const auto residual = feed_forward.bind(input, config.hidden_spec());
    const auto mixer = feed_forward.bind(
        deferred_delta_commit && delta ? delta->output() : mixer_output, config.hidden_spec());
    auto &tail = feed_forward;
    if (compiled_feed_forward) {
      validate_compiled_feed_forward(*compiled_feed_forward, weights, config, layer);
      const std::vector<Tensor> semantic{residual, mixer};
      auto inputs = semantic;
      for (unsigned index = 0; index < semantic.size(); ++index) {
        inputs[compiled_feed_forward->input_order[index]] = semantic[index];
      }
      const auto outputs = tail.compiled(compiled_feed_forward->graph, inputs);
      tail.outputs = {{"output", outputs.at(compiled_feed_forward->output_order.at(0))}};
      tail.compile_outputs({residual, mixer});
      return;
    }
    const auto sum = tail.binary(VSI_NN_OP_ADD, residual, mixer);
    const auto norm = tail.norm(weights, prefix + "post_attention_layernorm.weight", sum);
    const auto mlp = swiglu(tail, norm, weights, weights.find(prefix + "mlp.gate_proj.weight"),
                            weights.find(prefix + "mlp.up_proj.weight"),
                            weights.find(prefix + "mlp.down_proj.weight"), backend);
    const auto output = tail.binary(VSI_NN_OP_ADD, sum, mlp);
    tail.outputs = {{"residual", sum}, {"post_norm", norm}, {"mlp", mlp}, {"output", output}};
    tail.compile_outputs({residual, mixer});
  }
};

DecoderLayer::DecoderLayer(Context &context, const WeightStore &weights, const Config &config,
                           unsigned layer, TensorBinding input, WeightBank *bank,
                           LinearBackend backend, const CompiledBlock *compiled,
                           const CompiledBlock *compiled_feed_forward, bool deferred_delta_commit)
    : impl_(nullptr) {
  validate_layer_weights(weights, config, layer, true);
  impl_ = std::make_unique<Impl>(context, weights, config, layer, input, bank, backend, compiled,
                                 compiled_feed_forward, deferred_delta_commit);
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
  if (state.normalization) {
    check(sdk_call("vsi_nn_RunGraph",
                   [&] { return vsi_nn_RunGraph(state.normalization->graph.get()); }),
          "decoder input normalization");
  }
  if (state.delta) {
    state.delta->step();
  } else {
    state.attention->step();
  }
  check(
      sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(state.feed_forward.graph.get()); }),
      "decoder residual and MLP");
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

std::vector<TensorCopyBinding> DecoderLayer::state_copies() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("state copies require a live decoder layer");
  }
  return impl_->delta ? impl_->delta->state_copies() : std::vector<TensorCopyBinding>{};
}

void DecoderLayer::confirm_state_commit() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("state commit requires a live decoder layer");
  }
  if (impl_->delta) {
    impl_->delta->confirm_state_commit();
  }
}

std::vector<std::uint8_t> DecoderLayer::read(const std::string &name, TensorReadback method) {
  if (!impl_ || impl_->failed || !impl_->completed) {
    throw std::logic_error("decoder layer has no completed output");
  }
  if (name == "normalized") {
    if (!impl_->normalization) {
      throw std::invalid_argument("compiled input normalization has no public diagnostic output");
    }
    return read_tensor(impl_->normalization->graph, impl_->normalization->outputs.at(name).id,
                       method);
  }
  if (name == "mixer") {
    if (impl_->delta) {
      return impl_->delta->read("output", method);
    }
    return read_tensor(impl_->storage, impl_->mixed, method);
  }
  const auto &outputs = impl_->feed_forward.outputs;
  if (outputs.count(name)) {
    return read_tensor(impl_->feed_forward.graph, outputs.at(name).id, method);
  }
  return impl_->delta ? impl_->delta->read(name, method) : impl_->attention->read(name, method);
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
    if (impl_->normalization) {
      impl_->normalization->graph.close();
    }
    impl_->storage.close();
    impl_.reset();
  }
}

} // namespace specferry::models::qwen3_5
