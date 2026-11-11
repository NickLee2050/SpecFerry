#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/ops/graph_builder.hpp"
#include "devices/np101/api/ops/mixers.hpp"
#include "devices/np101/api/tensor_spec.hpp"
#include "devices/np101/api/weight_bank.hpp"
#include "models/qwen3_5/attention.hpp"
#include "models/qwen3_5/decoder.hpp"
#include "models/qwen3_5/delta_net.hpp"
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
  TensorBinding incoming;
  unsigned batch_tokens;
  bool commit_state_in_ffn;
  Graph storage;
  vsi_nn_tensor_id_t mixed;
  vsi_nn_tensor_id_t public_output = VSI_NN_TENSOR_ID_NA;
  std::unique_ptr<DecoderGraph> normalization;
  std::unique_ptr<DeltaNet> delta;
  std::unique_ptr<Attention> attention;
  DecoderGraph feed_forward;
  std::unique_ptr<GraphBuilder> local_commit;
  vsi_nn_tensor_id_t residual_row = VSI_NN_TENSOR_ID_NA, mixer_row = VSI_NN_TENSOR_ID_NA;
  vsi_nn_tensor_id_t residual_batch = VSI_NN_TENSOR_ID_NA, mixer_batch = VSI_NN_TENSOR_ID_NA;
  std::vector<std::uint8_t> residual_bytes, mixer_bytes, row_bytes;
  std::size_t completed = 0;
  bool failed = false;

  Impl(Context &context, const WeightStore &weights, const Config &configuration, unsigned layer,
       TensorBinding input, WeightBank *bank, LinearBackend backend, const CompiledBlock *compiled,
       const CompiledBlock *compiled_feed_forward, bool deferred_delta_commit,
       KvWriteMode kv_write_mode, bool fuse_attention_input_norm, bool commit_in_ffn)
      : config(configuration), incoming(input),
        batch_tokens(
            compiled_feed_forward ? compiled_feed_forward_batch_tokens(*compiled_feed_forward) : 1),
        commit_state_in_ffn(commit_in_ffn), storage(context, batch_tokens > 1 ? 2 : 1, 0),
        mixed(add_tensor(storage, config.hidden_spec(), false,
                         std::vector<std::uint8_t>(config.hidden_spec().bytes()))),
        feed_forward(context, config, bank) {
    const auto prefix = config.layer_prefix(layer);
    TensorBinding mixer_output{storage, mixed};
    const bool fused_attention_norm =
        fuse_attention_input_norm && config.mixer(layer) == MixerKind::Attention;
    if ((!compiled || !compiled->includes_input_norm) && !fused_attention_norm) {
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
      attention = std::make_unique<Attention>(
          context, weights, config, layer, mixer_input, mixer_output, bank,
          backend == LinearBackend::SharedFullyConnected, kv_write_mode, fused_attention_norm);
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
    residual_row = residual.id;
    mixer_row = mixer.id;
    auto &tail = feed_forward;
    if (compiled_feed_forward) {
      validate_compiled_feed_forward(*compiled_feed_forward, weights, config, layer);
      std::vector<Tensor> semantic{residual, mixer};
      if (batch_tokens > 1) {
        const TensorSpec batch_spec{f16, {config.hidden, batch_tokens}};
        semantic = {tail.tensor(batch_spec), tail.tensor(batch_spec)};
        residual_batch = semantic[0].id;
        mixer_batch = semantic[1].id;
        residual_bytes.resize(batch_spec.bytes());
        mixer_bytes.resize(batch_spec.bytes());
        row_bytes.resize(config.hidden_spec().bytes());
        public_output = add_tensor(storage, config.hidden_spec(), false, row_bytes);
      }
      auto inputs = semantic;
      for (unsigned index = 0; index < semantic.size(); ++index) {
        inputs[compiled_feed_forward->input_order[index]] = semantic[index];
      }
      const auto outputs = tail.compiled(compiled_feed_forward->graph, inputs);
      tail.outputs = {{"output", outputs.at(compiled_feed_forward->output_order.at(0))}};
      if (commit_state_in_ffn && delta) {
        // FFN never reads the old recurrent state. Copying here has no feedback
        // edge into DeltaNet, which already finished before this graph runs.
        // Preserve the NBG wrapper's original graph IO. Promoting the state
        // copies to extra graph ports corrupted FFN output on the pinned SoC
        // SDK, despite correct copied states. Ordinary retained tensors keep
        // these fixed copy bindings alive without changing that boundary.
        for (const auto &copy : delta->state_copies()) {
          const auto source = tail.bind(copy.source, copy.spec);
          const auto destination = tail.bind(copy.destination, copy.spec);
          tail.node(VSI_NN_OP_DATACONVERT, {source}, destination);
        }
      }
      tail.compile_outputs(semantic);
      if (batch_tokens > 1 && delta && deferred_delta_commit) {
        // Layer-major prefill must commit this layer before its next token. The
        // global decode commit would also overwrite states of inactive layers.
        local_commit = std::make_unique<GraphBuilder>(context, 4, 2);
        std::vector<Tensor> sources, destinations;
        for (const auto &copy : delta->state_copies()) {
          const auto source = local_commit->bind(copy.source, copy.spec);
          const auto destination = local_commit->bind(copy.destination, copy.spec);
          local_commit->node(VSI_NN_OP_DATACONVERT, {source}, destination);
          sources.push_back(source);
          destinations.push_back(destination);
        }
        local_commit->compile(sources, destinations);
      }
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

  void run_mixer() {
    if (normalization) {
      check(
          sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(normalization->graph.get()); }),
          "decoder input normalization");
    }
    if (delta) {
      delta->step();
    } else {
      attention->step();
    }
  }

  void require_position() {
    if ((delta && delta->steps() != completed) || (attention && attention->length() != completed)) {
      failed = true;
      throw std::logic_error("decoder mixer position is inconsistent");
    }
  }

  std::vector<std::uint8_t> run_batch_ffn(unsigned count) {
    const std::vector<std::uint32_t> start{0, 0}, end{config.hidden, batch_tokens};
    upload_tensor_region(feed_forward.graph, residual_batch, start, end, residual_bytes);
    upload_tensor_region(feed_forward.graph, mixer_batch, start, end, mixer_bytes);
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(feed_forward.graph.get()); }),
          "decoder batched residual and MLP");
    if (commit_state_in_ffn && delta) {
      // A failed RunGraph must leave the pending state unconfirmed and the
      // decoder failed; no subsequent token may consume a partial state update.
      delta->confirm_state_commit();
    }
    auto output = read_tensor_region(feed_forward.graph, feed_forward.outputs.at("output").id,
                                     start, {config.hidden, count});
    std::copy(output.end() - row_bytes.size(), output.end(), row_bytes.begin());
    upload_tensor_region(storage, public_output, start, {config.hidden, 1}, row_bytes);
    return output;
  }
};

DecoderLayer::DecoderLayer(Context &context, const WeightStore &weights, const Config &config,
                           unsigned layer, TensorBinding input, WeightBank *bank,
                           LinearBackend backend, const CompiledBlock *compiled,
                           const CompiledBlock *compiled_feed_forward, bool deferred_delta_commit,
                           KvWriteMode kv_write_mode, bool fuse_attention_input_norm,
                           bool commit_state_in_ffn)
    : impl_(nullptr) {
  validate_layer_weights(weights, config, layer, true);
  if (commit_state_in_ffn && (!deferred_delta_commit || !compiled_feed_forward ||
                              compiled_feed_forward_batch_tokens(*compiled_feed_forward) == 1)) {
    throw std::invalid_argument("FFN state commit requires deferred batched execution");
  }
  impl_ = std::make_unique<Impl>(context, weights, config, layer, input, bank, backend, compiled,
                                 compiled_feed_forward, deferred_delta_commit, kv_write_mode,
                                 fuse_attention_input_norm, commit_state_in_ffn);
}

DecoderLayer::~DecoderLayer() = default;

std::size_t DecoderLayer::kv_revalidations() const {
  return impl_ && impl_->attention ? impl_->attention->kv_revalidations() : 0;
}

void DecoderLayer::step() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder layer must be recreated");
  }
  if (impl_->completed >= impl_->config.kv.capacity) {
    throw std::out_of_range("decoder capacity exhausted");
  }
  auto &state = *impl_;
  state.require_position();
  state.failed = true;
  state.run_mixer();
  if (state.batch_tokens > 1) {
    std::fill(state.residual_bytes.begin(), state.residual_bytes.end(), 0);
    read_tensor_region_into(state.feed_forward.graph, state.residual_row, {0, 0},
                            {state.config.hidden, 1}, state.row_bytes);
    std::copy(state.row_bytes.begin(), state.row_bytes.end(), state.residual_bytes.begin());
    std::fill(state.mixer_bytes.begin(), state.mixer_bytes.end(), 0);
    read_tensor_region_into(state.feed_forward.graph, state.mixer_row, {0, 0},
                            {state.config.hidden, 1}, state.row_bytes);
    std::copy(state.row_bytes.begin(), state.row_bytes.end(), state.mixer_bytes.begin());
    state.run_batch_ffn(1);
  } else {
    check(sdk_call("vsi_nn_RunGraph",
                   [&] { return vsi_nn_RunGraph(state.feed_forward.graph.get()); }),
          "decoder residual and MLP");
  }
  ++state.completed;
  state.failed = false;
}

unsigned DecoderLayer::feed_forward_batch_tokens() const {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder layer must be recreated");
  }
  return impl_->batch_tokens;
}

std::vector<std::uint8_t> DecoderLayer::prefill_block(const std::vector<std::uint8_t> &packed_rows,
                                                      unsigned count) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed decoder layer must be recreated");
  }
  auto &state = *impl_;
  const auto row_size = state.config.hidden_spec().bytes();
  if (state.batch_tokens == 1 || !count || count > state.batch_tokens ||
      packed_rows.size() != count * row_size) {
    throw std::invalid_argument("prefill requires a valid fixed-batch FFN block");
  }
  if (state.completed > state.config.kv.capacity ||
      count > state.config.kv.capacity - state.completed) {
    throw std::out_of_range("decoder capacity exhausted by prefill block");
  }
  state.require_position();
  state.failed = true;
  std::fill(state.residual_bytes.begin(), state.residual_bytes.end(), 0);
  std::copy(packed_rows.begin(), packed_rows.end(), state.residual_bytes.begin());
  std::fill(state.mixer_bytes.begin(), state.mixer_bytes.end(), 0);
  for (unsigned row = 0; row < count; ++row) {
    std::copy_n(packed_rows.begin() + row * row_size, row_size, state.row_bytes.begin());
    upload_tensor_region(state.incoming.owner, state.incoming.id, {0, 0}, {state.config.hidden, 1},
                         state.row_bytes);
    state.run_mixer();
    read_tensor_region_into(state.feed_forward.graph, state.mixer_row, {0, 0},
                            {state.config.hidden, 1}, state.row_bytes);
    std::copy(state.row_bytes.begin(), state.row_bytes.end(),
              state.mixer_bytes.begin() + row * row_size);
    if (state.local_commit && (!state.commit_state_in_ffn || row + 1 < count)) {
      check(sdk_call("vsi_nn_RunGraph",
                     [&] { return vsi_nn_RunGraph(state.local_commit->graph.get()); }),
            "prefill layer DeltaNet state commit");
      state.delta->confirm_state_commit();
    }
    ++state.completed;
  }
  auto output = state.run_batch_ffn(count);
  state.failed = false;
  return output;
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
  if (impl_->batch_tokens > 1) {
    return {impl_->storage, impl_->public_output};
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
      throw std::invalid_argument("fused input normalization has no separate diagnostic output");
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
  if (name == "output" && impl_->batch_tokens > 1) {
    return read_tensor(impl_->storage, impl_->public_output, method);
  }
  const auto &outputs = impl_->feed_forward.outputs;
  if (outputs.count(name)) {
    return read_tensor(impl_->feed_forward.graph, outputs.at(name).id, method);
  }
  return impl_->delta ? impl_->delta->read(name, method) : impl_->attention->read(name, method);
}

void DecoderLayer::close() {
  if (impl_) {
    if (impl_->local_commit) {
      impl_->local_commit->graph.close();
      impl_->local_commit.reset();
    }
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
