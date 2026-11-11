#include "models/qwen3_5/decoder.hpp"
#include "models/qwen3_5/model.hpp"
#include "np101/diagnostics.hpp"
#include "np101/memory_budget.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string_view>

namespace specferry::models::qwen3_5 {
using namespace np101;
using namespace np101::ops;

struct Model::Impl {
  Config config;
  inference::GenerationPolicy policy;
  WeightBank bank;
  std::unique_ptr<Vocabulary> vocabulary;
  std::vector<std::unique_ptr<DecoderLayer>> layers;
  std::unique_ptr<GraphBuilder> state_commit;
  std::unique_ptr<GraphBuilder> normalization;
  std::unique_ptr<VocabularyHead> head;
  unsigned position = 0, batch_tokens = 1;
  bool failed = false, prediction_ready = false;
  std::function<void(const std::string &)> progress;
  bool trace_steps = false;

  Impl(Context &context, const WeightStore &weights, const Config &spec,
       const inference::GenerationPolicy &selection,
       const std::map<unsigned, CompiledBlock> &compiled,
       const std::map<unsigned, CompiledBlock> &compiled_feed_forward, const ModelOptions &options,
       const std::function<void(const std::string &)> &phase)
      : config(spec), policy(selection), bank(context), progress(phase) {
    const auto lookup = options.lookup;
    const bool combined_state_commit = options.state_commit != StateCommit::PerLayer;
    const bool commit_state_in_ffn = options.state_commit == StateCommit::FeedForward;
    const auto *timing = std::getenv("SPECFERRY_SDK_TIMING");
    trace_steps = timing && std::string_view(timing) == "calls";
    try {
      if (phase) {
        phase(lookup == VocabularyLookup::GatherNd ? "embedding.gather-nd" : "embedding.gather");
      }
      vocabulary = std::make_unique<Vocabulary>(
          context, weights, weights.find("model.embed_tokens.weight"), 4096, lookup);
      for (unsigned layer = 0; layer < config.layer_types.size(); ++layer) {
        if (phase) {
          phase("layer." + std::to_string(layer));
        }
        const auto artifact = compiled.find(layer);
        const auto feed_forward = compiled_feed_forward.find(layer);
        const auto input = layers.empty() ? vocabulary->embedding() : layers.back()->output();
        layers.push_back(std::make_unique<DecoderLayer>(
            context, weights, config, layer, input, &bank, options.projections,
            artifact == compiled.end() ? nullptr : &artifact->second,
            feed_forward == compiled_feed_forward.end() ? nullptr : &feed_forward->second,
            combined_state_commit, options.kv_write, options.layout != DecoderLayout::Separate,
            commit_state_in_ffn));
      }
      batch_tokens = layers.front()->feed_forward_batch_tokens();
      for (const auto &layer : layers) {
        if (layer->feed_forward_batch_tokens() != batch_tokens) {
          throw std::invalid_argument("Qwen prefill requires one common FFN batch width");
        }
      }
      if (combined_state_commit && !commit_state_in_ffn) {
        if (phase) {
          phase("combined-state-commit");
        }
        state_commit = std::make_unique<GraphBuilder>(context, 72, 36);
        std::vector<Tensor> sources, destinations;
        for (auto &layer : layers) {
          for (const auto &copy : layer->state_copies()) {
            const auto source = state_commit->bind(copy.source, copy.spec);
            const auto destination = state_commit->bind(copy.destination, copy.spec);
            state_commit->node(VSI_NN_OP_DATACONVERT, {source}, destination);
            sources.push_back(source);
            destinations.push_back(destination);
          }
        }
        if (sources.size() != 36) {
          throw std::invalid_argument("combined Qwen commit requires two states for each DeltaNet");
        }
        state_commit->compile(sources, destinations);
      }
      if (phase) {
        phase("final-normalization");
      }
      normalization = std::make_unique<GraphBuilder>(context, 32, 24, &bank);
      const auto hidden = normalization->bind(layers.back()->output(), config.hidden_spec());
      const auto &record = weights.find("model.norm.weight");
      const auto scale =
          normalization->weight(weights, record, {DataType::Float16, {config.hidden}});
      const auto normalized =
          normalization->rms_norm(hidden, scale, config.epsilon, 1.0f, DataType::Float16);
      normalization->compile({hidden}, {normalized});
      if (phase) {
        phase("lm-head");
      }
      head = std::make_unique<VocabularyHead>(
          context, *vocabulary, TensorBinding{normalization->graph, normalized.id},
          SamplingOptions{}, options.head, options.head_blocks_per_graph);
    } catch (...) {
      close();
      throw;
    }
  }

  ~Impl() {
    try {
      close();
    } catch (...) {
      // Explicit close reports release failures; destructors cannot.
    }
  }

  void require_live() const {
    if (failed || !head) {
      throw std::logic_error("closed or failed Qwen model must be recreated");
    }
  }

  void consume(std::int32_t token) {
    require_live();
    if (token < 0 || unsigned(token) >= policy.vocabulary) {
      throw std::invalid_argument("Qwen token exceeds vocabulary");
    }
    if (position >= policy.capacity) {
      throw std::out_of_range("Qwen capacity exhausted");
    }
    for (const auto &layer : layers) {
      if (layer->length() != position) {
        failed = true;
        throw std::logic_error("Qwen layer positions disagree");
      }
    }
    failed = true;
    prediction_ready = false;
    vocabulary->lookup(token);
    for (unsigned layer = 0; layer < layers.size(); ++layer) {
      if (trace_steps && progress) {
        progress("token." + std::to_string(position) + ".decoder." + std::to_string(layer));
      }
      layers[layer]->step();
    }
    if (state_commit) {
      // Every activation has been consumed, but each NBG still owns distinct new
      // state outputs. Commit them before exposing a completed token or step.
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(state_commit->graph.get()); }),
            "Qwen combined DeltaNet state commit");
      for (auto &layer : layers) {
        layer->confirm_state_commit();
      }
    }
    ++position;
    prediction_ready = true;
    failed = false;
  }

  void prefill(const std::vector<std::int32_t> &tokens) {
    require_live();
    if (tokens.size() > policy.capacity - position) {
      throw std::out_of_range("Qwen prompt exceeds remaining capacity");
    }
    for (auto token : tokens) {
      if (token < 0 || unsigned(token) >= policy.vocabulary) {
        throw std::invalid_argument("Qwen prompt token exceeds vocabulary");
      }
    }
    failed = true;
    prediction_ready = false;
    const auto bytes_per_row = config.hidden_spec().bytes();
    for (std::size_t offset = 0; offset < tokens.size(); offset += batch_tokens) {
      const auto count = unsigned(std::min<std::size_t>(batch_tokens, tokens.size() - offset));
      for (const auto &layer : layers) {
        if (layer->length() != position) {
          throw std::logic_error("Qwen prefill layer positions disagree");
        }
      }
      std::vector<std::uint8_t> rows;
      rows.reserve(count * bytes_per_row);
      for (unsigned row = 0; row < count; ++row) {
        vocabulary->lookup(tokens[offset + row]);
        const auto binding = vocabulary->embedding();
        const auto embedding = read_tensor(binding.owner, binding.id, TensorReadback::MappedHost);
        rows.insert(rows.end(), embedding.begin(), embedding.end());
      }
      for (unsigned layer = 0; layer < layers.size(); ++layer) {
        if (trace_steps && progress) {
          progress("token." + std::to_string(position) + ".prefill.decoder." +
                   std::to_string(layer));
        }
        // Every mixer consumes real tokens in order and commits only its own
        // recurrence. Padding exists solely inside the independent FFN rows.
        rows = layers[layer]->prefill_block(rows, count);
        if (rows.size() != count * bytes_per_row || layers[layer]->length() != position + count) {
          throw std::logic_error("Qwen prefill returned an incomplete layer block");
        }
      }
      position += count;
    }
    prediction_ready = !tokens.empty();
    failed = false;
  }

  std::int32_t predict(TensorReadback token_readback) {
    require_live();
    if (!prediction_ready) {
      throw std::logic_error("Qwen has no completed decoder output");
    }
    failed = true;
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(normalization->graph.get()); }),
          "Qwen final normalization");

    const auto token = head->select(token_readback);
    failed = false;
    return token;
  }

  void reset() {
    require_live();
    failed = true;
    if (position) {
      for (auto &layer : layers) {
        layer->reset();
      }
    }
    head->reset();
    position = 0;
    prediction_ready = false;
    failed = false;
  }

  void close() {
    failed = true;
    prediction_ready = false;
    if (head) {
      head->close();
      head.reset();
    }

    if (normalization) {
      normalization->graph.close();
      normalization.reset();
    }
    if (state_commit) {
      state_commit->graph.close();
      state_commit.reset();
    }
    while (!layers.empty()) {
      layers.back()->close();
      layers.pop_back();
    }
    if (vocabulary) {
      vocabulary->close();
      vocabulary.reset();
    }
    bank.close();
  }
};

Model::Model(Context &context, const WeightStore &weights, const Config &config,
             const inference::GenerationPolicy &policy,
             const std::map<unsigned, CompiledBlock> &compiled,
             const std::map<unsigned, CompiledBlock> &compiled_feed_forward,
             const ModelOptions &options, const std::function<void(const std::string &)> &phase) {
  const auto head_backend = options.head;
  const auto projection_backend = options.projections;
  const auto kv_write_mode = options.kv_write;
  const auto layout = options.layout;
  const bool combined_state_commit = options.state_commit != StateCommit::PerLayer;
  const bool commit_state_in_ffn = options.state_commit == StateCommit::FeedForward;
  if (options.state_commit != StateCommit::PerLayer &&
      options.state_commit != StateCommit::Combined &&
      options.state_commit != StateCommit::FeedForward) {
    throw std::invalid_argument("unsupported Qwen state commit policy");
  }
  config.validate();
  if (layout != DecoderLayout::Separate && layout != DecoderLayout::FusedAttentionNorm) {
    throw std::invalid_argument("unsupported Qwen decoder layout");
  }
  const auto &table = weights.find("model.embed_tokens.weight");
  if (config.hidden != 1024 || config.intermediate != 3584 || config.layer_types.size() != 24 ||
      policy.vocabulary != 248320 || policy.capacity != config.kv.capacity ||
      policy.bos >= policy.vocabulary || policy.eos >= policy.vocabulary ||
      table.spec.type != DataType::Float16 || table.spec.shape != Shape{1024, 248320}) {
    throw std::invalid_argument("complete Qwen model requires the fixed text deployment contract");
  }
  if (head_backend != LinearBackend::MatMul && head_backend != LinearBackend::FullyConnected) {
    throw std::invalid_argument("Qwen head requires one explicit projection implementation");
  }
  const bool complete_feed_forward = compiled_feed_forward.size() == 24;
  if (projection_backend != LinearBackend::MatMul &&
      (projection_backend != LinearBackend::SharedFullyConnected ||
       (!complete_feed_forward &&
        (compiled_feed_forward.size() != 23 || compiled_feed_forward.count(9))))) {
    throw std::invalid_argument(
        "Qwen shared FCL requires the fixed fallback or complete FFN package");
  }
  std::size_t constants = 0, ordinary_weights = table.bytes;
  unsigned common_batch = 0;
  for (const auto &entry : compiled_feed_forward) {
    if (entry.first >= 24) {
      throw std::invalid_argument("compiled Qwen feed-forward layer exceeds model bounds");
    }
  }
  for (unsigned layer = 0; layer < 24; ++layer) {
    validate_layer_weights(weights, config, layer, true);
    const auto feed_forward = compiled_feed_forward.find(layer);
    const bool handwritten = feed_forward == compiled_feed_forward.end();
    if (!handwritten) {
      validate_compiled_feed_forward(feed_forward->second, weights, config, layer);
      constants += feed_forward->second.graph.constant_bytes;
    }
    const auto batch = handwritten ? 1 : compiled_feed_forward_batch_tokens(feed_forward->second);
    if (common_batch && batch != common_batch) {
      throw std::invalid_argument("Qwen FFN package contains mixed batch widths");
    }
    common_batch = batch;
    if (config.mixer(layer) == MixerKind::DeltaNet) {
      const auto artifact = compiled.find(layer);
      if (artifact == compiled.end()) {
        throw std::invalid_argument("complete Qwen requires every compiled DeltaNet mixer");
      }
      validate_compiled_mixer(artifact->second, weights, config, layer);
      constants += artifact->second.graph.constant_bytes;
    }
    const auto prefix = config.layer_prefix(layer);
    for (const auto &record : weights.records()) {
      if (record.name.compare(0, prefix.size(), prefix) != 0) {
        continue;
      }
      const auto mixer = compiled.find(layer);
      if (mixer != compiled.end() && mixer->second.includes_input_norm &&
          record.name == prefix + "input_layernorm.weight") {
        continue;
      }
      if ((handwritten || (record.name.find(".mlp.") == std::string::npos &&
                           record.name != prefix + "post_attention_layernorm.weight")) &&
          record.name.find(".linear_attn.") == std::string::npos) {
        ordinary_weights += record.bytes;
      }
    }
  }
  if (common_batch > 1) {
    if (!complete_feed_forward || (common_batch != 8 && common_batch != 16) ||
        !combined_state_commit) {
      throw std::invalid_argument("Qwen block prefill requires M8/M16 and combined decode commits");
    }
  }
  if (layout != DecoderLayout::Separate &&
      (common_batch == 1 || projection_backend != LinearBackend::SharedFullyConnected ||
       kv_write_mode != KvWriteMode::MappedHost)) {
    throw std::invalid_argument("optimized Qwen decoder layout requires mapped batched FFN");
  }
  if (commit_state_in_ffn && (!combined_state_commit || common_batch == 1 ||
                              projection_backend != LinearBackend::SharedFullyConnected ||
                              kv_write_mode != KvWriteMode::MappedHost)) {
    throw std::invalid_argument("FFN state commit requires deferred mapped batched execution");
  }
  // The exact budget is charged during construction. Reject an impossible lower
  // bound before any tensor allocation; each bank remains independently capped.
  if (compiled.size() != 18 || constants > segment_payload_limit ||
      ordinary_weights +
              2 * 18 * (config.delta.recurrent().bytes() + config.delta.convolution().bytes()) >
          segment_payload_limit) {
    throw std::invalid_argument("complete Qwen exceeds its separate resident payload budgets");
  }
  try {
    impl_ = std::make_unique<Impl>(context, weights, config, policy, compiled,
                                   compiled_feed_forward, options, phase);
  } catch (const std::exception &error) {
    const auto &memory = context.memory();
    throw std::runtime_error(
        "Qwen initialization failed: const_peak_bytes=" + std::to_string(memory.peak(true)) +
        " nonconst_peak_bytes=" + std::to_string(memory.peak(false)) +
        " const_live_bytes=" + std::to_string(memory.live(true)) +
        " nonconst_live_bytes=" + std::to_string(memory.live(false)) + ": " + error.what());
  }
}

Model::~Model() = default;

inference::Generation Model::generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
                                      TensorReadback token_readback) {
  if (token_readback != TensorReadback::SdkCopy && token_readback != TensorReadback::MappedHost) {
    throw std::invalid_argument("unsupported Qwen token readback transport");
  }
  if (!impl_ || prompt.empty()) {
    throw std::invalid_argument(
        "Qwen generation requires a live model and a nonempty rendered prompt");
  }
  inference::GenerationExecutor executor{[&] { impl_->reset(); },
                                         [&](std::int32_t token) { impl_->consume(token); },
                                         [&] { return impl_->predict(token_readback); }};
  if (impl_->batch_tokens > 1) {
    executor.prefill = [&](const auto &tokens) { impl_->prefill(tokens); };
  }
  return inference::generate_tokens(impl_->policy, executor, prompt, maximum);
}

std::vector<std::uint8_t> Model::read(unsigned layer, const std::string &field) {
  if (!impl_) {
    throw std::logic_error("closed Qwen model");
  }
  impl_->require_live();
  return impl_->layers.at(layer)->read(field, TensorReadback::MappedHost);
}

std::size_t Model::kv_revalidations() const {
  std::size_t count = 0;
  if (impl_) {
    for (const auto &layer : impl_->layers) {
      count += layer->kv_revalidations();
    }
  }
  return count;
}

void Model::close() {
  if (impl_) {
    impl_->close();
    impl_.reset();
  }
}
} // namespace specferry::models::qwen3_5
