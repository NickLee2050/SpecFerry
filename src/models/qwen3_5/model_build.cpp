#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/memory_budget.hpp"
#include "models/qwen3_5/model_internal.hpp"
#include "vsi_nn_pub.h"

#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string_view>

namespace specferry::models::qwen3_5 {
Model::Impl::Impl(Context &context, const WeightStore &weights, const Config &spec,
                  const inference::GenerationPolicy &selection,
                  const std::map<unsigned, CompiledBlock> &compiled,
                  const std::map<unsigned, CompiledBlock> &compiled_feed_forward,
                  const ModelOptions &options,
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
          commit_state_in_ffn,
          options.delta_batch_tokens == 8 && !layers.empty()
              ? std::optional<TensorBinding>(layers.back()->block_output())
              : std::nullopt));
    }
    block_prefill = options.delta_batch_tokens == 8;
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
    const auto scale = normalization->weight(weights, record, {DataType::Float16, {config.hidden}});
    const auto normalized =
        normalization->rms_norm(hidden, scale, config.epsilon, 1.0f, DataType::Float16);
    normalization->compile({hidden}, {normalized});
    if (phase) {
      phase("lm-head");
    }
    head = std::make_unique<VocabularyHead>(
        context, *vocabulary, TensorBinding{normalization->graph, normalized.id}, SamplingOptions{},
        options.head, options.head_blocks_per_graph);
  } catch (...) {
    close();
    throw;
  }
}

Model::Model(Context &context, const WeightStore &weights, const Config &config,
             const inference::GenerationPolicy &policy,
             const std::map<unsigned, CompiledBlock> &compiled,
             const std::map<unsigned, CompiledBlock> &compiled_feed_forward,
             const ModelOptions &options, const std::function<void(const std::string &)> &phase,
             const std::map<unsigned, CompiledBlock> &compiled_prefill) {
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
  if ((options.delta_batch_tokens != 1 && options.delta_batch_tokens != 8) ||
      (options.delta_batch_tokens == 1 && !compiled_prefill.empty()) ||
      (options.delta_batch_tokens == 8 &&
       (compiled_prefill.size() != 18 || common_batch != 8 || !commit_state_in_ffn ||
        layout != DecoderLayout::FusedAttentionNorm))) {
    throw std::invalid_argument(
        "block DeltaNet requires all 18 M8 artifacts and the fused FFN-commit chain");
  }
  std::size_t prefill_constants = 0;
  for (const auto &entry : compiled_prefill) {
    validate_compiled_prefill(entry.second, weights, config, entry.first);
    prefill_constants += entry.second.graph.constant_bytes;
  }
  if (constants + prefill_constants > std::size_t{1408} * 1024 * 1024) {
    throw std::invalid_argument("Qwen block prefill exceeds the explicit constant profile");
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
    // Match the accepted allocation order: build M1/FFN/head first, then M8.
    // Each block borrows M1 state; its graph must be destroyed before that owner.
    for (const auto &entry : compiled_prefill) {
      if (phase) {
        phase("delta-prefill.layer." + std::to_string(entry.first));
      }
      impl_->layers.at(entry.first)->enable_block_prefill(context, entry.second);
    }
  } catch (const std::exception &error) {
    const auto &memory = context.memory();
    throw std::runtime_error(
        "Qwen initialization failed: const_peak_bytes=" + std::to_string(memory.peak(true)) +
        " nonconst_peak_bytes=" + std::to_string(memory.peak(false)) +
        " const_live_bytes=" + std::to_string(memory.live(true)) +
        " nonconst_live_bytes=" + std::to_string(memory.live(false)) + ": " + error.what());
  }
}

} // namespace specferry::models::qwen3_5
