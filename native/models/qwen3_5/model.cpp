#include "models/qwen3_5/decoder.hpp"
#include "models/qwen3_5/model.hpp"
#include "np101/diagnostics.hpp"
#include "np101/memory_budget.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_pub.h"

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
  std::unique_ptr<GraphBuilder> normalization;
  std::unique_ptr<VocabularyHead> head;
  unsigned position = 0;
  bool failed = false, prediction_ready = false;
  std::function<void(const std::string &)> progress;
  bool trace_steps = false;

  Impl(Context &context, const WeightStore &weights, const Config &spec,
       const inference::GenerationPolicy &selection,
       const std::map<unsigned, CompiledBlock> &compiled,
       const std::map<unsigned, CompiledBlock> &compiled_feed_forward,
       const std::function<void(const std::string &)> &phase, VocabularyLookup lookup)
      : config(spec), policy(selection), bank(context), progress(phase) {
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
            context, weights, config, layer, input, &bank, LinearBackend::MatMul,
            artifact == compiled.end() ? nullptr : &artifact->second,
            feed_forward == compiled_feed_forward.end() ? nullptr : &feed_forward->second));
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
      head = std::make_unique<VocabularyHead>(context, *vocabulary,
                                              TensorBinding{normalization->graph, normalized.id});
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
    ++position;
    prediction_ready = true;
    failed = false;
  }

  std::int32_t predict() {
    require_live();
    if (!prediction_ready) {
      throw std::logic_error("Qwen has no completed decoder output");
    }
    failed = true;
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(normalization->graph.get()); }),
          "Qwen final normalization");
    const auto token = head->select();
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
      head->graph.close();
      head.reset();
    }
    if (normalization) {
      normalization->graph.close();
      normalization.reset();
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
             const std::function<void(const std::string &)> &phase, VocabularyLookup lookup) {
  config.validate();
  const auto &table = weights.find("model.embed_tokens.weight");
  if (config.hidden != 1024 || config.intermediate != 3584 || config.layer_types.size() != 24 ||
      policy.vocabulary != 248320 || policy.capacity != config.kv.capacity ||
      policy.bos >= policy.vocabulary || policy.eos >= policy.vocabulary ||
      table.spec.type != DataType::Float16 || table.spec.shape != Shape{1024, 248320}) {
    throw std::invalid_argument("complete Qwen model requires the fixed text deployment contract");
  }
  std::size_t constants = 0, ordinary_weights = table.bytes;
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
      if ((handwritten || (record.name.find(".mlp.") == std::string::npos &&
                           record.name != prefix + "post_attention_layernorm.weight")) &&
          record.name.find(".linear_attn.") == std::string::npos) {
        ordinary_weights += record.bytes;
      }
    }
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
                                   compiled_feed_forward, phase, lookup);
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

inference::Generation Model::generate(const std::vector<std::int32_t> &prompt, unsigned maximum) {
  if (!impl_ || prompt.empty()) {
    throw std::invalid_argument(
        "Qwen generation requires a live model and a nonempty rendered prompt");
  }
  return inference::generate_tokens(impl_->policy,
                                    {[&] { impl_->reset(); },
                                     [&](std::int32_t token) { impl_->consume(token); },
                                     [&] { return impl_->predict(); }},
                                    prompt, maximum);
}

std::vector<std::uint8_t> Model::read(unsigned layer, const std::string &field) {
  if (!impl_) {
    throw std::logic_error("closed Qwen model");
  }
  impl_->require_live();
  return impl_->layers.at(layer)->read(field, TensorReadback::MappedHost);
}

void Model::close() {
  if (impl_) {
    impl_->close();
    impl_.reset();
  }
}
} // namespace specferry::models::qwen3_5
