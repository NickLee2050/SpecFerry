#include "devices/np101/api/diagnostics.hpp"
#include "models/qwen3_5/model.hpp"
#include "models/qwen3_5/model_internal.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <stdexcept>

namespace specferry::models::qwen3_5 {
Model::Impl::~Impl() {
  try {
    close();
  } catch (...) {
    // Explicit close reports release failures; destructors cannot.
  }
}

void Model::Impl::require_live() const {
  if (failed || !head) {
    throw std::logic_error("closed or failed Qwen model must be recreated");
  }
}

void Model::Impl::consume(std::int32_t token) {
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

void Model::Impl::prefill(const std::vector<std::int32_t> &tokens) {
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
        progress("token." + std::to_string(position) + ".prefill.decoder." + std::to_string(layer));
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

std::int32_t Model::Impl::predict(TensorReadback token_readback) {
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

void Model::Impl::reset() {
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

void Model::Impl::close() {
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

Model::~Model() = default;

inference::Generation Model::generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
                                      TensorReadback token_readback, HistoryMode history) {
  if (token_readback != TensorReadback::SdkCopy && token_readback != TensorReadback::MappedHost) {
    throw std::invalid_argument("unsupported Qwen token readback transport");
  }
  if (!impl_ || prompt.empty()) {
    throw std::invalid_argument(
        "Qwen generation requires a live model and a nonempty rendered prompt");
  }
  if (history != HistoryMode::Reset && history != HistoryMode::Retain) {
    throw std::invalid_argument("unsupported Qwen history mode");
  }
  if (history == HistoryMode::Reset) {
    impl_->history = {};
  } else if (!impl_->history.valid || impl_->history.consumed != impl_->position) {
    throw std::invalid_argument("Qwen history no longer matches model state");
  }
  inference::GenerationExecutor executor{[&] { impl_->reset(); },
                                         [&](std::int32_t token) { impl_->consume(token); },
                                         [&] { return impl_->predict(token_readback); }};
  if (impl_->batch_tokens > 1) {
    executor.prefill = [&](const auto &tokens) { impl_->prefill(tokens); };
  }
  // The model owns its history: callers cannot attach another model's cache position.
  // Retained prompts include the complete exact prefix, including the last prediction
  // that has not yet entered KV or the recurrent/convolution state.
  if (!maximum && history == HistoryMode::Reset) {
    impl_->history.valid = false;
    return inference::generate_tokens(impl_->policy, executor, prompt, maximum);
  }
  return inference::generate_tokens(impl_->policy, executor, prompt, maximum, {}, {},
                                    &impl_->history);
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
