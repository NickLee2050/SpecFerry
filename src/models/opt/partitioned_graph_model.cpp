#include "devices/np101/api/diagnostics.hpp"
#include "models/opt/partitioned_graph_internal.hpp"
#include "models/opt/partitioned_graph_model.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace specferry::models::opt {
namespace {
constexpr auto f16 = DataType::Float16;

std::vector<std::uint8_t> integers(const std::vector<std::int32_t> &values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(std::int32_t));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
} // namespace

PartitionedGraphModel::Impl::~Impl() {
  try {
    close();
  } catch (...) {
    // Explicit close reports SDK release errors. Destructors cannot report them.
  }
}

Graph &PartitionedGraphModel::Impl::cache_storage() {
  return kv_transfer == KvTransfer::MappedSlot ? decode->head->graph : *cache;
}

void PartitionedGraphModel::Impl::clear_mapped_cache() {
  for (const auto &tensor : cache_tensors) {
    upload_tensor(cache_storage(), tensor.id, cache_zeroes);
  }
}

void PartitionedGraphModel::Impl::commit_mapped_slot() {
  const SdkScope timing_role("role", "block1.kv-commit-mapped-slot");
  slot_start[1] = position;
  slot_end[1] = position + 1;
  auto &graph = cache_storage();
  auto source_start = slot_start;
  auto source_end = slot_end;
  if (compiled_full_model.current_cache_outputs()) {
    source_start[1] = 0;
    source_end[1] = 1;
  }
  // RunGraph has completed synchronously. Each map is unmapped before the
  // next operation; failure leaves consume() unusable with no position advance.
  for (std::size_t index = 0; index < cache_tensors.size(); ++index) {
    read_tensor_region_into(graph, mapped_cache_outputs.at(index).id, source_start, source_end,
                            cache_slot);
    upload_tensor_region(graph, cache_tensors[index].id, slot_start, slot_end, cache_slot);
  }
}

void PartitionedGraphModel::Impl::require_open() const {
  if (closed || failed) {
    throw std::logic_error("closed or failed partitioned OPT graph");
  }
}

void PartitionedGraphModel::Impl::run(GraphBuilder &builder, const std::string &role) {
  const SdkScope timing_role("role", role);
  if (!vxIsGraphVerified(builder.graph.get()->g)) {
    throw std::runtime_error("partitioned OPT graph became unverified");
  }
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(builder.graph.get()); }),
        "run partitioned OPT graph");
}

void PartitionedGraphModel::Impl::consume(ExecutionSet &execution,
                                          const std::vector<std::int32_t> &tokens, bool run_head) {
  const SdkScope timing_phase("phase", "token.position." + std::to_string(position));
  require_open();
  if (tokens.size() != execution.block || (execution.block != 1 && position % execution.block)) {
    throw std::invalid_argument("partitioned OPT append differs from its fixed block geometry");
  }
  if (tokens.size() > config.decoder.capacity - position) {
    throw std::out_of_range("partitioned OPT cache capacity exhausted");
  }
  for (const auto token : tokens) {
    config.validate_token(token);
  }
  std::vector<std::int32_t> controls;
  controls.reserve(2 * execution.block + 1);
  controls.insert(controls.end(), tokens.begin(), tokens.end());
  for (unsigned offset = 0; offset < execution.block; ++offset) {
    controls.push_back(std::int32_t(position + offset));
  }
  controls.push_back(std::int32_t(position / execution.block));
  failed = true;
  prediction_valid = false;
  {
    const SdkScope timing_role("role", execution.timing_prefix + ".controls");
    if (variant == Variant::CompiledFullModel) {
      upload_tensor(execution.head->graph, execution.controls.id, integers({tokens.front()}));
      upload_tensor(execution.head->graph, execution.full_position.id,
                    integers({std::int32_t(position)}));
    } else {
      upload_tensor(execution.embedding->graph, execution.controls.id, integers(controls));
    }
  }
  if (variant == Variant::CompiledDecoder || variant == Variant::CompiledFullModel) {
    const SdkScope timing_role("role", execution.timing_prefix + ".compiled-controls");
    std::vector<std::int32_t> indices;
    indices.reserve(2 * config.decoder.heads);
    for (unsigned head = 0; head < config.decoder.heads; ++head) {
      indices.push_back(std::int32_t(head));
      indices.push_back(std::int32_t(position));
    }
    std::vector<float> mask(config.decoder.capacity, -1e9f);
    std::fill_n(mask.begin(), position + 1, 0.0f);
    std::vector<std::uint8_t> mask_bytes(mask.size() * sizeof(float));
    std::memcpy(mask_bytes.data(), mask.data(), mask_bytes.size());
    upload_tensor(cache_storage(), compiled_indices.id, integers(indices));
    upload_tensor(cache_storage(), compiled_mask.id, mask_bytes);
  }
  if (variant == Variant::CompiledFullModel) {
    run(*execution.head, execution.timing_prefix + ".fullgraph");
  } else {
    run(*execution.embedding, execution.timing_prefix + ".embedding");
  }
  for (std::size_t index = 0; index < execution.decoder_graphs.size(); ++index) {
    run(*execution.decoder_graphs[index], execution.decoder_roles[index]);
  }
  if (execution.cache_commit) {
    run(*execution.cache_commit, execution.timing_prefix + ".kv-commit");
  } else if (kv_transfer == KvTransfer::MappedSlot) {
    commit_mapped_slot();
  }
  if (run_head || variant == Variant::CompiledFullModel) {
    if (variant != Variant::CompiledFullModel) {
      run(*execution.head, execution.timing_prefix + ".head");
    }
    auto &prediction = execution.prediction();
    if (execution.shared_head) {
      run(*prediction.head, execution.timing_prefix + ".shared-head");
    }
    if (prediction.selection) {
      run(*prediction.selection, execution.timing_prefix + ".selection");
    }
    ++completed_head_runs;
    prediction_valid = true;
  }
  position += execution.block;
  ++completed_steps;
  if (execution.block == 1) {
    ++completed_decode_steps;
  } else {
    ++completed_prefill_steps;
  }
  latest = &execution;
  failed = false;
}

void PartitionedGraphModel::Impl::step(std::int32_t token) {
  require_open();
  consume(*decode, {token}, true);
}

void PartitionedGraphModel::Impl::consume_prompt(const std::vector<std::int32_t> &tokens) {
  require_open();
  if (full_model_prefill) {
    if (position || tokens.empty()) {
      throw std::invalid_argument("full-model block prefill requires a fresh nonempty prompt");
    }
    // The embedded M1 head remains the sole prediction owner. Keep at least
    // one prompt token for it, including prompts divisible by the block size.
    const auto prefix = ((tokens.size() - 1) / prefill_block) * prefill_block;
    failed = true;
    for (std::size_t begin = 0; begin < prefix; begin += prefill_block) {
      full_model_prefill->consume({tokens.begin() + begin, tokens.begin() + begin + prefill_block});
      ++completed_steps;
      ++completed_prefill_steps;
    }
    position = prefix;
    failed = false;
    for (std::size_t index = prefix; index < tokens.size(); ++index) {
      consume(*decode, {tokens[index]}, index + 1 == tokens.size());
    }
    return;
  }
  for (const auto chunk : inference::plan_prefill(tokens.size(), prefill_block, position)) {
    auto &execution = chunk.count == 1 ? *decode : *prefill;
    const bool run_head = prefill_head_policy == PrefillHeadPolicy::EveryToken ||
                          chunk.begin + chunk.count == tokens.size();
    const std::vector<std::int32_t> block_tokens(tokens.begin() + chunk.begin,
                                                 tokens.begin() + chunk.begin + chunk.count);
    const auto started = std::chrono::steady_clock::now();
    consume(execution, block_tokens, run_head);
    if (!run_head && prefill_block > 1) {
      const auto seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      // Headless prompt steps are comparable. Conservative extrema avoid
      // treating head execution or a slow first single step as replay savings.
      if (chunk.count == 1) {
        single_prefill_seconds =
            single_prefill_seconds > 0 ? std::min(single_prefill_seconds, seconds) : seconds;
      } else {
        block_prefill_seconds = std::max(block_prefill_seconds, seconds);
      }
    }
  }
}

void PartitionedGraphModel::Impl::reset() {
  const SdkScope timing_role("role", "kv-reset");
  require_open();
  failed = true;
  prediction_valid = false;
  if (position) {
    if (full_model_prefill) {
      full_model_prefill->reset();
    }
    if (cache_reset) {
      // Synchronous completion precedes control upload and every next-request consumer.
      run(*cache_reset, "kv-reset");
    } else if (kv_transfer == KvTransfer::MappedSlot) {
      clear_mapped_cache();
    } else {
      for (const auto &tensor : cache_tensors) {
        upload_tensor(*cache, tensor.id, std::vector<std::uint8_t>(tensor.spec.bytes()));
      }
    }
  }
  position = 0;
  failed = false;
}

std::int32_t PartitionedGraphModel::Impl::predict() {
  require_open();
  if (!position || !prediction_valid) {
    throw std::logic_error("no partitioned OPT prediction is available");
  }
  const SdkScope timing_role("role", latest->timing_prefix + ".selection");
  auto &prediction = latest->prediction();
  // Reuse the validated mapped INT32 readback path; never fall back to SDK-copy on failure.
  const auto bytes =
      read_tensor(prediction.selection ? prediction.selection->graph : prediction.head->graph,
                  prediction.selected.id, TensorReadback::MappedHost);
  if (bytes.size() != sizeof(std::int32_t)) {
    throw std::runtime_error("partitioned OPT prediction is not one INT32");
  }
  std::int32_t token;
  std::memcpy(&token, bytes.data(), sizeof(token));
  config.validate_token(token);
  return token;
}

void PartitionedGraphModel::Impl::close() {
  if (closed) {
    return;
  }
  // A release error may leave only part of an execution set alive. Reject
  // inference and reads immediately; a later close() may still finish release.
  failed = true;
  prediction_valid = false;
  latest = nullptr;
  if (full_model_prefill) {
    full_model_prefill->close();
    full_model_prefill.reset();
  }
  if (cache_reset) {
    cache_reset->graph.close();
    cache_reset.reset();
  }
  if (prefill) {
    prefill->close();
    prefill.reset();
  }
  if (decode) {
    decode->close();
    decode.reset();
  }
  if (cache) {
    cache->close();
  }
  if (bank) {
    bank->close();
  }
  closed = true;
}

PartitionedGraphModel::~PartitionedGraphModel() = default;

PartitionedGraphModel::Impl &PartitionedGraphModel::require_impl() const {
  if (!impl_) {
    throw std::logic_error("closed partitioned OPT graph");
  }
  return *impl_;
}

void PartitionedGraphModel::step(std::int32_t token) {
  auto &impl = require_impl();
  impl.history.valid = false;
  impl.step(token);
}

inference::Generation
PartitionedGraphModel::generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
                                const inference::GenerationPhaseObserver &on_phase) {
  require_impl().history = {};
  // Keep the existing zero-output/reset contract outside continuation sessions.
  if (maximum) {
    return continue_generation(prompt, maximum, on_phase);
  }
  const auto &config = impl_->config;
  return inference::generate_tokens(
      {config.vocabulary, config.decoder.capacity, config.bos, config.eos},
      {[&] { impl_->reset(); }, [&](std::int32_t token) { impl_->step(token); },
       [&] { return impl_->predict(); }},
      prompt, maximum, {}, on_phase);
}

inference::Generation
PartitionedGraphModel::continue_generation(const std::vector<std::int32_t> &prompt,
                                           unsigned maximum,
                                           const inference::GenerationPhaseObserver &on_phase) {
  const auto &config = require_impl().config;
  if (impl_->kv_transfer == KvTransfer::MappedSlot && !impl_->history.tokens.empty()) {
    throw std::invalid_argument(
        "mapped-slot continuation has not been validated; reset the request");
  }
  inference::GenerationExecutor executor{[&] { impl_->reset(); },
                                         [&](std::int32_t token) { impl_->step(token); },
                                         [&] { return impl_->predict(); },
                                         {}};
  if (impl_->prefill_block != 1 || impl_->prefill_head_policy == PrefillHeadPolicy::LastToken) {
    executor.prefill = [&](const std::vector<std::int32_t> &tokens) {
      impl_->consume_prompt(tokens);
    };
  }
  if (impl_->prefill_block == 16 && impl_->variant == Variant::MergedKvCompiledFfn &&
      impl_->projections.compiled_query && impl_->projections.compiled_output) {
    executor.prepare_prefill = [&](unsigned prefix, unsigned appended) {
      impl_->require_open();
      if (impl_->position != prefix) {
        throw std::logic_error("OPT position differs from the validated generation history");
      }
      const auto replay = inference::prefill_replay_tokens(prefix, appended, impl_->prefill_block,
                                                           impl_->single_prefill_seconds,
                                                           impl_->block_prefill_seconds);
      if (replay) {
        // Keep the buffers; causal masks hide old future slots while known IDs
        // overwrite this partial block. The generation history is already invalid.
        impl_->position -= replay;
        impl_->prediction_valid = false;
      }
      return replay;
    };
  }
  return inference::generate_tokens(
      {config.vocabulary, config.decoder.capacity, config.bos, config.eos}, executor, prompt,
      maximum, {}, on_phase, &impl_->history);
}

std::vector<std::uint8_t> PartitionedGraphModel::read_cache(unsigned layer, bool values,
                                                            np101::TensorReadback method) {
  require_impl().require_open();
  const auto &decoder = impl_->config.decoder;
  if (layer >= decoder.layers) {
    throw std::out_of_range("partitioned OPT cache layer exceeds configuration");
  }
  const auto tensor = impl_->cache_tensors.at(layer * 2 + values);
  const auto packed = read_tensor(impl_->cache_storage(), tensor.id, method);
  if (impl_->variant == Variant::CompiledDecoder || impl_->variant == Variant::CompiledFullModel) {
    if (packed.size() != cache_spec().bytes()) {
      throw std::runtime_error("compiled OPT cache readback has an unexpected size");
    }
    return packed;
  }
  const auto row_bytes = std::size_t(decoder.kv_spec().head_dim) * dtype_bytes(f16);
  const auto slot_bytes = row_bytes * decoder.heads;
  if (packed.size() != slot_bytes * decoder.capacity) {
    throw std::runtime_error("merged OPT cache readback has an unexpected size");
  }
  // Diagnostic-only host rearrangement keeps every existing CPU reference in
  // [head, slot, dimension] order while the actual backing stores [slot, head, dimension].
  std::vector<std::uint8_t> result(packed.size());
  for (unsigned head = 0; head < decoder.heads; ++head) {
    for (unsigned slot = 0; slot < decoder.capacity; ++slot) {
      const auto source = std::size_t(slot) * slot_bytes + head * row_bytes;
      const auto destination = (std::size_t(head) * decoder.capacity + slot) * row_bytes;
      std::memcpy(result.data() + destination, packed.data() + source, row_bytes);
    }
  }
  return result;
}

TensorSpec PartitionedGraphModel::cache_spec() const {
  const auto &impl = require_impl();
  const auto &decoder = impl.config.decoder;
  const auto type = impl.variant == Variant::CompiledFullModel
                        ? impl.compiled_full_model.cache_type()
                    : impl.variant == Variant::CompiledDecoder ? DataType::Float32
                                                               : f16;
  return {type, {decoder.kv_spec().head_dim, decoder.capacity, decoder.heads}};
}

std::vector<std::uint8_t> PartitionedGraphModel::read_output(const std::string &name,
                                                             np101::TensorReadback method) {
  require_impl().require_open();
  if (!impl_->prediction_valid) {
    throw std::logic_error("no partitioned OPT prediction is available");
  }
  auto &prediction = impl_->latest->prediction();
  if (name == "projected") {
    if (prediction.selection || impl_->variant == Variant::CompiledFullModel) {
      throw std::invalid_argument("compiled head does not expose its intermediate projection");
    }
    return read_tensor(prediction.head->graph, prediction.projected.id, method);
  }
  if (name == "selected") {
    return read_tensor(prediction.selection ? prediction.selection->graph : prediction.head->graph,
                       prediction.selected.id, method);
  }
  if (name != "logits") {
    throw std::invalid_argument("unknown partitioned OPT output");
  }
  std::vector<std::uint8_t> result;
  for (const auto &part : prediction.logits) {
    const auto bytes = read_tensor(prediction.head->graph, part.id, method);
    result.insert(result.end(), bytes.begin(), bytes.end());
  }
  return result;
}

std::size_t PartitionedGraphModel::launches() const { return impl_ ? impl_->completed_steps : 0; }

std::size_t PartitionedGraphModel::decode_steps() const {
  return impl_ ? impl_->completed_decode_steps : 0;
}

std::size_t PartitionedGraphModel::prefill_steps() const {
  return impl_ ? impl_->completed_prefill_steps : 0;
}

std::size_t PartitionedGraphModel::head_runs() const {
  return impl_ ? impl_->completed_head_runs : 0;
}

std::size_t PartitionedGraphModel::cache_reset_uploads_per_request() const {
  const auto &impl = require_impl();
  return impl.cache_reset ? 0 : impl.cache_tensors.size();
}

bool PartitionedGraphModel::shares_prefill_cache() const {
  const auto &impl = require_impl();
  return static_cast<bool>(impl.full_model_prefill);
}

std::size_t PartitionedGraphModel::weight_bytes() const {
  return impl_ ? (impl_->bank ? impl_->bank->payload_bytes() : 0) + impl_->constant_weight_bytes +
                     (impl_->full_model_prefill ? impl_->full_model_prefill->weight_bytes() : 0)
               : 0;
}

void PartitionedGraphModel::close() {
  if (impl_) {
    impl_->close();
    impl_.reset();
  }
}

} // namespace specferry::models::opt
