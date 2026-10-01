#include "models/opt/layer.hpp"
#include "models/opt/partitioned_graph_model.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/tensor.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;

WeightStorage weight_storage(PartitionedGraphModel::Variant variant) {
  switch (variant) {
  case PartitionedGraphModel::Variant::MergedKv:
    return WeightStorage::SharedMutable;
  case PartitionedGraphModel::Variant::MergedKvLocalConst:
    return WeightStorage::LocalConst;
  case PartitionedGraphModel::Variant::MergedKvLocalMutable:
    return WeightStorage::LocalMutable;
  }
  throw std::invalid_argument("unknown partitioned OPT variant");
}

std::vector<std::uint8_t> integers(const std::vector<std::int32_t> &values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(std::int32_t));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
} // namespace

struct PartitionedGraphModel::Impl {
  struct LayerObservation {
    std::size_t graph_index;
    Tensor hidden;
    std::map<std::string, Tensor> stages;
  };

  struct ExecutionSet {
    explicit ExecutionSet(unsigned tokens)
        : block(tokens), timing_prefix("block" + std::to_string(tokens)) {}

    const unsigned block;
    const std::string timing_prefix;
    std::unique_ptr<GraphBuilder> embedding;
    Tensor controls{}, embedded{};
    std::vector<std::unique_ptr<GraphBuilder>> decoder_graphs;
    std::vector<std::string> decoder_roles;
    std::vector<LayerObservation> layers;
    std::unique_ptr<GraphBuilder> head;
    Tensor projected{}, selected{};
    std::vector<Tensor> logits;

    std::string prefix() const { return block == 1 ? "" : "prefill" + std::to_string(block) + "."; }

    void close() {
      if (head) {
        head->graph.close();
        head.reset();
      }
      layers.clear();
      while (!decoder_graphs.empty()) {
        decoder_graphs.back()->graph.close();
        decoder_graphs.pop_back();
      }
      if (embedding) {
        embedding->graph.close();
        embedding.reset();
      }
    }
  };

  ModelConfig config;
  unsigned decoder_group_size;
  Variant variant;
  PrefillHeadPolicy prefill_head_policy;
  unsigned prefill_block;
  WeightBank bank;
  Graph cache;
  std::vector<Tensor> cache_tensors;
  std::unique_ptr<ExecutionSet> decode, prefill;
  ExecutionSet *latest = nullptr;
  unsigned position = 0;
  std::size_t completed_steps = 0;
  std::size_t completed_decode_steps = 0;
  std::size_t completed_prefill_steps = 0;
  std::size_t completed_head_runs = 0;
  bool prediction_valid = false;
  bool failed = false;
  bool closed = false;

  Impl(Context &context, const WeightStore &weights, const ModelConfig &spec,
       const PhaseObserver &phase, unsigned group_size, Variant selected_variant,
       PrefillHeadPolicy prefill_head, unsigned block)
      : config(spec), decoder_group_size(group_size), variant(selected_variant),
        prefill_head_policy(prefill_head), prefill_block(block), bank(context),
        cache(context, 2 * spec.decoder.layers, 0) {
    try {
      build(context, weights, phase);
    } catch (...) {
      try {
        close();
      } catch (...) {
        // Keep the construction error; member destructors still release handles.
      }
      throw;
    }
  }

  ~Impl() {
    try {
      close();
    } catch (...) {
      // Explicit close reports SDK release errors. Destructors cannot report them.
    }
  }

  std::pair<Tensor, Tensor> update_merged_cache(GraphBuilder &builder, unsigned layer, Tensor key,
                                                Tensor value, Tensor slot_index, unsigned block) {
    const auto &decoder = config.decoder;
    const auto dimension = decoder.kv_spec().head_dim;
    const auto key_storage = cache_tensors.at(layer * 2);
    const auto value_storage = cache_tensors.at(layer * 2 + 1);
    const auto append = [&](Tensor column, Tensor storage) {
      auto bound = builder.bind({cache, storage.id}, storage.spec);
      if (block != 1) {
        // This writable alias is fixed before Verify and shares the original allocation.
        bound = storage_reshape(builder, bound, {decoder.hidden * block, decoder.capacity / block});
        column = builder.reshape(column, {decoder.hidden * block, 1});
      }
      return indexed_append(builder, column, slot_index, bound);
    };
    auto updated_key = append(key, key_storage);
    auto updated_value = append(value, value_storage);
    // The backing is contiguous [dimension, head, slot]. Explicit graph nodes
    // establish write-before-read dependencies, including after a block-alias write.
    // Do not replace these post-write nodes with storage-only aliases.
    const Shape stored_shape{dimension, decoder.heads, decoder.capacity};
    auto keys = builder.permute(builder.reshape(updated_key, stored_shape), {0, 2, 1});
    auto values = builder.permute(builder.reshape(updated_value, stored_shape), {0, 2, 1});
    return {keys, values};
  }

  LayerObservation build_decoder(GraphBuilder &builder, const WeightStore &weights, unsigned layer,
                                 Tensor hidden, Tensor shared_controls, std::size_t graph_index,
                                 unsigned block) {
    const auto &decoder = config.decoder;
    const auto dimension = decoder.kv_spec().head_dim;
    LayerObservation stage;
    stage.graph_index = graph_index;
    // Keep each layer's control operators unchanged when grouping graphs.
    auto slot_index = builder.slice(shared_controls, {2 * block}, {1});
    auto current_position = builder.slice(shared_controls, {block}, {block});
    auto limit = builder.binary(VSI_NN_OP_ADD, current_position, builder.integer(1));
    auto qkv = build_projections(builder, weights, decoder, layer, hidden);
    stage.stages = {{"query", qkv.query}, {"key", qkv.key}, {"value", qkv.value}};
    const auto [batched_keys, batched_values] =
        update_merged_cache(builder, layer, qkv.key, qkv.value, slot_index, block);
    stage.stages.emplace("keys", batched_keys);
    stage.stages.emplace("values", batched_values);
    auto query =
        block == 1 ? builder.reshape(qkv.query, {dimension, 1, decoder.heads})
                   : builder.permute(builder.reshape(qkv.query, {dimension, decoder.heads, block}),
                                     {0, 2, 1});
    auto attention = attention_core(builder, query, batched_keys, batched_values, limit, 1.0f);
    auto attended = attention.attended;
    if (block != 1) {
      attended =
          builder.permute(builder.reshape(attended, {dimension, block, decoder.heads}), {0, 2, 1});
    }
    attended = builder.reshape(attended, {decoder.hidden, block});
    stage.stages.emplace("attended", attended);
    auto tail = build_decoder_tail(builder, weights, decoder, layer, hidden, attended);
    stage.stages.insert(tail.begin(), tail.end());
    stage.hidden = tail.at("output");
    return stage;
  }

  void build(Context &context, const WeightStore &weights, const PhaseObserver &phase) {
    const auto &decoder = config.decoder;
    const auto cache_count = 2 * decoder.layers;
    const TensorSpec cache_spec{f16, {decoder.hidden, decoder.capacity}};
    const std::vector<std::uint8_t> zeros(cache_spec.bytes());
    for (unsigned index = 0; index < cache_count; ++index) {
      cache_tensors.push_back({add_tensor(cache, cache_spec, false, zeros), cache_spec});
    }

    decode = std::make_unique<ExecutionSet>(1);
    build_execution_set(*decode, context, weights, phase);
    if (prefill_block != 1) {
      prefill = std::make_unique<ExecutionSet>(prefill_block);
      build_execution_set(*prefill, context, weights, phase);
    }
    latest = decode.get();
  }

  void build_execution_set(ExecutionSet &execution, Context &context, const WeightStore &weights,
                           const PhaseObserver &phase) {
    const auto &decoder = config.decoder;
    const auto block = execution.block;
    const auto storage = weight_storage(variant);
    execution.embedding = std::make_unique<GraphBuilder>(context, 512, 256, &bank, storage);
    auto &entry = *execution.embedding;
    const auto controls = entry.tensor({DataType::Int32, {2 * block + 1}});
    execution.controls = controls;
    auto token = entry.slice(controls, {0}, {block});
    auto token_position = entry.slice(controls, {block}, {block});
    auto slot = entry.slice(controls, {2 * block}, {1});
    const auto &record = weights.find("decoder.embed_tokens.weight");
    std::vector<Tensor> table;
    for (unsigned first = 0; first < config.vocabulary; first += config.block_rows) {
      const auto rows = std::min(config.block_rows, config.vocabulary - first);
      table.push_back(entry.weight_chunk(weights, record, {f16, {config.embedding, rows}},
                                         std::size_t(first) * config.embedding * 2,
                                         WeightStorage::SharedMutable));
    }
    auto lookup = lookup_blocks(entry, table, token);
    auto input_projection =
        entry.project(lookup, weights, weights.find("decoder.project_in.weight"));
    const auto &positions = weights.find("decoder.embed_positions.weight");
    auto learned = entry.gather(
        entry.weight(weights, positions, positions.spec),
        entry.binary(VSI_NN_OP_ADD, token_position, entry.integer(config.position_offset)), 1);
    auto embedded = entry.binary(VSI_NN_OP_ADD, input_projection, learned);
    execution.embedded = embedded;

    if (phase) {
      phase("setup." + execution.prefix() + "embedding");
    }
    entry.compile({controls}, {embedded});

    GraphBuilder *producer = execution.embedding.get();
    Tensor previous = embedded;
    for (unsigned first = 0; first < decoder.layers; first += decoder_group_size) {
      const auto count = std::min(decoder_group_size, decoder.layers - first);
      auto owner =
          std::make_unique<GraphBuilder>(context, 512 * count, 256 * count, &bank, storage);
      auto &builder = *owner;
      const auto group_input = builder.share(*producer, previous);
      auto shared_controls = builder.share(entry, controls);
      auto hidden = group_input;
      for (unsigned layer = first; layer < first + count; ++layer) {
        auto stage = build_decoder(builder, weights, layer, hidden, shared_controls,
                                   execution.decoder_graphs.size(), block);
        // Inside a group the next decoder consumes the existing tensor directly.
        // Only the group's final hidden state is declared as a graph output.
        hidden = stage.hidden;
        execution.layers.push_back(std::move(stage));
      }
      if (phase) {
        phase("setup." + execution.prefix() +
              (count == 1
                   ? "layer." + std::to_string(first)
                   : "layers." + std::to_string(first) + "-" + std::to_string(first + count - 1)));
      }
      builder.compile({group_input, shared_controls}, {hidden});
      producer = owner.get();
      previous = hidden;
      execution.decoder_graphs.push_back(std::move(owner));
      execution.decoder_roles.push_back(
          execution.timing_prefix + ".decoder." + std::to_string(first) +
          (count == 1 ? "" : "-" + std::to_string(first + count - 1)));
    }

    execution.head = std::make_unique<GraphBuilder>(context, 512, 256, &bank, storage);
    auto &output = *execution.head;
    auto shared_hidden = output.share(*producer, previous);
    auto hidden = block == 1 ? shared_hidden
                             : output.slice(shared_hidden, {0, block - 1}, {decoder.hidden, 1});
    execution.projected =
        output.project(hidden, weights, weights.find("decoder.project_out.weight"));
    for (unsigned first = 0; first < config.vocabulary; first += config.block_rows) {
      const auto rows = std::min(config.block_rows, config.vocabulary - first);
      auto weight = output.weight_chunk(weights, record, {f16, {config.embedding, rows}},
                                        std::size_t(first) * config.embedding * 2,
                                        WeightStorage::SharedMutable);
      execution.logits.push_back(
          output.matmul(execution.projected, weight, {rows, 1}, false, true));
    }
    execution.selected = greedy_token(output, execution.logits, config.block_rows);
    if (phase) {
      phase("setup." + execution.prefix() + "head");
    }
    output.compile({shared_hidden}, {execution.selected});
  }

  void require_open() const {
    if (closed || failed) {
      throw std::logic_error("closed or failed partitioned OPT graph");
    }
  }

  void run(GraphBuilder &builder, const std::string &role) {
    const SdkScope timing_role("role", role);
    if (!vxIsGraphVerified(builder.graph.get()->g)) {
      throw std::runtime_error("partitioned OPT graph became unverified");
    }
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(builder.graph.get()); }),
          "run partitioned OPT graph");
  }

  void consume(ExecutionSet &execution, const std::vector<std::int32_t> &tokens, bool run_head) {
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
      upload_tensor(execution.embedding->graph, execution.controls.id, integers(controls));
    }
    run(*execution.embedding, execution.timing_prefix + ".embedding");
    for (std::size_t index = 0; index < execution.decoder_graphs.size(); ++index) {
      run(*execution.decoder_graphs[index], execution.decoder_roles[index]);
    }
    if (run_head) {
      run(*execution.head, execution.timing_prefix + ".head");
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

  void step(std::int32_t token) {
    require_open();
    consume(*decode, {token}, true);
  }

  void consume_prompt(const std::vector<std::int32_t> &tokens) {
    require_open();
    for (const auto chunk : inference::plan_prefill(tokens.size(), prefill_block)) {
      auto &execution = chunk.count == 1 ? *decode : *prefill;
      const bool run_head = prefill_head_policy == PrefillHeadPolicy::EveryToken ||
                            chunk.begin + chunk.count == tokens.size();
      const std::vector<std::int32_t> block_tokens(tokens.begin() + chunk.begin,
                                                   tokens.begin() + chunk.begin + chunk.count);
      consume(execution, block_tokens, run_head);
    }
  }

  void reset() {
    const SdkScope timing_role("role", "kv-reset");
    require_open();
    failed = true;
    prediction_valid = false;
    if (position) {
      for (const auto &tensor : cache_tensors) {
        upload_tensor(cache, tensor.id, std::vector<std::uint8_t>(tensor.spec.bytes()));
      }
    }
    position = 0;
    failed = false;
  }

  std::int32_t predict() {
    require_open();
    if (!position || !prediction_valid) {
      throw std::logic_error("no partitioned OPT prediction is available");
    }
    const SdkScope timing_role("role", latest->timing_prefix + ".selection");
    const auto bytes = read_tensor(latest->head->graph, latest->selected.id);
    if (bytes.size() != sizeof(std::int32_t)) {
      throw std::runtime_error("partitioned OPT prediction is not one INT32");
    }
    std::int32_t token;
    std::memcpy(&token, bytes.data(), sizeof(token));
    config.validate_token(token);
    return token;
  }

  void close() {
    if (closed) {
      return;
    }
    // A release error may leave only part of an execution set alive. Reject
    // inference and reads immediately; a later close() may still finish release.
    failed = true;
    prediction_valid = false;
    latest = nullptr;
    if (prefill) {
      prefill->close();
      prefill.reset();
    }
    if (decode) {
      decode->close();
      decode.reset();
    }
    cache.close();
    bank.close();
    closed = true;
  }
};

PartitionedGraphModel::PartitionedGraphModel(Context &context, const WeightStore &weights,
                                             const ModelConfig &config, PhaseObserver phase,
                                             unsigned decoder_group_size, Variant variant,
                                             PrefillHeadPolicy prefill_head,
                                             unsigned prefill_block) {
  if (decoder_group_size != 1 && decoder_group_size != 2) {
    throw std::invalid_argument("partitioned OPT decoder group size must be 1 or 2");
  }
  config.validate();
  weight_storage(variant);
  if (prefill_head != PrefillHeadPolicy::EveryToken &&
      prefill_head != PrefillHeadPolicy::LastToken) {
    throw std::invalid_argument("unknown partitioned OPT prefill head policy");
  }
  if (prefill_block != 1 && prefill_block != 4) {
    throw std::invalid_argument("partitioned OPT prefill block must be 1 or 4");
  }
  if (prefill_block == 4 &&
      (variant != Variant::MergedKv || decoder_group_size != 1 ||
       prefill_head != PrefillHeadPolicy::EveryToken || config.decoder.capacity % prefill_block)) {
    throw std::invalid_argument("block-4 prefill requires merged-kv, group 1, every-token heads "
                                "and capacity divisible by 4");
  }
  validate_model_weights(weights, config, config.decoder.layers);
  impl_ = std::make_unique<Impl>(context, weights, config, phase, decoder_group_size, variant,
                                 prefill_head, prefill_block);
}

PartitionedGraphModel::~PartitionedGraphModel() = default;

PartitionedGraphModel::Impl &PartitionedGraphModel::require_impl() const {
  if (!impl_) {
    throw std::logic_error("closed partitioned OPT graph");
  }
  return *impl_;
}

void PartitionedGraphModel::step(std::int32_t token) { require_impl().step(token); }

inference::Generation PartitionedGraphModel::generate(const std::vector<std::int32_t> &prompt,
                                                      unsigned maximum) {
  const auto &config = require_impl().config;
  inference::GenerationExecutor executor{[&] { impl_->reset(); },
                                         [&](std::int32_t token) { impl_->step(token); },
                                         [&] { return impl_->predict(); },
                                         {}};
  if (impl_->prefill_block != 1 || impl_->prefill_head_policy == PrefillHeadPolicy::LastToken) {
    executor.prefill = [&](const std::vector<std::int32_t> &tokens) {
      impl_->consume_prompt(tokens);
    };
  }
  return inference::generate_tokens(
      {config.vocabulary, config.decoder.capacity, config.bos, config.eos}, executor, prompt,
      maximum);
}

std::vector<std::uint8_t> PartitionedGraphModel::read_layer(unsigned layer,
                                                            const std::string &name) {
  require_impl().require_open();
  auto &stage = impl_->latest->layers.at(layer);
  auto &graph = impl_->latest->decoder_graphs.at(stage.graph_index)->graph;
  return read_tensor(graph, stage.stages.at(name).id);
}

std::vector<std::uint8_t> PartitionedGraphModel::read_cache(unsigned layer, bool values) {
  require_impl().require_open();
  const auto &decoder = impl_->config.decoder;
  if (layer >= decoder.layers) {
    throw std::out_of_range("partitioned OPT cache layer exceeds configuration");
  }
  const auto tensor = impl_->cache_tensors.at(layer * 2 + values);
  const auto packed = read_tensor(impl_->cache, tensor.id);
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

std::vector<std::uint8_t> PartitionedGraphModel::read_output(const std::string &name) {
  require_impl().require_open();
  if (!impl_->prediction_valid) {
    throw std::logic_error("no partitioned OPT prediction is available");
  }
  if (name == "projected") {
    return read_tensor(impl_->latest->head->graph, impl_->latest->projected.id);
  }
  if (name == "selected") {
    return read_tensor(impl_->latest->head->graph, impl_->latest->selected.id);
  }
  if (name != "logits") {
    throw std::invalid_argument("unknown partitioned OPT output");
  }
  std::vector<std::uint8_t> result;
  for (const auto &part : impl_->latest->logits) {
    const auto bytes = read_tensor(impl_->latest->head->graph, part.id);
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
  return require_impl().cache_tensors.size();
}

std::size_t PartitionedGraphModel::weight_bytes() const {
  if (!impl_) {
    return 0;
  }
  std::size_t bytes = impl_->bank.payload_bytes();
  for (const auto *execution : {impl_->decode.get(), impl_->prefill.get()}) {
    if (!execution) {
      continue;
    }
    bytes += execution->embedding->local_weight_bytes() + execution->head->local_weight_bytes();
    for (const auto &graph : execution->decoder_graphs) {
      bytes += graph->local_weight_bytes();
    }
  }
  return bytes;
}

PartitionedGraphModel::Variant PartitionedGraphModel::variant() const {
  return require_impl().variant;
}

void PartitionedGraphModel::close() {
  if (impl_) {
    impl_->close();
    impl_.reset();
  }
}
} // namespace specferry::models::opt
