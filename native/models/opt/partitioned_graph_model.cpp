#include "models/opt/partitioned_graph_model.hpp"
#include "np101/diagnostics.hpp"
#include "np101/memory_budget.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;

std::vector<std::uint8_t> integers(const std::vector<std::int32_t> &values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(std::int32_t));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
} // namespace

struct PartitionedGraphModel::Impl {
  struct ExecutionSet {
    explicit ExecutionSet(unsigned tokens)
        : block(tokens), timing_prefix("block" + std::to_string(tokens)) {}

    const unsigned block;
    const std::string timing_prefix;
    std::unique_ptr<GraphBuilder> embedding;
    Tensor controls{}, embedded{};
    std::vector<std::unique_ptr<GraphBuilder>> decoder_graphs;
    std::vector<std::string> decoder_roles;
    std::unique_ptr<GraphBuilder> cache_commit;
    std::unique_ptr<GraphBuilder> head;
    Tensor projected{}, selected{};
    std::vector<Tensor> logits;

    std::string prefix() const { return block == 1 ? "" : "prefill" + std::to_string(block) + "."; }

    void close() {
      if (cache_commit) {
        cache_commit->graph.close();
        cache_commit.reset();
      }
      if (head) {
        head->graph.close();
        head.reset();
      }
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
  Variant variant;
  PrefillHeadPolicy prefill_head_policy;
  unsigned prefill_block;
  const std::vector<CompiledFfn> compiled_ffn;
  const std::vector<CompiledDecoder> compiled_decoders;
  const std::vector<CompiledFfn> compiled_prefill_ffn;
  Tensor compiled_indices{}, compiled_mask{};
  WeightBank bank;
  Graph cache;
  std::vector<Tensor> cache_tensors;
  std::unique_ptr<GraphBuilder> cache_reset;
  std::unique_ptr<ExecutionSet> decode, prefill;
  ExecutionSet *latest = nullptr;
  unsigned position = 0;
  std::size_t completed_steps = 0;
  std::size_t completed_decode_steps = 0;
  std::size_t completed_prefill_steps = 0;
  std::size_t completed_head_runs = 0;
  std::size_t constant_weight_bytes = 0;
  bool prediction_valid = false;
  bool failed = false;
  bool closed = false;

  Impl(Context &context, const WeightStore &weights, const ModelConfig &spec,
       const PhaseObserver &phase, Variant selected_variant, PrefillHeadPolicy prefill_head,
       unsigned block, std::vector<CompiledFfn> compiled, std::vector<CompiledDecoder> decoders,
       std::vector<CompiledFfn> prefill_artifacts)
      : config(spec), variant(selected_variant), prefill_head_policy(prefill_head),
        prefill_block(block), compiled_ffn(std::move(compiled)),
        compiled_decoders(std::move(decoders)), compiled_prefill_ffn(std::move(prefill_artifacts)),
        bank(context), cache(context, 2 * spec.decoder.layers + 2, 0) {
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

  Tensor build_attention(GraphBuilder &builder, const WeightStore &weights, unsigned layer,
                         Tensor hidden, Tensor shared_controls, unsigned block) {
    const auto &decoder = config.decoder;
    const auto dimension = decoder.kv_spec().head_dim;
    auto slot_index = builder.slice(shared_controls, {2 * block}, {1});
    auto current_position = builder.slice(shared_controls, {block}, {block});
    auto limit = builder.binary(VSI_NN_OP_ADD, current_position, builder.integer(1));
    auto qkv = build_projections(builder, weights, decoder, layer, hidden);
    const auto [batched_keys, batched_values] =
        update_merged_cache(builder, layer, qkv.key, qkv.value, slot_index, block);
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
    return builder.reshape(attended, {decoder.hidden, block});
  }

  Tensor build_decoder(GraphBuilder &builder, const WeightStore &weights, unsigned layer,
                       Tensor hidden, Tensor shared_controls, unsigned block) {
    const auto &decoder = config.decoder;
    const auto attended = build_attention(builder, weights, layer, hidden, shared_controls, block);
    const auto tail = build_decoder_tail(builder, weights, decoder, layer, hidden, attended,
                                         variant == Variant::MergedKvFc1Split2);
    return tail.at("output");
  }

  void build(Context &context, const WeightStore &weights, const PhaseObserver &phase) {
    const auto &decoder = config.decoder;
    if (variant == Variant::MergedKvFc1Split2) {
      // Load the large fc2 tensors before decoder graphs occupy SDK resources.
      // Subsequent bindings reuse these same chunks and the same memory budget.
      for (unsigned layer = 0; layer < decoder.layers; ++layer) {
        const auto &matrix = weights.find(decoder.prefix(layer) + "fc2.weight");
        bank.preload(weights, matrix, matrix.spec);
      }
    }
    const auto cache_count = 2 * decoder.layers;
    const bool compiled_decoder = variant == Variant::CompiledDecoder;
    const TensorSpec cache_spec =
        compiled_decoder ? TensorSpec{DataType::Float32,
                                      {decoder.kv_spec().head_dim, decoder.capacity, decoder.heads}}
                         : TensorSpec{f16, {decoder.hidden, decoder.capacity}};
    if (compiled_decoder) {
      const auto ports = compiled_decoder_inputs(decoder);
      compiled_indices = {add_tensor(cache, ports[3]), ports[3]};
      compiled_mask = {add_tensor(cache, ports[4]), ports[4]};
    }
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
    if (!compiled_prefill_ffn.empty()) {
      build_cache_reset(context, phase);
    }
    latest = decode.get();
  }

  void build_cache_reset(Context &context, const PhaseObserver &phase) {
    const auto count = static_cast<unsigned>(cache_tensors.size());
    cache_reset = std::make_unique<GraphBuilder>(context, count + 1, count);
    auto &builder = *cache_reset;
    const auto &spec = cache_tensors.front().spec;
    // Initialize one runtime input once, keeping the reset copies out of constant folding.
    // Reuse the validated KV commit operation; all destinations retain fixed backing.
    const Tensor zeros{
        add_tensor(builder.graph, spec, false, std::vector<std::uint8_t>(spec.bytes())), spec};
    std::vector<Tensor> outputs;
    for (const auto &storage : cache_tensors) {
      const auto destination = builder.bind({cache, storage.id}, storage.spec);
      builder.node(VSI_NN_OP_DATACONVERT, {zeros}, destination);
      outputs.push_back(destination);
    }
    if (phase) {
      phase("setup.kv-reset");
    }
    builder.compile({zeros}, outputs);
  }

  void build_execution_set(ExecutionSet &execution, Context &context, const WeightStore &weights,
                           const PhaseObserver &phase) {
    const auto &decoder = config.decoder;
    const auto block = execution.block;
    execution.embedding = std::make_unique<GraphBuilder>(context, 512, 256, &bank);
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
                                         std::size_t(first) * config.embedding * 2));
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

    std::vector<Tensor> commit_inputs, commit_outputs;
    if (variant == Variant::CompiledDecoder) {
      execution.cache_commit =
          std::make_unique<GraphBuilder>(context, 4 * decoder.layers, 2 * decoder.layers);
    }
    GraphBuilder *producer = execution.embedding.get();
    Tensor previous = embedded;
    for (unsigned layer = 0; layer < decoder.layers; ++layer) {
      auto owner =
          std::make_unique<GraphBuilder>(context, variant == Variant::CompiledDecoder ? 16 : 512,
                                         variant == Variant::CompiledDecoder ? 1 : 256, &bank);
      auto &builder = *owner;
      const auto group_input = builder.share(*producer, previous);

      const auto stage = "layer." + std::to_string(layer);
      const auto compile = [&](GraphBuilder &graph, const std::string &role,
                               const std::vector<Tensor> &inputs,
                               const std::vector<Tensor> &outputs) {
        if (phase) {
          phase("setup." + execution.prefix() + role);
        }
        graph.compile(inputs, outputs);
      };
      const auto keep = [&](std::unique_ptr<GraphBuilder> graph, const std::string &role) {
        execution.decoder_graphs.push_back(std::move(graph));
        execution.decoder_roles.push_back(execution.timing_prefix + ".decoder." + role);
      };
      if (variant == Variant::CompiledDecoder) {
        const auto &compiled = compiled_decoders.at(layer).graph;
        std::vector<Tensor> inputs{group_input};
        for (unsigned index : {0u, 1u}) {
          const auto storage = cache_tensors.at(2 * layer + index);
          inputs.push_back(builder.bind({cache, storage.id}, storage.spec));
        }
        inputs.push_back(builder.bind({cache, compiled_indices.id}, compiled_indices.spec));
        inputs.push_back(builder.bind({cache, compiled_mask.id}, compiled_mask.spec));
        const auto outputs = builder.compiled(compiled, inputs);
        compile(builder, stage, inputs, outputs);
        // Separate copy graph avoids unvalidated NBG input/output aliasing or mixed-node graphs.
        // It runs after all decoders, before any next-token consumer can read the new state.
        auto &commit = *execution.cache_commit;
        for (unsigned index : {1u, 2u}) {
          auto source = commit.share(builder, outputs[index]);
          auto destination = commit.share(builder, inputs[index]);
          commit.node(VSI_NN_OP_DATACONVERT, {source}, destination);
          commit_inputs.push_back(source);
          commit_outputs.push_back(destination);
        }
        producer = owner.get();
        previous = outputs[0];
        constant_weight_bytes += compiled.constant_bytes;
        keep(std::move(owner), std::to_string(layer) + ".compiled");
        continue;
      }
      const auto shared_controls = builder.share(entry, controls);
      if (variant == Variant::MergedKvCompiledFfn) {
        const auto &compiled = (block == 1 ? compiled_ffn : compiled_prefill_ffn).at(layer).graph;
        const auto version = compiled.graph_version;
        check(vsi_nn_SetGraphVersion(builder.graph.get(), version[0], version[1], version[2]),
              "set compiled OPT attention graph version");
        const auto attended =
            build_attention(builder, weights, layer, group_input, shared_controls, block);
        const auto normalized =
            build_attention_tail(builder, weights, decoder, layer, group_input, attended)
                .normalized;
        compile(builder, stage + ".attention", {group_input, shared_controls}, {normalized});
        auto *attention_graph = owner.get();
        keep(std::move(owner), std::to_string(layer) + ".attention");

        // v105 rejected NBG mixed with ordinary decoder nodes. Separate graphs
        // retain fixed device buffers; no FFN weights enter the shared bank.
        auto ffn = std::make_unique<GraphBuilder>(context, 16, 8);
        const auto ffn_input = ffn->share(*attention_graph, normalized);
        const auto mlp = ffn->compiled(compiled, {ffn_input}).at(0);
        compile(*ffn, stage + ".ffn", {ffn_input}, {mlp});
        auto *ffn_graph = ffn.get();
        keep(std::move(ffn), std::to_string(layer) + ".ffn");

        auto finish = std::make_unique<GraphBuilder>(context, 128, 64, &bank);
        check(vsi_nn_SetGraphVersion(finish->graph.get(), version[0], version[1], version[2]),
              "set compiled OPT residual graph version");
        const auto residual = finish->share(*attention_graph, normalized);
        const auto ffn_output = finish->share(*ffn_graph, mlp);
        const auto output =
            build_ffn_finish(*finish, weights, decoder, layer, residual, ffn_output).output;
        compile(*finish, stage + ".residual", {residual, ffn_output}, {output});
        producer = finish.get();
        previous = output;
        keep(std::move(finish), std::to_string(layer) + ".residual");
        constant_weight_bytes += compiled.constant_bytes;
        continue;
      }
      auto hidden = build_decoder(builder, weights, layer, group_input, shared_controls, block);
      compile(builder, stage, {group_input, shared_controls}, {hidden});
      producer = owner.get();
      previous = hidden;
      keep(std::move(owner), std::to_string(layer));
    }

    if (execution.cache_commit) {
      if (phase) {
        phase("setup.kv-commit");
      }
      execution.cache_commit->compile(commit_inputs, commit_outputs);
    }
    execution.head = std::make_unique<GraphBuilder>(context, 512, 256, &bank);
    auto &output = *execution.head;
    auto shared_hidden = output.share(*producer, previous);
    auto hidden = block == 1 ? shared_hidden
                             : output.slice(shared_hidden, {0, block - 1}, {decoder.hidden, 1});
    execution.projected =
        output.project(hidden, weights, weights.find("decoder.project_out.weight"));
    for (unsigned first = 0; first < config.vocabulary; first += config.block_rows) {
      const auto rows = std::min(config.block_rows, config.vocabulary - first);
      auto weight = output.weight_chunk(weights, record, {f16, {config.embedding, rows}},
                                        std::size_t(first) * config.embedding * 2);
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
    if (variant == Variant::CompiledDecoder) {
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
      upload_tensor(cache, compiled_indices.id, integers(indices));
      upload_tensor(cache, compiled_mask.id, mask_bytes);
    }
    run(*execution.embedding, execution.timing_prefix + ".embedding");
    for (std::size_t index = 0; index < execution.decoder_graphs.size(); ++index) {
      run(*execution.decoder_graphs[index], execution.decoder_roles[index]);
    }
    if (execution.cache_commit) {
      run(*execution.cache_commit, execution.timing_prefix + ".kv-commit");
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
      if (cache_reset) {
        // Synchronous completion precedes control upload and every next-request consumer.
        run(*cache_reset, "kv-reset");
      } else {
        for (const auto &tensor : cache_tensors) {
          upload_tensor(cache, tensor.id, std::vector<std::uint8_t>(tensor.spec.bytes()));
        }
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
    cache.close();
    bank.close();
    closed = true;
  }
};

PartitionedGraphModel::PartitionedGraphModel(Context &context, const WeightStore &weights,
                                             const ModelConfig &config, PhaseObserver phase,
                                             Variant variant, PrefillHeadPolicy prefill_head,
                                             unsigned prefill_block,
                                             std::vector<CompiledFfn> compiled_ffn,
                                             std::vector<CompiledDecoder> compiled_decoders,
                                             std::vector<CompiledFfn> compiled_prefill_ffn) {
  config.validate();
  if (variant != Variant::MergedKv && variant != Variant::MergedKvFc1Split2 &&
      variant != Variant::MergedKvCompiledFfn && variant != Variant::CompiledDecoder) {
    throw std::invalid_argument("unknown partitioned OPT variant");
  }
  if (prefill_head != PrefillHeadPolicy::EveryToken &&
      prefill_head != PrefillHeadPolicy::LastToken) {
    throw std::invalid_argument("unknown partitioned OPT prefill head policy");
  }
  if (prefill_block != 1 && prefill_block != 4 && prefill_block != 8 && prefill_block != 16) {
    throw std::invalid_argument("partitioned OPT prefill block must be 1, 4, 8 or 16");
  }
  if (variant != Variant::MergedKv && variant != Variant::MergedKvFc1Split2 &&
      variant != Variant::MergedKvCompiledFfn && prefill_block != 1) {
    throw std::invalid_argument("compiled decoder has no validated block-prefill weight sharing");
  }
  if (variant == Variant::MergedKvFc1Split2 && config.decoder.intermediate % 2) {
    throw std::invalid_argument("split-fc1 OPT requires an even intermediate width");
  }
  if (config.decoder.capacity % prefill_block) {
    throw std::invalid_argument("capacity must be divisible by the prefill block");
  }
  validate_model_weights(weights, config, config.decoder.layers);
  if (variant == Variant::MergedKvCompiledFfn) {
    if ((prefill_block != 1 && prefill_block != 16) ||
        (prefill_block == 1 && !compiled_prefill_ffn.empty())) {
      throw std::invalid_argument("compiled OPT prefill requires an explicit block16 artifact set");
    }
    std::size_t constants = 0;
    const auto validate_set = [&](const std::vector<CompiledFfn> &artifacts, unsigned tokens) {
      if (artifacts.size() != config.decoder.layers) {
        throw std::invalid_argument("compiled OPT FFN requires one artifact per decoder layer");
      }
      for (unsigned layer = 0; layer < config.decoder.layers; ++layer) {
        validate_compiled_ffn(artifacts[layer], weights, config.decoder, layer, tokens);
        const auto bytes = artifacts[layer].graph.constant_bytes;
        if (bytes > segment_payload_limit - constants) {
          throw std::invalid_argument("compiled OPT FFN sets exceed the constant payload limit");
        }
        constants += bytes;
      }
    };
    validate_set(compiled_ffn, 1);
    if (prefill_block != 1) {
      validate_set(compiled_prefill_ffn, prefill_block);
    }
  } else if (!compiled_ffn.empty() || !compiled_prefill_ffn.empty()) {
    throw std::invalid_argument("compiled OPT FFN artifacts require the compiled variant");
  }
  if (variant == Variant::CompiledDecoder) {
    if (compiled_decoders.size() != config.decoder.layers) {
      throw std::invalid_argument("compiled OPT decoder requires one artifact per layer");
    }
    std::size_t constants = 0;
    for (unsigned layer = 0; layer < config.decoder.layers; ++layer) {
      validate_compiled_decoder(compiled_decoders[layer], weights, config.decoder, layer);
      const auto bytes = compiled_decoders[layer].graph.constant_bytes;
      if (bytes > segment_payload_limit - constants) {
        throw std::invalid_argument("compiled OPT decoders exceed the constant payload limit");
      }
      constants += bytes;
    }
  } else if (!compiled_decoders.empty()) {
    throw std::invalid_argument("compiled decoder artifacts require the compiled decoder variant");
  }
  impl_ = std::make_unique<Impl>(context, weights, config, phase, variant, prefill_head,
                                 prefill_block, std::move(compiled_ffn),
                                 std::move(compiled_decoders), std::move(compiled_prefill_ffn));
}

PartitionedGraphModel::~PartitionedGraphModel() = default;

PartitionedGraphModel::Impl &PartitionedGraphModel::require_impl() const {
  if (!impl_) {
    throw std::logic_error("closed partitioned OPT graph");
  }
  return *impl_;
}

void PartitionedGraphModel::step(std::int32_t token) { require_impl().step(token); }

inference::Generation
PartitionedGraphModel::generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
                                const inference::GenerationPhaseObserver &on_phase) {
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
      maximum, {}, on_phase);
}

std::vector<std::uint8_t> PartitionedGraphModel::read_cache(unsigned layer, bool values,
                                                            np101::TensorReadback method) {
  require_impl().require_open();
  const auto &decoder = impl_->config.decoder;
  if (layer >= decoder.layers) {
    throw std::out_of_range("partitioned OPT cache layer exceeds configuration");
  }
  const auto tensor = impl_->cache_tensors.at(layer * 2 + values);
  const auto packed = read_tensor(impl_->cache, tensor.id, method);
  if (impl_->variant == Variant::CompiledDecoder) {
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
  return {impl.variant == Variant::CompiledDecoder ? DataType::Float32 : f16,
          {decoder.kv_spec().head_dim, decoder.capacity, decoder.heads}};
}

std::vector<std::uint8_t> PartitionedGraphModel::read_output(const std::string &name,
                                                             np101::TensorReadback method) {
  require_impl().require_open();
  if (!impl_->prediction_valid) {
    throw std::logic_error("no partitioned OPT prediction is available");
  }
  if (name == "projected") {
    return read_tensor(impl_->latest->head->graph, impl_->latest->projected.id, method);
  }
  if (name == "selected") {
    return read_tensor(impl_->latest->head->graph, impl_->latest->selected.id, method);
  }
  if (name != "logits") {
    throw std::invalid_argument("unknown partitioned OPT output");
  }
  std::vector<std::uint8_t> result;
  for (const auto &part : impl_->latest->logits) {
    const auto bytes = read_tensor(impl_->latest->head->graph, part.id, method);
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

std::size_t PartitionedGraphModel::weight_bytes() const {
  return impl_ ? impl_->bank.payload_bytes() + impl_->constant_weight_bytes : 0;
}

void PartitionedGraphModel::close() {
  if (impl_) {
    impl_->close();
    impl_.reset();
  }
}
} // namespace specferry::models::opt
