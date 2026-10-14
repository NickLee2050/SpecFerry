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
    Tensor controls{}, embedded{}, full_position{};
    std::vector<std::unique_ptr<GraphBuilder>> decoder_graphs;
    std::vector<std::string> decoder_roles;
    std::unique_ptr<GraphBuilder> cache_commit;
    std::unique_ptr<GraphBuilder> head;
    std::unique_ptr<GraphBuilder> selection;
    Tensor projected{}, selected{};
    std::vector<Tensor> logits;

    std::string prefix() const { return block == 1 ? "" : "prefill" + std::to_string(block) + "."; }

    void close() {
      if (selection) {
        selection->graph.close();
        selection.reset();
      }
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
  const ProjectionPolicy projections;
  const CompiledHead compiled_head;
  const std::vector<CompiledProjections> compiled_projections;
  const CompiledFullModel compiled_full_model;
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
       std::vector<CompiledFfn> prefill_artifacts, ProjectionPolicy projection_policy,
       CompiledHead head_artifact, std::vector<CompiledProjections> projection_artifacts,
       CompiledFullModel full_artifact)
      : config(spec), variant(selected_variant), prefill_head_policy(prefill_head),
        prefill_block(block), compiled_ffn(std::move(compiled)),
        compiled_decoders(std::move(decoders)), compiled_prefill_ffn(std::move(prefill_artifacts)),
        projections(projection_policy), compiled_head(std::move(head_artifact)),
        compiled_projections(std::move(projection_artifacts)),
        compiled_full_model(std::move(full_artifact)), bank(context),
        cache(context, 2 * spec.decoder.layers + 2, 0) {
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
                         Tensor hidden, Tensor shared_controls, unsigned block,
                         const Projections *provided = nullptr) {
    const auto &decoder = config.decoder;
    const auto dimension = decoder.kv_spec().head_dim;
    auto slot_index = builder.slice(shared_controls, {2 * block}, {1});
    auto current_position = builder.slice(shared_controls, {block}, {block});
    auto limit = builder.binary(VSI_NN_OP_ADD, current_position, builder.integer(1));
    const auto backend = block == 1 ? projections.attention : LinearBackend::MatMul;
    auto qkv =
        provided ? *provided : build_projections(builder, weights, decoder, layer, hidden, backend);
    if (!provided && backend == LinearBackend::FullyConnected) {
      constant_weight_bytes += 3 * std::size_t(decoder.hidden) * decoder.hidden * 2;
    }
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
    const bool compiled_decoder =
        variant == Variant::CompiledDecoder || variant == Variant::CompiledFullModel;
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
    if (variant == Variant::MergedKvCompiledFfn || compiled_decoder) {
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
    if (variant == Variant::CompiledFullModel) {
      build_full_execution(execution, context, phase);
      return;
    }
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
        const auto &artifact = compiled_decoders.at(layer);
        const auto &compiled = artifact.graph;
        std::vector<Tensor> inputs{group_input};
        for (unsigned index : {0u, 1u}) {
          const auto storage = cache_tensors.at(2 * layer + index);
          inputs.push_back(builder.bind({cache, storage.id}, storage.spec));
        }
        inputs.push_back(builder.bind({cache, compiled_indices.id}, compiled_indices.spec));
        inputs.push_back(builder.bind({cache, compiled_mask.id}, compiled_mask.spec));
        std::vector<Tensor> sdk_inputs(inputs.size());
        for (std::size_t index = 0; index < inputs.size(); ++index) {
          sdk_inputs.at(artifact.input_order[index]) = inputs[index];
        }
        const auto sdk_outputs = builder.compiled(compiled, sdk_inputs);
        std::vector<Tensor> outputs;
        for (const auto index : artifact.output_order) {
          outputs.push_back(sdk_outputs.at(index));
        }
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
        const auto &artifact = (block == 1 ? compiled_ffn : compiled_prefill_ffn).at(layer);
        const auto &compiled = artifact.graph;
        const auto version = compiled.graph_version;
        check(vsi_nn_SetGraphVersion(builder.graph.get(), version[0], version[1], version[2]),
              "set compiled OPT attention graph version");
        GraphBuilder *attention_graph = nullptr;
        Tensor normalized{};
        if (!compiled_projections.empty() &&
            (projections.compiled_query || projections.compiled_output)) {
          const auto &projection = compiled_projections.at(layer);
          Tensor attended{};
          std::vector<Tensor> core_inputs{group_input, shared_controls};
          if (projections.compiled_query) {
            auto qkv_graph = std::make_unique<GraphBuilder>(context, 8, 1);
            const auto input = qkv_graph->share(*producer, previous);
            const auto projected = qkv_graph->compiled(projection.qkv, {input});
            const auto &order = projection.qkv_order;
            // Match the generated wrapper: NBG node ports use SDK order, while
            // public graph outputs retain ONNX Q/K/V order. The validated collection
            // uses this same distinction; exporting raw node order changes bindings.
            compile(*qkv_graph, stage + ".qkv", {input},
                    {projected[order[0]], projected[order[1]], projected[order[2]]});
            Projections qkv{builder.share(*qkv_graph, projected[order[0]]),
                            builder.share(*qkv_graph, projected[order[1]]),
                            builder.share(*qkv_graph, projected[order[2]])};
            keep(std::move(qkv_graph), std::to_string(layer) + ".qkv");
            attended =
                build_attention(builder, weights, layer, group_input, shared_controls, block, &qkv);
            core_inputs.insert(core_inputs.end(), {qkv.query, qkv.key, qkv.value});
            constant_weight_bytes += projection.qkv.constant_bytes;
          } else {
            attended =
                build_attention(builder, weights, layer, group_input, shared_controls, block);
          }
          if (projections.compiled_output) {
            compile(builder, stage + ".attention", core_inputs, {attended});
            auto projected_output = std::make_unique<GraphBuilder>(context, 8, 1);
            const auto out_input = projected_output->share(builder, attended);
            const auto mixer = projected_output->compiled(projection.output, {out_input}).at(0);
            compile(*projected_output, stage + ".out_proj", {out_input}, {mixer});
            auto finish = std::make_unique<GraphBuilder>(context, 128, 64, &bank);
            check(vsi_nn_SetGraphVersion(finish->graph.get(), version[0], version[1], version[2]),
                  "set compiled projection residual graph version");
            const auto residual_input = finish->share(*producer, previous);
            const auto mixed = finish->share(*projected_output, mixer);
            normalized =
                build_attention_finish(*finish, weights, decoder, layer, residual_input, mixed)
                    .normalized;
            compile(*finish, stage + ".attention_finish", {residual_input, mixed}, {normalized});
            attention_graph = finish.get();
            keep(std::move(owner), std::to_string(layer) + ".attention");
            keep(std::move(projected_output), std::to_string(layer) + ".out_proj");
            keep(std::move(finish), std::to_string(layer) + ".attention_finish");
            constant_weight_bytes += projection.output.constant_bytes;
          } else {
            normalized = build_attention_tail(builder, weights, decoder, layer, group_input,
                                              attended, projections.attention)
                             .normalized;
            if (projections.attention == LinearBackend::FullyConnected) {
              constant_weight_bytes += std::size_t(decoder.hidden) * decoder.hidden * 2;
            }
            compile(builder, stage + ".attention", core_inputs, {normalized});
            attention_graph = owner.get();
            keep(std::move(owner), std::to_string(layer) + ".attention");
          }
        } else {
          const auto attended =
              build_attention(builder, weights, layer, group_input, shared_controls, block);
          normalized =
              build_attention_tail(builder, weights, decoder, layer, group_input, attended,
                                   block == 1 ? projections.attention : LinearBackend::MatMul)
                  .normalized;
          if (block == 1 && projections.attention == LinearBackend::FullyConnected) {
            constant_weight_bytes += std::size_t(decoder.hidden) * decoder.hidden * 2;
          }
          compile(builder, stage + ".attention", {group_input, shared_controls}, {normalized});
          attention_graph = owner.get();
          keep(std::move(owner), std::to_string(layer) + ".attention");
        }

        // v105 rejected NBG mixed with ordinary decoder nodes. Separate graphs
        // retain fixed device buffers; no FFN weights enter the shared bank.
        auto ffn = std::make_unique<GraphBuilder>(context, 16, 8);
        const auto ffn_input = ffn->share(*attention_graph, normalized);
        const auto ffn_result = ffn->compiled(compiled, {ffn_input}).at(0);
        const auto ffn_role = artifact.includes_final_norm() ? ".ffn_finish" : ".ffn";
        compile(*ffn, stage + ffn_role, {ffn_input}, {ffn_result});
        auto *ffn_graph = ffn.get();
        keep(std::move(ffn), std::to_string(layer) + ffn_role);
        constant_weight_bytes += compiled.constant_bytes;
        if (artifact.includes_final_norm()) {
          producer = ffn_graph;
          previous = ffn_result;
          continue;
        }

        auto finish = std::make_unique<GraphBuilder>(context, 128, 64, &bank);
        check(vsi_nn_SetGraphVersion(finish->graph.get(), version[0], version[1], version[2]),
              "set compiled OPT residual graph version");
        const auto residual = finish->share(*attention_graph, normalized);
        const auto ffn_output = finish->share(*ffn_graph, ffn_result);
        const auto output =
            build_ffn_finish(*finish, weights, decoder, layer, residual, ffn_output).output;
        compile(*finish, stage + ".residual", {residual, ffn_output}, {output});
        producer = finish.get();
        previous = output;
        keep(std::move(finish), std::to_string(layer) + ".residual");
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
    if (!compiled_head.graph.path.empty()) {
      execution.logits = output.compiled(compiled_head.graph, {hidden});
      if (phase) {
        phase("setup." + execution.prefix() + "head-projection");
      }
      output.compile({shared_hidden}, execution.logits);
      execution.selection = std::make_unique<GraphBuilder>(context, 16, 8);
      auto &selection = *execution.selection;
      const auto logits = selection.share(output, execution.logits[0]);
      execution.selected = selection.argmax(logits, 0);
      if (phase) {
        phase("setup." + execution.prefix() + "selection");
      }
      selection.compile({logits}, {execution.selected});
      constant_weight_bytes += compiled_head.graph.constant_bytes;
      return;
    }
    if (projections.head == LinearBackend::FullyConnected) {
      const auto &project_out = weights.find("decoder.project_out.weight");
      execution.projected =
          output.fully_connected(hidden, output.constant_weight(weights, project_out));
      execution.logits.push_back(
          output.fully_connected(execution.projected, output.constant_weight(weights, record)));
      execution.selected = output.argmax(execution.logits[0], 0);
      constant_weight_bytes += project_out.bytes + record.bytes;
      if (phase) {
        phase("setup." + execution.prefix() + "head");
      }
      output.compile({shared_hidden}, {execution.selected});
      return;
    }
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

  void build_full_execution(ExecutionSet &execution, Context &context, const PhaseObserver &phase) {
    const auto &artifact = compiled_full_model;
    execution.head = std::make_unique<GraphBuilder>(context, 128, 1);
    auto &builder = *execution.head;
    const auto semantic_port = [&](unsigned index) {
      return artifact.graph.inputs.at(artifact.input_order.at(index));
    };
    execution.controls = builder.tensor(semantic_port(0));
    execution.full_position = builder.tensor(semantic_port(1));
    std::vector<Tensor> inputs{execution.controls, execution.full_position,
                               builder.bind({cache, compiled_indices.id}, compiled_indices.spec),
                               builder.bind({cache, compiled_mask.id}, compiled_mask.spec)};
    for (const auto &storage : cache_tensors) {
      inputs.push_back(builder.bind({cache, storage.id}, storage.spec));
    }
    std::vector<Tensor> sdk_inputs(inputs.size());
    for (unsigned index = 0; index < inputs.size(); ++index) {
      sdk_inputs.at(artifact.input_order.at(index)) = inputs[index];
    }
    const auto sdk_outputs = builder.compiled(artifact.graph, sdk_inputs);
    std::vector<Tensor> outputs;
    for (const auto index : artifact.output_order) {
      outputs.push_back(sdk_outputs.at(index));
    }
    if (phase) {
      phase("setup.fullgraph");
    }
    builder.compile(inputs, outputs);
    execution.logits = {outputs[0]};
    execution.selected = outputs[1];
    constant_weight_bytes += artifact.graph.constant_bytes;
    execution.cache_commit = std::make_unique<GraphBuilder>(context, 4 * config.decoder.layers,
                                                            2 * config.decoder.layers);
    auto &commit = *execution.cache_commit;
    std::vector<Tensor> commit_inputs, commit_outputs;
    for (unsigned index = 0; index < cache_tensors.size(); ++index) {
      auto source = commit.share(builder, outputs.at(2 + index));
      auto destination = commit.share(builder, inputs.at(4 + index));
      commit.node(VSI_NN_OP_DATACONVERT, {source}, destination);
      commit_inputs.push_back(source);
      commit_outputs.push_back(destination);
    }
    if (phase) {
      phase("setup.kv-commit");
    }
    commit.compile(commit_inputs, commit_outputs);
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
      upload_tensor(cache, compiled_indices.id, integers(indices));
      upload_tensor(cache, compiled_mask.id, mask_bytes);
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
    }
    if (run_head) {
      if (variant != Variant::CompiledFullModel) {
        run(*execution.head, execution.timing_prefix + ".head");
      }
      if (execution.selection) {
        run(*execution.selection, execution.timing_prefix + ".selection");
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
    const auto bytes = read_tensor(
        latest->selection ? latest->selection->graph : latest->head->graph, latest->selected.id);
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

PartitionedGraphModel::PartitionedGraphModel(
    Context &context, const WeightStore &weights, const ModelConfig &config, PhaseObserver phase,
    Variant variant, PrefillHeadPolicy prefill_head, unsigned prefill_block,
    std::vector<CompiledFfn> compiled_ffn, std::vector<CompiledDecoder> compiled_decoders,
    std::vector<CompiledFfn> compiled_prefill_ffn, ProjectionPolicy projections,
    CompiledHead compiled_head, std::vector<CompiledProjections> compiled_projections,
    CompiledFullModel compiled_full_model) {
  config.validate();
  if (variant != Variant::MergedKv && variant != Variant::MergedKvFc1Split2 &&
      variant != Variant::MergedKvCompiledFfn && variant != Variant::CompiledDecoder &&
      variant != Variant::CompiledFullModel) {
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
  if (variant == Variant::CompiledFullModel) {
    if (prefill_block != 1 || prefill_head != PrefillHeadPolicy::EveryToken ||
        !compiled_head.graph.path.empty()) {
      throw std::invalid_argument("compiled full OPT requires M1 and its embedded head");
    }
    validate_compiled_full_model(compiled_full_model, weights, config);
  } else if (!compiled_full_model.graph.path.empty()) {
    throw std::invalid_argument("compiled full OPT artifact requires its explicit variant");
  }
  if (!compiled_head.graph.path.empty()) {
    if (variant != Variant::MergedKvCompiledFfn || prefill_block != 1) {
      throw std::invalid_argument("compiled head candidate requires the compiled FFN M1 variant");
    }
    validate_compiled_head(compiled_head, weights, config);
  }
  const auto valid_backend = [](LinearBackend backend) {
    return backend == LinearBackend::MatMul || backend == LinearBackend::FullyConnected;
  };
  if (!valid_backend(projections.attention) || !valid_backend(projections.head) ||
      ((projections.attention != LinearBackend::MatMul ||
        projections.head != LinearBackend::MatMul) &&
       (variant != Variant::MergedKvCompiledFfn || prefill_block != 1))) {
    throw std::invalid_argument("FCL model candidate currently requires compiled FFN and M1");
  }
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
    if (!compiled_projections.empty()) {
      if (prefill_block != 1 || compiled_projections.size() != config.decoder.layers) {
        throw std::invalid_argument("compiled projections require all layers and M1");
      }
      for (unsigned layer = 0; layer < config.decoder.layers; ++layer) {
        const auto &artifact = compiled_projections[layer];
        validate_compiled_projections(artifact, weights, config.decoder, layer);
        for (const auto bytes :
             {projections.compiled_query ? artifact.qkv.constant_bytes : 0,
              projections.compiled_output ? artifact.output.constant_bytes : 0}) {
          if (bytes > segment_payload_limit - constants) {
            throw std::invalid_argument(
                "compiled projection sets exceed the constant payload limit");
          }
          constants += bytes;
        }
      }
    }
    if (prefill_block != 1) {
      validate_set(compiled_prefill_ffn, prefill_block);
    }
    std::size_t projection_constants = 0;
    if (projections.attention == LinearBackend::FullyConnected) {
      const auto matrices =
          compiled_projections.empty()
              ? 4u
              : (projections.compiled_query ? 0u : 3u) + (projections.compiled_output ? 0u : 1u);
      projection_constants += matrices * std::size_t(config.decoder.layers) *
                              config.decoder.hidden * config.decoder.hidden * 2;
    }
    if (!compiled_head.graph.path.empty()) {
      projection_constants += compiled_head.graph.constant_bytes;
    } else if (projections.head == LinearBackend::FullyConnected) {
      projection_constants += weights.find("decoder.project_out.weight").bytes +
                              weights.find("decoder.embed_tokens.weight").bytes;
    }
    if (projection_constants > segment_payload_limit - constants) {
      throw std::invalid_argument("compiled FFN and FCL projections exceed the constant budget");
    }
  } else if (!compiled_ffn.empty() || !compiled_prefill_ffn.empty() ||
             !compiled_projections.empty()) {
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
  impl_ =
      std::make_unique<Impl>(context, weights, config, phase, variant, prefill_head, prefill_block,
                             std::move(compiled_ffn), std::move(compiled_decoders),
                             std::move(compiled_prefill_ffn), projections, std::move(compiled_head),
                             std::move(compiled_projections), std::move(compiled_full_model));
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
  return {impl.variant == Variant::CompiledDecoder || impl.variant == Variant::CompiledFullModel
              ? DataType::Float32
              : f16,
          {decoder.kv_spec().head_dim, decoder.capacity, decoder.heads}};
}

std::vector<std::uint8_t> PartitionedGraphModel::read_output(const std::string &name,
                                                             np101::TensorReadback method) {
  require_impl().require_open();
  if (!impl_->prediction_valid) {
    throw std::logic_error("no partitioned OPT prediction is available");
  }
  if (name == "projected") {
    if (impl_->latest->selection || impl_->variant == Variant::CompiledFullModel) {
      throw std::invalid_argument("compiled head does not expose its intermediate projection");
    }
    return read_tensor(impl_->latest->head->graph, impl_->latest->projected.id, method);
  }
  if (name == "selected") {
    return read_tensor(impl_->latest->selection ? impl_->latest->selection->graph
                                                : impl_->latest->head->graph,
                       impl_->latest->selected.id, method);
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
