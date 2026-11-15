#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/memory_budget.hpp"
#include "devices/np101/api/ops/cache_update.hpp"
#include "devices/np101/api/ops/vocabulary.hpp"
#include "models/opt/partitioned_graph_internal.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
namespace {
constexpr auto f16 = DataType::Float16;
} // namespace

PartitionedGraphModel::Impl::Impl(Context &context, const WeightStore &weights,
                                  const ModelConfig &spec, const GraphOptions &options,
                                  CompiledArtifacts artifacts, const PhaseObserver &phase)
    : config(spec), variant(options.variant), kv_transfer(options.kv_transfer),
      prefill_head_policy(options.prefill_head), prefill_block(options.prefill_block),
      compiled_ffn(std::move(artifacts.feed_forward)),
      compiled_decoders(std::move(artifacts.decoders)),
      compiled_prefill_ffn(std::move(artifacts.prefill_feed_forward)),
      compiled_prefill_layers(std::move(artifacts.prefill_layers)),
      projections(options.projections), compiled_head(std::move(artifacts.head)),
      compiled_projections(std::move(artifacts.projections)),
      compiled_full_model(std::move(artifacts.full_model)) {
  try {
    // The mapped M1 full model has one graph with independent fixed input/output
    // allocations. Optional M16 prefill owns a separate chain and cache.
    if (kv_transfer == KvTransfer::DeviceCopy) {
      bank = std::make_unique<WeightBank>(context);
      cache = std::make_unique<Graph>(context, 2 * spec.decoder.layers + 2, 0);
    }
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

std::pair<Tensor, Tensor>
PartitionedGraphModel::Impl::update_merged_cache(GraphBuilder &builder, unsigned layer, Tensor key,
                                                 Tensor value, Tensor slot_index, unsigned block) {
  const auto &decoder = config.decoder;
  const auto dimension = decoder.kv_spec().head_dim;
  const auto key_storage = cache_tensors.at(layer * 2);
  const auto value_storage = cache_tensors.at(layer * 2 + 1);
  const auto append = [&](Tensor column, Tensor storage) {
    auto bound = builder.bind({*cache, storage.id}, storage.spec);
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

Tensor PartitionedGraphModel::Impl::build_attention(GraphBuilder &builder,
                                                    const WeightStore &weights, unsigned layer,
                                                    Tensor hidden, Tensor shared_controls,
                                                    unsigned block, const Projections *provided) {
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
  auto query = block == 1
                   ? builder.reshape(qkv.query, {dimension, 1, decoder.heads})
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

Tensor PartitionedGraphModel::Impl::build_decoder(GraphBuilder &builder, const WeightStore &weights,
                                                  unsigned layer, Tensor hidden,
                                                  Tensor shared_controls, unsigned block) {
  const auto &decoder = config.decoder;
  const auto attended = build_attention(builder, weights, layer, hidden, shared_controls, block);
  const auto tail = build_decoder_tail(builder, weights, decoder, layer, hidden, attended,
                                       variant == Variant::MergedKvFc1Split2);
  return tail.at("output");
}

void PartitionedGraphModel::Impl::build(Context &context, const WeightStore &weights,
                                        const PhaseObserver &phase) {
  if (kv_transfer == KvTransfer::MappedSlot) {
    decode = std::make_unique<ExecutionSet>(1);
    build_full_execution(*decode, context, phase);
    if (prefill_block == 16) {
      const SdkScope timing_role("role", "block16.storage");
      std::vector<TensorBinding> shared_cache;
      shared_cache.reserve(cache_tensors.size());
      for (const auto &tensor : cache_tensors) {
        shared_cache.push_back({decode->head->graph, tensor.id});
      }
      full_model_prefill = std::make_unique<BlockPrefill>(
          context, weights, config, compiled_prefill_layers, phase, shared_cache);
    }
    latest = decode.get();
    return;
  }
  const auto &decoder = config.decoder;
  if (variant == Variant::MergedKvFc1Split2 || !compiled_prefill_ffn.empty()) {
    // Load the large fc2 tensors before decoder graphs occupy SDK resources.
    // Subsequent bindings reuse these same chunks and the same memory budget.
    for (unsigned layer = 0; layer < decoder.layers; ++layer) {
      if (variant != Variant::MergedKvFc1Split2 &&
          !compiled_prefill_ffn.at(layer).graph.path.empty()) {
        continue;
      }
      const auto &matrix = weights.find(decoder.prefix(layer) + "fc2.weight");
      bank->preload(weights, matrix, matrix.spec);
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
    compiled_indices = {add_tensor(*cache, ports[3]), ports[3]};
    compiled_mask = {add_tensor(*cache, ports[4]), ports[4]};
  }
  const std::vector<std::uint8_t> zeros(cache_spec.bytes());
  for (unsigned index = 0; index < cache_count; ++index) {
    cache_tensors.push_back({add_tensor(*cache, cache_spec, false, zeros), cache_spec});
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

void PartitionedGraphModel::Impl::build_cache_reset(Context &context, const PhaseObserver &phase) {
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
    const auto destination = builder.bind({*cache, storage.id}, storage.spec);
    builder.node(VSI_NN_OP_DATACONVERT, {zeros}, destination);
    outputs.push_back(destination);
  }
  if (phase) {
    phase("setup.kv-reset");
  }
  builder.compile({zeros}, outputs);
}

void PartitionedGraphModel::Impl::build_execution_set(ExecutionSet &execution, Context &context,
                                                      const WeightStore &weights,
                                                      const PhaseObserver &phase) {
  if (variant == Variant::CompiledFullModel) {
    build_full_execution(execution, context, phase);
    return;
  }
  const auto &decoder = config.decoder;
  const auto block = execution.block;
  execution.embedding = std::make_unique<GraphBuilder>(context, 512, 256, bank.get());
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
  auto input_projection = entry.project(lookup, weights, weights.find("decoder.project_in.weight"));
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
                                       variant == Variant::CompiledDecoder ? 1 : 256, bank.get());
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
        inputs.push_back(builder.bind({*cache, storage.id}, storage.spec));
      }
      inputs.push_back(builder.bind({*cache, compiled_indices.id}, compiled_indices.spec));
      inputs.push_back(builder.bind({*cache, compiled_mask.id}, compiled_mask.spec));
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
      const auto version = compiled.path.empty() ? compiled_ffn.at(layer).graph.graph_version
                                                 : compiled.graph_version;
      check(vsi_nn_SetGraphVersion(builder.graph.get(), version[0], version[1], version[2]),
            "set compiled OPT attention graph version");
      GraphBuilder *attention_graph = nullptr;
      Tensor normalized{};
      if (block == 1 && !compiled_projections.empty() &&
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
          attended = build_attention(builder, weights, layer, group_input, shared_controls, block);
        }
        if (projections.compiled_output) {
          compile(builder, stage + ".attention", core_inputs, {attended});
          auto projected_output = std::make_unique<GraphBuilder>(context, 8, 1);
          const auto out_input = projected_output->share(builder, attended);
          const auto mixer = projected_output->compiled(projection.output, {out_input}).at(0);
          compile(*projected_output, stage + ".out_proj", {out_input}, {mixer});
          auto finish = std::make_unique<GraphBuilder>(context, 128, 64, bank.get());
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
          normalized = build_attention_tail(builder, weights, decoder, layer, group_input, attended,
                                            projections.attention)
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
      if (compiled.path.empty()) {
        // The ordinary prefill FFN keeps the existing FP16 boundaries and FP32 LN.
        // Its weights belong to the shared bank, without changing const classification.
        auto ffn = std::make_unique<GraphBuilder>(context, 128, 64, bank.get());
        check(vsi_nn_SetGraphVersion(ffn->graph.get(), version[0], version[1], version[2]),
              "set ordinary prefill FFN graph version");
        const auto input = ffn->share(*attention_graph, normalized);
        const auto prefix = decoder.prefix(layer);
        auto hidden = ffn->linear(input, weights, weights.find(prefix + "fc1.weight"),
                                  weights.find(prefix + "fc1.bias"), 2);
        hidden = ffn->unary(VSI_NN_OP_RELU, hidden, f16);
        hidden = ffn->linear(hidden, weights, weights.find(prefix + "fc2.weight"),
                             weights.find(prefix + "fc2.bias"));
        const auto output = build_ffn_finish(*ffn, weights, decoder, layer, input, hidden).output;
        compile(*ffn, stage + ".ordinary_ffn_finish", {input}, {output});
        producer = ffn.get();
        previous = output;
        keep(std::move(ffn), std::to_string(layer) + ".ordinary_ffn_finish");
        continue;
      }
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

      auto finish = std::make_unique<GraphBuilder>(context, 128, 64, bank.get());
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
  execution.head = std::make_unique<GraphBuilder>(context, 512, 256, bank.get());
  auto &output = *execution.head;
  auto shared_hidden = output.share(*producer, previous);
  auto hidden =
      block == 1 ? shared_hidden : output.slice(shared_hidden, {0, block - 1}, {decoder.hidden, 1});
  execution.head_input = hidden;
  if (block != 1 && !compiled_head.graph.path.empty()) {
    execution.shared_head = decode.get();
    const auto destination = output.share(*decode->head, decode->head_input);
    output.node(VSI_NN_OP_DATACONVERT, {hidden}, destination);
    if (phase) {
      phase("setup." + execution.prefix() + "head-bridge");
    }
    output.compile({shared_hidden}, {destination});
    return;
  }
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
  execution.projected = output.project(hidden, weights, weights.find("decoder.project_out.weight"));
  for (unsigned first = 0; first < config.vocabulary; first += config.block_rows) {
    const auto rows = std::min(config.block_rows, config.vocabulary - first);
    auto weight = output.weight_chunk(weights, record, {f16, {config.embedding, rows}},
                                      std::size_t(first) * config.embedding * 2);
    execution.logits.push_back(output.matmul(execution.projected, weight, {rows, 1}, false, true));
  }
  execution.selected = greedy_token(output, execution.logits, config.block_rows);
  if (phase) {
    phase("setup." + execution.prefix() + "head");
  }
  output.compile({shared_hidden}, {execution.selected});
}

void PartitionedGraphModel::Impl::build_full_execution(ExecutionSet &execution, Context &context,
                                                       const PhaseObserver &phase) {
  const SdkScope timing_role("role", "block1.fullgraph");
  const auto &artifact = compiled_full_model;
  const bool mapped = kv_transfer == KvTransfer::MappedSlot;
  execution.head = std::make_unique<GraphBuilder>(context, mapped ? 104 : 128, 1);
  auto &builder = *execution.head;
  std::vector<Tensor> inputs, sdk_inputs;
  if (mapped) {
    // Preserve the validated single-graph construction order in the transfer check.
    for (const auto &port : artifact.graph.inputs) {
      sdk_inputs.push_back(builder.tensor(port));
    }
    for (const auto index : artifact.input_order) {
      inputs.push_back(sdk_inputs.at(index));
    }
    execution.controls = inputs.at(0);
    execution.full_position = inputs.at(1);
    compiled_indices = inputs.at(2);
    compiled_mask = inputs.at(3);
    cache_tensors.assign(inputs.begin() + 4, inputs.end());
  } else {
    const auto semantic_port = [&](unsigned index) {
      return artifact.graph.inputs.at(artifact.input_order.at(index));
    };
    execution.controls = builder.tensor(semantic_port(0));
    execution.full_position = builder.tensor(semantic_port(1));
    inputs = {execution.controls, execution.full_position,
              builder.bind({*cache, compiled_indices.id}, compiled_indices.spec),
              builder.bind({*cache, compiled_mask.id}, compiled_mask.spec)};
    for (const auto &storage : cache_tensors) {
      inputs.push_back(builder.bind({*cache, storage.id}, storage.spec));
    }
    sdk_inputs.resize(inputs.size());
    for (unsigned index = 0; index < inputs.size(); ++index) {
      sdk_inputs.at(artifact.input_order.at(index)) = inputs[index];
    }
  }
  const auto sdk_outputs = builder.compiled(artifact.graph, sdk_inputs);
  std::vector<Tensor> outputs;
  for (const auto index : artifact.output_order) {
    outputs.push_back(sdk_outputs.at(index));
  }
  if (phase) {
    phase("setup.fullgraph");
  }
  builder.compile(mapped ? sdk_inputs : inputs, mapped ? sdk_outputs : outputs);
  execution.logits = {outputs[0]};
  execution.selected = outputs[1];
  constant_weight_bytes += artifact.graph.constant_bytes;
  if (mapped) {
    mapped_cache_outputs.assign(outputs.begin() + 2, outputs.end());
    const auto &spec = cache_tensors.front().spec;
    cache_zeroes.resize(spec.bytes());
    slot_end = {spec.shape.at(0), 1, spec.shape.at(2)};
    cache_slot.resize(std::size_t(slot_end[0]) * slot_end[2] * dtype_bytes(spec.type));
    // Verify is finished; initialize runtime KV once, outside generation timing.
    clear_mapped_cache();
    return;
  }
  execution.cache_commit =
      std::make_unique<GraphBuilder>(context, 4 * config.decoder.layers, 2 * config.decoder.layers);
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

PartitionedGraphModel::PartitionedGraphModel(Context &context, const WeightStore &weights,
                                             const ModelConfig &config, const GraphOptions &options,
                                             CompiledArtifacts artifacts, PhaseObserver phase) {
  const auto variant = options.variant;
  const auto prefill_head = options.prefill_head;
  const auto prefill_block = options.prefill_block;
  const auto projections = options.projections;
  const auto kv_transfer = options.kv_transfer;
  const auto &compiled_ffn = artifacts.feed_forward;
  const auto &compiled_decoders = artifacts.decoders;
  const auto &compiled_prefill_ffn = artifacts.prefill_feed_forward;
  const auto &compiled_head = artifacts.head;
  const auto &compiled_projections = artifacts.projections;
  const auto &compiled_full_model = artifacts.full_model;
  const auto &compiled_prefill_layers = artifacts.prefill_layers;
  config.validate();
  if (kv_transfer != KvTransfer::DeviceCopy && kv_transfer != KvTransfer::MappedSlot) {
    throw std::invalid_argument("unknown OPT KV transfer strategy");
  }
  if (kv_transfer == KvTransfer::MappedSlot &&
      (variant != Variant::CompiledFullModel ||
       (config.decoder.capacity != 64 && config.decoder.capacity != 128 &&
        config.decoder.capacity != 256 && config.decoder.capacity != 512) ||
       (prefill_block != 1 && prefill_block != 16))) {
    throw std::invalid_argument(
        "mapped-slot KV requires compiled full OPT C64/C128/C256/C512 with M1/M16");
  }
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
      variant != Variant::MergedKvCompiledFfn && variant != Variant::CompiledFullModel &&
      prefill_block != 1) {
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
    const bool block_prefill = prefill_block == 16 && kv_transfer == KvTransfer::MappedSlot &&
                               prefill_head == PrefillHeadPolicy::LastToken;
    if ((!block_prefill && (prefill_block != 1 || prefill_head != PrefillHeadPolicy::EveryToken)) ||
        !compiled_head.graph.path.empty() || !compiled_prefill_ffn.empty() ||
        block_prefill != !compiled_prefill_layers.empty()) {
      throw std::invalid_argument(
          "compiled full OPT requires its embedded head and explicit prefill");
    }
    validate_compiled_full_model(compiled_full_model, weights, config);
    if (compiled_full_model.current_cache_outputs() && kv_transfer != KvTransfer::MappedSlot) {
      throw std::invalid_argument("current-slot OPT outputs require mapped-slot KV commits");
    }
    std::size_t constants = compiled_full_model.graph.constant_bytes;
    for (const auto &artifact : compiled_prefill_layers) {
      if (artifact.graph.constant_bytes > segment_payload_limit - constants) {
        throw std::invalid_argument("compiled prefill chain exceeds constant budget");
      }
      constants += artifact.graph.constant_bytes;
    }
  } else if (!compiled_full_model.graph.path.empty() || !compiled_prefill_layers.empty()) {
    throw std::invalid_argument("compiled full OPT artifact requires its explicit variant");
  }
  if (!compiled_head.graph.path.empty()) {
    if (variant != Variant::MergedKvCompiledFfn) {
      throw std::invalid_argument("compiled head requires the compiled FFN variant");
    }
    validate_compiled_head(compiled_head, weights, config);
  }
  const auto valid_backend = [](LinearBackend backend) {
    return backend == LinearBackend::MatMul || backend == LinearBackend::FullyConnected;
  };
  if (!valid_backend(projections.attention) || !valid_backend(projections.head) ||
      ((projections.attention != LinearBackend::MatMul ||
        projections.head != LinearBackend::MatMul) &&
       variant != Variant::MergedKvCompiledFfn)) {
    throw std::invalid_argument("FCL projections require the compiled FFN variant");
  }
  if (prefill_block != 1 && projections.head == LinearBackend::FullyConnected &&
      compiled_head.graph.path.empty()) {
    throw std::invalid_argument("FCL block prefill requires the single shared compiled head");
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
        if (tokens == 16 && artifacts[layer].graph.path.empty()) {
          if (layer >= 4 || artifacts[layer].graph.constant_bytes) {
            throw std::invalid_argument("ordinary M16 FFNs are limited to the first four layers");
          }
          continue;
        }
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
      if (compiled_projections.size() != config.decoder.layers) {
        throw std::invalid_argument("compiled M1 projections require all layers");
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
  impl_ = std::make_unique<Impl>(context, weights, config, options, std::move(artifacts), phase);
}

} // namespace specferry::models::opt
