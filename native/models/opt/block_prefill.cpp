#include "models/opt/block_prefill.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/weight_bank.hpp"
#include "platform/runtime_profile.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;
constexpr unsigned block = 16;

TensorSpec projection_weight_spec(const WeightRecord &matrix, bool pointwise) {
  if (pointwise) {
    return {f16, {1, 1, matrix.spec.shape.at(0), matrix.spec.shape.at(1)}};
  }
  return matrix.spec;
}

Tensor shared_linear(GraphBuilder &builder, const WeightStore &weights, Tensor input,
                     const std::string &prefix, bool pointwise) {
  const auto &matrix = weights.find(prefix + ".weight");
  const auto &bias = weights.find(prefix + ".bias");
  // WeightBank owns ordinary backing that remains read-only for the graph lifetime.
  // Native 4D weight shape matches preloading: reshaping a second bank entry would
  // duplicate backing because WeightBank keys include the requested tensor shape.
  const auto kernel =
      builder.weight_chunk(weights, matrix, projection_weight_spec(matrix, pointwise), 0);
  const auto projected = pointwise ? builder.batch_pointwise_projection(input, kernel)
                                   : builder.fully_connected(input, kernel, true);
  return builder.binary(VSI_NN_OP_ADD, projected, builder.weight(weights, bias, bias.spec));
}

} // namespace

struct BlockPrefill::Impl {
  struct Stage {
    std::unique_ptr<GraphBuilder> builder;
    std::string role;
    std::optional<unsigned> cache_layer;
  };

  ModelConfig config;
  WeightBank bank;
  Graph cache;
  std::vector<Tensor> caches;
  std::vector<std::uint8_t> readback;
  std::vector<Stage> stages;
  Tensor controls{}, block_index{}, attention_mask{};
  bool reuse_cache_outputs = false;
  std::vector<std::pair<GraphBuilder *, Tensor>> cache_outputs;
  unsigned position = 0;
  std::size_t constants = 0;
  bool failed = false, closed = false;

  Impl(Context &context, const WeightStore &weights, const ModelConfig &spec,
       const std::vector<CompiledPrefillLayer> &compiled_layers, const PhaseObserver &phase,
       const std::vector<TensorBinding> &shared_cache)
      : config(spec), bank(context), cache(context, 2 * spec.decoder.layers + 4, 0) {
    try {
      build(context, weights, compiled_layers, phase, shared_cache);
    } catch (...) {
      try {
        close();
      } catch (...) {
      }
      throw;
    }
  }

  ~Impl() {
    try {
      close();
    } catch (...) {
    }
  }

  void require_open() const {
    if (closed || failed) {
      throw std::logic_error("closed or failed OPT block prefill");
    }
  }

  std::pair<std::unique_ptr<GraphBuilder>, Tensor>
  build_embedding(Context &context, const WeightStore &weights,
                  const std::array<unsigned, 3> &version) {
    const SdkScope timing_role("role", "block16.embedding");
    auto embedding = std::make_unique<GraphBuilder>(context, 512, 256, &bank);
    auto &entry = *embedding;
    check(vsi_nn_SetGraphVersion(entry.graph.get(), version[0], version[1], version[2]),
          "set compiled prefill embedding version");
    controls = entry.tensor({DataType::Int32, {2 * block + 1}});
    const auto tokens = entry.slice(controls, {0}, {block});
    const auto positions = entry.slice(controls, {block}, {block});
    const auto &table = weights.find("decoder.embed_tokens.weight");
    const auto block_rows = platform::prefill_embedding_block_rows(config.block_rows);
    std::vector<Tensor> chunks;
    for (unsigned first = 0; first < config.vocabulary; first += block_rows) {
      const auto rows = std::min(block_rows, config.vocabulary - first);
      chunks.push_back(entry.weight_chunk(weights, table, {f16, {config.embedding, rows}},
                                          std::size_t(first) * config.embedding * 2));
    }
    const auto lookup = lookup_blocks(entry, chunks, tokens);
    const auto &projection = weights.find("decoder.project_in.weight");
    const auto embedded =
        entry.fully_connected(lookup, entry.weight(weights, projection, projection.spec), true);
    const auto &position_table = weights.find("decoder.embed_positions.weight");
    const auto learned = entry.gather(
        entry.weight(weights, position_table, position_table.spec),
        entry.binary(VSI_NN_OP_ADD, positions, entry.integer(config.position_offset)), 1);
    const auto hidden = entry.binary(VSI_NN_OP_ADD, embedded, learned);
    return {std::move(embedding), hidden};
  }

  void build(Context &context, const WeightStore &weights,
             const std::vector<CompiledPrefillLayer> &artifacts, const PhaseObserver &phase,
             const std::vector<TensorBinding> &shared_cache) {
    reuse_cache_outputs = platform::reuse_prefill_cache_outputs();
    const auto &decoder = config.decoder;
    const auto pointwise_layers = platform::pointwise_prefill_layers();
    for (const auto &artifact : artifacts) {
      if (!artifact.full_decoder) {
        const SdkScope timing_role("role", "block16.preload");
        const auto &matrix = weights.find(decoder.prefix(artifact.layer) + "fc2.weight");
        bank.preload(weights, matrix,
                     projection_weight_spec(matrix, artifact.layer < pointwise_layers));
      }
    }
    const auto state = artifacts.front().graph.inputs.at(1);
    if ((state.type != f16 && state.type != DataType::Float32) ||
        state.shape != Shape{64, decoder.capacity, decoder.heads}) {
      throw std::invalid_argument("compiled prefill cache precision/shape differs");
    }
    for (unsigned index = 0; index < 2 * decoder.layers; ++index) {
      const SdkScope timing_role("role", "block16.cache-bindings");
      // Canonical input backing remains separate from every output. Serial SoC
      // producers may share output scratch, consumed immediately after each Run.
      const auto id = bind_tensor(shared_cache.at(index), cache, state);
      caches.push_back({id, state});
    }
    std::array<Tensor, 2> scratch{};
    if (reuse_cache_outputs) {
      for (auto &output : scratch) {
        output = {add_tensor(cache, state), state};
      }
    }
    const TensorSpec index_spec{DataType::Int32, {1}};
    const TensorSpec mask_spec{DataType::Float32, {decoder.capacity, block}};
    block_index = {add_tensor(cache, index_spec), index_spec};
    attention_mask = {add_tensor(cache, mask_spec), mask_spec};
    const auto compile = [&](std::unique_ptr<GraphBuilder> graph, const std::string &role,
                             const std::vector<Tensor> &inputs,
                             const std::vector<Tensor> &outputs) {
      const SdkScope timing_role("role", "block16." + role);
      if (phase) {
        phase("setup.prefill16." + role);
      }
      try {
        graph->compile(inputs, outputs);
      } catch (const std::exception &error) {
        throw std::runtime_error("compile OPT prefill " + role + ": " + error.what());
      }
      stages.push_back({std::move(graph), "block16." + role, {}});
    };
    auto [embedding, hidden] =
        build_embedding(context, weights, artifacts.front().graph.graph_version);
    compile(std::move(embedding), "embedding", {controls}, {hidden});
    for (unsigned layer = 0; layer < decoder.layers; ++layer) {
      const auto &artifact = artifacts.at(layer);
      const auto role = "decoder." + std::to_string(layer);
      const SdkScope timing_role("role", "block16." + role +
                                             (artifact.full_decoder ? ".compiled" : ".attention"));
      auto graph = std::make_unique<GraphBuilder>(context, 8, 1);
      const auto input = graph->share(*stages.back().builder, hidden);
      const auto key = graph->bind({cache, caches[2 * layer].id}, state);
      const auto value = graph->bind({cache, caches[2 * layer + 1].id}, state);
      const auto index = graph->bind({cache, block_index.id}, index_spec);
      const auto mask = graph->bind({cache, attention_mask.id}, mask_spec);
      const std::vector<Tensor> inputs{input, key, value, index, mask};
      std::vector<Tensor> output_bindings;
      if (reuse_cache_outputs) {
        output_bindings = {graph->tensor(artifact.graph.outputs.at(0)),
                           graph->bind({cache, scratch[0].id}, state),
                           graph->bind({cache, scratch[1].id}, state)};
      }
      const auto outputs = graph->compiled(artifact.graph, inputs, output_bindings);
      hidden = outputs.at(0);
      cache_outputs.emplace_back(graph.get(), outputs.at(1));
      cache_outputs.emplace_back(graph.get(), outputs.at(2));
      compile(std::move(graph), role + (artifact.full_decoder ? ".compiled" : ".attention"), inputs,
              outputs);
      if (reuse_cache_outputs) {
        stages.back().cache_layer = layer;
      }
      constants += artifact.graph.constant_bytes;
      if (!artifact.full_decoder) {
        const SdkScope ffn_timing_role("role", "block16." + role + ".ordinary_ffn");
        auto ffn = std::make_unique<GraphBuilder>(context, 128, 64, &bank);
        const auto &version = artifact.graph.graph_version;
        check(vsi_nn_SetGraphVersion(ffn->graph.get(), version[0], version[1], version[2]),
              "set compiled-attention prefill FFN version");
        const auto normalized = ffn->share(*stages.back().builder, hidden);
        const auto prefix = decoder.prefix(layer);
        const auto pointwise = layer < pointwise_layers;
        auto mlp = shared_linear(*ffn, weights, normalized, prefix + "fc1", pointwise);
        mlp = ffn->unary(VSI_NN_OP_RELU, mlp, f16);
        mlp = shared_linear(*ffn, weights, mlp, prefix + "fc2", pointwise);
        hidden = build_ffn_finish(*ffn, weights, decoder, layer, normalized, mlp).output;
        compile(std::move(ffn), role + ".ordinary_ffn", {normalized}, {hidden});
      }
    }
  }

  void commit_cache_layer(unsigned layer) {
    const SdkScope role("role", "block16.kv-commit");
    const Shape start{0, position, 0};
    const Shape end{64, position + block, config.decoder.heads};
    for (unsigned index = 2 * layer; index < 2 * layer + 2; ++index) {
      const auto &[owner, value] = cache_outputs.at(index);
      read_tensor_region_into(owner->graph, value.id, start, end, readback);
      upload_tensor_region(cache, caches.at(index).id, start, end, readback);
    }
  }

  void close() {
    if (closed) {
      return;
    }
    failed = true;
    while (!stages.empty()) {
      stages.back().builder->graph.close();
      stages.pop_back();
    }
    cache.close();
    bank.close();
    closed = true;
  }
};

BlockPrefill::BlockPrefill(Context &context, const WeightStore &weights, const ModelConfig &config,
                           const std::vector<CompiledPrefillLayer> &compiled_layers,
                           PhaseObserver phase, const std::vector<TensorBinding> &shared_cache) {
  config.validate();
  const auto &decoder = config.decoder;
  if (decoder.layers != 24 || decoder.hidden != 1024 || decoder.heads != 16 ||
      decoder.intermediate != 4096 ||
      (decoder.capacity != 64 && decoder.capacity != 128 && decoder.capacity != 256 &&
       decoder.capacity != 512) ||
      decoder.epsilon != 1e-5f || config.embedding != 512 || config.vocabulary != 50272 ||
      config.position_offset != 2 || compiled_layers.size() != decoder.layers ||
      shared_cache.size() != 2 * decoder.layers) {
    throw std::invalid_argument("compiled OPT M16 requires matching layers and shared KV");
  }
  validate_model_weights(weights, config, decoder.layers);
  impl_ = std::make_unique<Impl>(context, weights, config, compiled_layers, phase, shared_cache);
}

BlockPrefill::~BlockPrefill() = default;

BlockPrefill::Impl &BlockPrefill::require_impl() const {
  if (!impl_) {
    throw std::logic_error("OPT block prefill is unavailable");
  }
  impl_->require_open();
  return *impl_;
}

void BlockPrefill::consume(const std::vector<std::int32_t> &tokens) {
  auto &impl = require_impl();
  if (tokens.size() != block || impl.position + block >= impl.config.decoder.capacity) {
    throw std::invalid_argument("OPT block prefill needs sixteen tokens and an M1 tail slot");
  }
  const SdkScope timing_phase("phase", "prefill.position." + std::to_string(impl.position));
  std::vector<std::int32_t> controls(tokens);
  for (unsigned index = 0; index < block; ++index) {
    impl.config.validate_token(tokens[index]);
    controls.push_back(static_cast<std::int32_t>(impl.position + index));
  }
  controls.push_back(static_cast<std::int32_t>(impl.position / block));
  std::vector<std::uint8_t> bytes(controls.size() * sizeof(std::int32_t));
  std::memcpy(bytes.data(), controls.data(), bytes.size());
  impl.failed = true;
  {
    const SdkScope role("role", "block16.controls");
    upload_tensor_region(impl.stages.front().builder->graph, impl.controls.id, {0}, {2 * block + 1},
                         bytes);
  }
  const std::int32_t index = impl.position / block;
  std::vector<std::uint8_t> index_bytes(sizeof(index));
  std::memcpy(index_bytes.data(), &index, sizeof(index));
  upload_tensor_region(impl.cache, impl.block_index.id, {0}, {1}, index_bytes);
  std::vector<float> mask(block * impl.config.decoder.capacity);
  for (unsigned row = 0; row < block; ++row) {
    for (unsigned slot = 0; slot < impl.config.decoder.capacity; ++slot) {
      mask[row * impl.config.decoder.capacity + slot] = slot <= impl.position + row ? 0.0f : -1e9f;
    }
  }
  std::vector<std::uint8_t> mask_bytes(mask.size() * sizeof(float));
  std::memcpy(mask_bytes.data(), mask.data(), mask_bytes.size());
  upload_tensor_region(impl.cache, impl.attention_mask.id, {0, 0},
                       {impl.config.decoder.capacity, block}, mask_bytes);
  for (const auto &stage : impl.stages) {
    {
      const SdkScope role("role", stage.role);
      auto &graph = stage.builder->graph;
      if (!vxIsGraphVerified(graph.get()->g)) {
        throw std::runtime_error("OPT block prefill graph became unverified");
      }
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.get()); }),
            "run OPT block prefill graph");
    }
    if (stage.cache_layer) {
      // Both scratch outputs are consumed before another NBG producer can overwrite them.
      impl.commit_cache_layer(*stage.cache_layer);
    }
  }
  if (!impl.reuse_cache_outputs) {
    for (unsigned layer = 0; layer < impl.config.decoder.layers; ++layer) {
      impl.commit_cache_layer(layer);
    }
  }
  impl.position += block;
  impl.failed = false;
}

void BlockPrefill::reset() {
  auto &impl = require_impl();
  // The M1 owner clears shared inputs before the next request.
  impl.position = 0;
}

std::size_t BlockPrefill::weight_bytes() const {
  const auto &impl = require_impl();
  return impl.constants + impl.bank.payload_bytes();
}

void BlockPrefill::close() {
  if (impl_) {
    impl_->close();
  }
}
} // namespace specferry::models::opt
