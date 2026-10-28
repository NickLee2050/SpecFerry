#include "models/opt/block_prefill.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/weight_bank.hpp"
#include "platform/runtime_profile.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;
constexpr unsigned block = 16, ordinary_layers = 6;

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

std::uint32_t widen_half(std::uint16_t bits) {
  const auto sign = std::uint32_t(bits & 0x8000u) << 16;
  auto fraction = std::uint32_t(bits & 0x3ffu);
  const auto exponent = (bits >> 10) & 0x1fu;
  if (!exponent) {
    if (!fraction) {
      return sign;
    }
    unsigned shift = 0;
    while (!(fraction & 0x400u)) {
      fraction <<= 1;
      ++shift;
    }
    return sign | ((113u - shift) << 23) | ((fraction & 0x3ffu) << 13);
  }
  if (exponent == 31) {
    return sign | 0x7f800000u | (fraction << 13);
  }
  return sign | ((exponent + 112u) << 23) | (fraction << 13);
}
} // namespace

std::vector<std::uint8_t> widen_prefill_cache(const std::vector<std::uint8_t> &packed,
                                              unsigned positions, unsigned heads, unsigned width) {
  if (!heads || !width ||
      std::size_t(heads) > std::numeric_limits<std::size_t>::max() / width / 4 ||
      std::size_t(positions) >
          std::numeric_limits<std::size_t>::max() / (std::size_t(heads) * width * 4) ||
      packed.size() != std::size_t(positions) * heads * width * 2) {
    throw std::invalid_argument("prefill KV prefix size differs from its geometry");
  }
  std::vector<std::uint8_t> result(std::size_t(positions) * heads * width * 4);
  for (unsigned head = 0; head < heads; ++head) {
    for (unsigned slot = 0; slot < positions; ++slot) {
      for (unsigned lane = 0; lane < width; ++lane) {
        const auto source = ((std::size_t(slot) * heads + head) * width + lane) * 2;
        const auto destination = ((std::size_t(head) * positions + slot) * width + lane) * 4;
        std::uint16_t value;
        std::memcpy(&value, packed.data() + source, sizeof(value));
        const auto wide = widen_half(value);
        std::memcpy(result.data() + destination, &wide, sizeof(wide));
      }
    }
  }
  return result;
}

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
  std::vector<std::uint8_t> zeroes, readback;
  std::vector<Stage> stages;
  Tensor controls{}, block_index{}, attention_mask{};
  bool compiled_attention = false, shared_decode_cache = false;
  bool reuse_cache_outputs = false;
  std::vector<std::pair<GraphBuilder *, Tensor>> cache_outputs;
  unsigned position = 0;
  std::size_t constants = 0;
  bool failed = false, closed = false;

  Impl(Context &context, const WeightStore &weights, const ModelConfig &spec,
       const std::vector<CompiledFfn> &artifacts, const PhaseObserver &phase,
       const std::vector<CompiledPrefillLayer> &compiled_layers,
       const std::vector<TensorBinding> &shared_cache)
      : config(spec), bank(context), cache(context, 2 * spec.decoder.layers + 4, 0) {
    try {
      if (compiled_layers.empty()) {
        build(context, weights, artifacts, phase);
      } else {
        build_compiled(context, weights, compiled_layers, phase, shared_cache);
      }
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
                  const std::array<unsigned, 3> &version = {}) {
    const SdkScope timing_role("role", "block16.embedding");
    auto embedding = std::make_unique<GraphBuilder>(context, 512, 256, &bank);
    auto &entry = *embedding;
    if (compiled_attention) {
      check(vsi_nn_SetGraphVersion(entry.graph.get(), version[0], version[1], version[2]),
            "set compiled prefill embedding version");
    }
    controls = entry.tensor({DataType::Int32, {2 * block + 1}});
    const auto tokens = entry.slice(controls, {0}, {block});
    const auto positions = entry.slice(controls, {block}, {block});
    const auto &table = weights.find("decoder.embed_tokens.weight");
    const auto block_rows = compiled_attention
                                ? platform::prefill_embedding_block_rows(config.block_rows)
                                : config.block_rows;
    std::vector<Tensor> chunks;
    for (unsigned first = 0; first < config.vocabulary; first += block_rows) {
      const auto rows = std::min(block_rows, config.vocabulary - first);
      chunks.push_back(entry.weight_chunk(weights, table, {f16, {config.embedding, rows}},
                                          std::size_t(first) * config.embedding * 2));
    }
    const auto lookup = lookup_blocks(entry, chunks, tokens);
    const auto &projection = weights.find("decoder.project_in.weight");
    // The compiled-attention chain uses the validated shared-weight FCL path;
    // the older handwritten candidate remains available for explicit comparison.
    const auto embedded =
        compiled_attention ? entry.fully_connected(
                                 lookup, entry.weight(weights, projection, projection.spec), true)
                           : entry.project(lookup, weights, projection);
    const auto &position_table = weights.find("decoder.embed_positions.weight");
    const auto learned = entry.gather(
        entry.weight(weights, position_table, position_table.spec),
        entry.binary(VSI_NN_OP_ADD, positions, entry.integer(config.position_offset)), 1);
    const auto hidden = entry.binary(VSI_NN_OP_ADD, embedded, learned);
    return {std::move(embedding), hidden};
  }

  void build_compiled(Context &context, const WeightStore &weights,
                      const std::vector<CompiledPrefillLayer> &artifacts,
                      const PhaseObserver &phase, const std::vector<TensorBinding> &shared_cache) {
    compiled_attention = true;
    reuse_cache_outputs = platform::reuse_prefill_cache_outputs();
    shared_decode_cache = !shared_cache.empty();
    const auto &decoder = config.decoder;
    const auto cached_layers =
        std::count_if(artifacts.begin(), artifacts.end(),
                      [](const auto &artifact) { return artifact.verify_cache.has_value(); });
    const bool cached = cached_layers != 0;
    const auto pointwise_layers = platform::pointwise_prefill_layers(cached_layers);
    for (const auto &artifact : artifacts) {
      if (artifact.verify_cache.has_value() != (cached && artifact.layer < pointwise_layers)) {
        throw std::invalid_argument("prefill cache selection must cover the complete fast prefix");
      }
    }
    auto cache_session = cached ? std::make_unique<GraphCacheSession>(context) : nullptr;

    struct PendingCache {
      GraphBuilder *builder;
      Tensor input;
      const VerifiedGraphCache *cache;
      std::string role;
    };

    std::vector<PendingCache> pending;
    for (const auto &artifact : artifacts) {
      if (!artifact.full_decoder) {
        const SdkScope timing_role("role", "block16.preload");
        const auto &matrix = weights.find(decoder.prefix(artifact.layer) + "fc2.weight");
        bank.preload(weights, matrix,
                     projection_weight_spec(matrix, artifact.layer < pointwise_layers));
      }
    }
    const TensorSpec state{DataType::Float32, {64, decoder.capacity, decoder.heads}};
    if (!shared_decode_cache) {
      zeroes.resize(state.bytes());
    }
    for (unsigned index = 0; index < 2 * decoder.layers; ++index) {
      const SdkScope timing_role("role", "block16.cache-bindings");
      // Canonical input backing remains separate from every output. Serial SoC
      // producers may share output scratch, consumed immediately after each Run.
      const auto id = shared_decode_cache ? bind_tensor(shared_cache.at(index), cache, state)
                                          : add_tensor(cache, state, false, zeroes);
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
                             const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs,
                             const VerifiedGraphCache *verified = nullptr) {
      const SdkScope timing_role("role", "block16." + role);
      if (phase) {
        phase("setup.prefill16." + role);
      }
      try {
        if (verified) {
          // Setup must finish before the next attention graph retains this output.
          // Defer only OpenVX Verify, which preserves the public tensor references.
          graph->setup(inputs, outputs);
          pending.push_back({graph.get(), inputs.front(), verified, role});
        } else {
          graph->compile(inputs, outputs);
        }
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
        compile(std::move(ffn), role + ".ordinary_ffn", {normalized}, {hidden},
                artifact.verify_cache ? &*artifact.verify_cache : nullptr);
      }
    }
    // All graph construction, NBG parameter binding and noncached Verify finish while
    // the Context cache flag is still disabled. No graph parameters change afterwards.
    const std::vector<std::uint8_t> warmup_input(std::size_t(decoder.hidden) * block * 2);
    for (const auto &entry : pending) {
      if (phase) {
        phase("verify.prefill16." + entry.role + ".cache");
      }
      cache_session->verify(*entry.builder, *entry.cache);
      // These are temporary hidden tensors, never canonical KV. Producers overwrite
      // them during inference. Warm once now to release pending SDK import temporaries.
      upload_tensor_region(entry.builder->graph, entry.input.id, {0, 0}, {decoder.hidden, block},
                           warmup_input);
      if (phase) {
        phase("warmup.prefill16." + entry.role);
      }
      check(vsi_nn_RunGraph(entry.builder->graph.get()), "cached prefill FFN warmup");
    }
    if (cache_session) {
      if (phase) {
        phase("verify.prefill16.restore-cache-policy");
      }
      cache_session->close();
    }
  }

  void build(Context &context, const WeightStore &weights,
             const std::vector<CompiledFfn> &artifacts, const PhaseObserver &phase) {
    const auto &decoder = config.decoder;
    // The ordinary fc2 tensors are materialized before decoder graphs occupy SDK resources.
    for (unsigned layer = 0; layer < ordinary_layers; ++layer) {
      const auto &matrix = weights.find(decoder.prefix(layer) + "fc2.weight");
      bank.preload(weights, matrix, matrix.spec);
    }
    const TensorSpec cache_spec{f16, {decoder.hidden, decoder.capacity}};
    zeroes.resize(cache_spec.bytes());
    for (unsigned index = 0; index < 2 * decoder.layers; ++index) {
      caches.push_back({add_tensor(cache, cache_spec, false, zeroes), cache_spec});
    }

    const auto compile = [&](std::unique_ptr<GraphBuilder> graph, std::string role,
                             const std::vector<Tensor> &inputs, Tensor output) {
      if (phase) {
        phase("setup.prefill16." + role);
      }
      graph->compile(inputs, {output});
      stages.push_back({std::move(graph), "block16." + role, {}});
    };

    auto [embedding, hidden] = build_embedding(context, weights);
    compile(std::move(embedding), "embedding", {controls}, hidden);

    for (unsigned layer = 0; layer < decoder.layers; ++layer) {
      const auto role = "decoder." + std::to_string(layer);
      const auto version = artifacts.at(ordinary_layers).graph.graph_version;
      auto attention = std::make_unique<GraphBuilder>(context, 512, 256, &bank);
      check(vsi_nn_SetGraphVersion(attention->graph.get(), version[0], version[1], version[2]),
            "set OPT prefill attention version");
      auto &builder = *attention;
      const auto input = builder.share(*stages.back().builder, hidden);
      const auto shared_controls = builder.share(*stages.front().builder, controls);
      const auto qkv = build_projections(builder, weights, decoder, layer, input);
      const auto slot = builder.slice(shared_controls, {2 * block}, {1});
      const auto append = [&](Tensor column, unsigned index) {
        const auto storage = caches.at(2 * layer + index);
        auto bound = builder.bind({cache, storage.id}, storage.spec);
        bound = storage_reshape(builder, bound, {decoder.hidden * block, decoder.capacity / block});
        column = builder.reshape(column, {decoder.hidden * block, 1});
        const auto updated = indexed_append(builder, column, slot, bound);
        return builder.permute(builder.reshape(updated, {64, decoder.heads, decoder.capacity}),
                               {0, 2, 1});
      };
      const auto keys = append(qkv.key, 0), values = append(qkv.value, 1);
      const auto query =
          builder.permute(builder.reshape(qkv.query, {64, decoder.heads, block}), {0, 2, 1});
      const auto limits = builder.binary(
          VSI_NN_OP_ADD, builder.slice(shared_controls, {block}, {block}), builder.integer(1));
      const auto attended = attention_core(builder, query, keys, values, limits, 1.0f).attended;
      const auto joined = builder.reshape(
          builder.permute(builder.reshape(attended, {64, block, decoder.heads}), {0, 2, 1}),
          {decoder.hidden, block});
      hidden = build_attention_tail(builder, weights, decoder, layer, input, joined).normalized;
      compile(std::move(attention), role + ".attention", {input, shared_controls}, hidden);

      const auto &artifact = artifacts.at(layer);
      if (artifact.graph.path.empty()) {
        auto ffn = std::make_unique<GraphBuilder>(context, 128, 64, &bank);
        check(vsi_nn_SetGraphVersion(ffn->graph.get(), version[0], version[1], version[2]),
              "set ordinary OPT prefill FFN version");
        const auto input = ffn->share(*stages.back().builder, hidden);
        const auto prefix = decoder.prefix(layer);
        auto mlp = ffn->linear(input, weights, weights.find(prefix + "fc1.weight"),
                               weights.find(prefix + "fc1.bias"), 2);
        mlp = ffn->unary(VSI_NN_OP_RELU, mlp, f16);
        mlp = ffn->linear(mlp, weights, weights.find(prefix + "fc2.weight"),
                          weights.find(prefix + "fc2.bias"));
        hidden = build_ffn_finish(*ffn, weights, decoder, layer, input, mlp).output;
        compile(std::move(ffn), role + ".ordinary_ffn_finish", {input}, hidden);
      } else {
        auto *residual_owner = stages.back().builder.get();
        const auto residual_tensor = hidden;
        auto ffn = std::make_unique<GraphBuilder>(context, 16, 8);
        const auto input = ffn->share(*residual_owner, hidden);
        hidden = ffn->compiled(artifact.graph, {input}).at(0);
        compile(std::move(ffn), role + ".ffn", {input}, hidden);
        constants += artifact.graph.constant_bytes;
        auto finish = std::make_unique<GraphBuilder>(context, 128, 64, &bank);
        check(vsi_nn_SetGraphVersion(finish->graph.get(), version[0], version[1], version[2]),
              "set OPT prefill residual version");
        const auto residual = finish->share(*residual_owner, residual_tensor);
        const auto mlp = finish->share(*stages.back().builder, hidden);
        hidden = build_ffn_finish(*finish, weights, decoder, layer, residual, mlp).output;
        compile(std::move(finish), role + ".residual", {residual, mlp}, hidden);
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
                           std::vector<CompiledFfn> compiled_ffn, PhaseObserver phase,
                           std::vector<CompiledPrefillLayer> compiled_layers,
                           const std::vector<TensorBinding> &shared_cache) {
  config.validate();
  const auto &decoder = config.decoder;
  const bool supported_capacity =
      decoder.capacity == 64 ||
      (!compiled_layers.empty() && (decoder.capacity == 128 || decoder.capacity == 256));
  if (decoder.layers != 24 || decoder.hidden != 1024 || decoder.heads != 16 ||
      decoder.intermediate != 4096 || !supported_capacity || decoder.epsilon != 1e-5f ||
      config.embedding != 512 || config.vocabulary != 50272 || config.position_offset != 2 ||
      (compiled_layers.empty()
           ? compiled_ffn.size() != decoder.layers
           : (!compiled_ffn.empty() || compiled_layers.size() != decoder.layers))) {
    throw std::invalid_argument(
        "OPT block prefill requires C64/M16 or compiled-attention C128/C256/M16 geometry");
  }
  if (!shared_cache.empty() &&
      (compiled_layers.empty() || shared_cache.size() != 2 * decoder.layers)) {
    throw std::invalid_argument("shared prefill cache requires all compiled-attention KV inputs");
  }
  validate_model_weights(weights, config, decoder.layers);
  for (unsigned layer = 0; layer < compiled_ffn.size(); ++layer) {
    const auto &artifact = compiled_ffn[layer];
    if (layer < ordinary_layers) {
      if (!artifact.graph.path.empty() || artifact.graph.constant_bytes) {
        throw std::invalid_argument("OPT block prefill requires six ordinary FFN layers");
      }
    } else {
      validate_compiled_ffn(artifact, weights, decoder, layer, block);
      if (artifact.includes_final_norm()) {
        throw std::invalid_argument("OPT block prefill requires MLP-only M16 artifacts");
      }
    }
  }
  impl_ = std::make_unique<Impl>(context, weights, config, compiled_ffn, phase, compiled_layers,
                                 shared_cache);
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
    if (impl.compiled_attention) {
      upload_tensor_region(impl.stages.front().builder->graph, impl.controls.id, {0},
                           {2 * block + 1}, bytes);
    } else {
      upload_tensor(impl.stages.front().builder->graph, impl.controls.id, bytes);
    }
  }
  if (impl.compiled_attention) {
    const std::int32_t index = impl.position / block;
    std::vector<std::uint8_t> index_bytes(sizeof(index));
    std::memcpy(index_bytes.data(), &index, sizeof(index));
    upload_tensor_region(impl.cache, impl.block_index.id, {0}, {1}, index_bytes);
    std::vector<float> mask(block * impl.config.decoder.capacity);
    for (unsigned row = 0; row < block; ++row) {
      for (unsigned slot = 0; slot < impl.config.decoder.capacity; ++slot) {
        mask[row * impl.config.decoder.capacity + slot] =
            slot <= impl.position + row ? 0.0f : -1e9f;
      }
    }
    std::vector<std::uint8_t> mask_bytes(mask.size() * sizeof(float));
    std::memcpy(mask_bytes.data(), mask.data(), mask_bytes.size());
    upload_tensor_region(impl.cache, impl.attention_mask.id, {0, 0},
                         {impl.config.decoder.capacity, block}, mask_bytes);
  }
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
  if (impl.compiled_attention && !impl.reuse_cache_outputs) {
    for (unsigned layer = 0; layer < impl.config.decoder.layers; ++layer) {
      impl.commit_cache_layer(layer);
    }
  }
  impl.position += block;
  impl.failed = false;
}

void BlockPrefill::reset() {
  auto &impl = require_impl();
  const SdkScope role("role", "block16.kv-reset");
  impl.failed = true;
  // The M1 owner clears shared inputs once, after resetting this chain's position.
  if (!impl.shared_decode_cache) {
    for (const auto &storage : impl.caches) {
      if (impl.compiled_attention) {
        upload_tensor_region(impl.cache, storage.id, Shape(storage.spec.shape.size(), 0),
                             storage.spec.shape, impl.zeroes);
      } else {
        upload_tensor(impl.cache, storage.id, impl.zeroes);
      }
    }
  }
  impl.position = 0;
  impl.failed = false;
}

unsigned BlockPrefill::position() const { return require_impl().position; }

bool BlockPrefill::shares_decode_cache() const { return require_impl().shared_decode_cache; }

std::vector<std::uint8_t> BlockPrefill::read_cache(unsigned layer, bool values) {
  auto &impl = require_impl();
  if (layer >= impl.config.decoder.layers) {
    throw std::out_of_range("OPT block prefill cache layer exceeds the model");
  }
  if (!impl.position) {
    return {};
  }
  const SdkScope role("role", "block16.kv-handoff");
  const auto &storage = impl.caches.at(2 * layer + values);
  if (impl.compiled_attention) {
    read_tensor_region_into(impl.cache, storage.id, {0, 0, 0},
                            {64, impl.position, impl.config.decoder.heads}, impl.readback);
    return impl.readback;
  }
  read_tensor_region_into(impl.cache, storage.id, {0, 0},
                          {impl.config.decoder.hidden, impl.position}, impl.readback);
  return widen_prefill_cache(impl.readback, impl.position, impl.config.decoder.heads, 64);
}

std::size_t BlockPrefill::constant_weight_bytes() const { return require_impl().constants; }

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
