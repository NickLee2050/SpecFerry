#include "models/opt/graph_model.hpp"
#include "models/opt/layer.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/tensor.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_pub.h"

#include <cstring>
#include <map>
#include <stdexcept>

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

class CacheStorage {
public:
  Graph graph;
  const Config config;
  std::vector<Tensor> tensors;

  CacheStorage(Context &context, const Config &spec)
      : graph(context, 2 * spec.layers * spec.heads, 0), config(spec) {
    const TensorSpec shape{f16, {spec.kv_spec().head_dim, spec.capacity}};
    const std::vector<std::uint8_t> zeros(shape.bytes());
    for (unsigned index = 0; index < 2 * spec.layers * spec.heads; ++index) {
      tensors.push_back({add_tensor(graph, shape, false, zeros), shape});
    }
  }

  Tensor bind(GraphBuilder &g, unsigned layer, unsigned head, bool values) {
    const auto tensor = tensors.at((layer * config.heads + head) * 2 + values);
    return g.bind({graph, tensor.id}, tensor.spec);
  }

  std::vector<std::uint8_t> read(unsigned layer, bool values) {
    std::vector<std::uint8_t> result;
    for (unsigned head = 0; head < config.heads; ++head) {
      const auto tensor = tensors.at((layer * config.heads + head) * 2 + values);
      const auto part = read_tensor(graph, tensor.id);
      result.insert(result.end(), part.begin(), part.end());
    }
    return result;
  }
};

class ExecutionGraph : public GraphBuilder {
public:
  const unsigned block;
  Tensor controls, selected;
  std::map<std::string, Tensor> input_tensors;

  ExecutionGraph(Context &context, const WeightStore &weights, WeightBank &bank,
                 CacheStorage &cache, const ModelConfig &config, unsigned tokens)
      : GraphBuilder(context, 24000, 16000, &bank), block(tokens) {
    controls = tensor({DataType::Int32, {tokens + 2}});
    auto ids = slice(controls, {0}, {tokens});
    auto base = slice(controls, {tokens}, {1});
    auto slot = slice(controls, {tokens + 1}, {1});
    std::vector<std::int32_t> offsets(tokens);
    for (unsigned index = 0; index < tokens; ++index) {
      offsets[index] = index;
    }
    auto positions =
        binary(VSI_NN_OP_ADD, constant({DataType::Int32, {tokens}}, integers(offsets)), base);
    auto limits = binary(VSI_NN_OP_ADD, positions, integer(1));
    auto table = vocabulary(weights, config);
    auto embedding = lookup_blocks(*this, table, ids);
    auto projected_input = project(embedding, weights, weights.find("decoder.project_in.weight"));
    const auto &record = weights.find("decoder.embed_positions.weight");
    auto learned = gather(weight(weights, record, record.spec),
                          binary(VSI_NN_OP_ADD, positions, integer(config.position_offset)), 1);
    auto hidden = binary(VSI_NN_OP_ADD, projected_input, learned);
    input_tensors = {{"token", ids},
                     {"position", positions},
                     {"slot", slot},
                     {"lookup", embedding},
                     {"input_projection", projected_input},
                     {"position_embedding", learned},
                     {"embedding", hidden}};

    for (unsigned layer = 0; layer < config.decoder.layers; ++layer) {
      const auto qkv = build_projections(*this, weights, config.decoder, layer, hidden);
      const auto dimension = config.decoder.kv_spec().head_dim;
      Tensor attended{};
      for (unsigned head = 0; head < config.decoder.heads; ++head) {
        auto query = slice(qkv.query, {head * dimension, 0}, {dimension, tokens});
        auto key = slice(qkv.key, {head * dimension, 0}, {dimension, tokens});
        auto value = slice(qkv.value, {head * dimension, 0}, {dimension, tokens});
        auto keys = append(cache.bind(*this, layer, head, false), key, slot);
        auto values = append(cache.bind(*this, layer, head, true), value, slot);
        auto result = attention_core(
            *this, reshape(query, {dimension, tokens, 1}),
            storage_reshape(*this, keys, {dimension, config.decoder.capacity, 1}),
            storage_reshape(*this, values, {dimension, config.decoder.capacity, 1}), limits, 1.0f);
        auto output = reshape(result.attended, {dimension, tokens});
        attended = head ? concat(attended, output) : output;
      }
      hidden =
          build_decoder_tail(*this, weights, config.decoder, layer, hidden, attended).at("output");
    }
    // Only the last column needs a prediction. Earlier prompt columns still update KV.
    auto last = tokens == 1 ? hidden : slice(hidden, {0, tokens - 1}, {config.decoder.hidden, 1});
    auto projected = project(last, weights, weights.find("decoder.project_out.weight"));
    std::vector<Tensor> logits;
    for (auto rows : table) {
      logits.push_back(matmul(projected, rows, {rows.spec.shape[1], 1}, false, true));
    }
    selected = greedy_token(*this, logits, config.block_rows);
    compile({controls}, {selected});
  }

private:
  std::vector<Tensor> vocabulary(const WeightStore &weights, const ModelConfig &config) {
    const auto &record = weights.find("decoder.embed_tokens.weight");
    std::vector<Tensor> blocks;
    for (unsigned first = 0; first < config.vocabulary; first += config.block_rows) {
      const auto remaining = config.vocabulary - first;
      const auto rows = remaining < config.block_rows ? remaining : config.block_rows;
      blocks.push_back(weight_chunk(weights, record, {f16, {config.embedding, rows}},
                                    std::size_t(first) * config.embedding * 2));
    }
    return blocks;
  }

  Tensor append(Tensor parent, Tensor values, Tensor index) {
    const auto width = parent.spec.shape[0], capacity = parent.spec.shape[1];
    auto destination =
        block == 1 ? parent : storage_reshape(*this, parent, {width * block, capacity / block});
    auto updated = indexed_append(*this, reshape(values, {width * block, 1}), index, destination);
    return block == 1 ? updated : storage_reshape(*this, updated, parent.spec.shape);
  }
};
} // namespace

struct GraphModel::Impl {
  ModelConfig config;
  unsigned block, position = 0;
  std::size_t completed_launches = 0;
  bool failed = false;
  WeightBank weights;
  CacheStorage cache;
  std::unique_ptr<ExecutionGraph> decode, prefill;
  ExecutionGraph *latest = nullptr;

  Impl(Context &context, const WeightStore &store, const ModelConfig &spec, unsigned size)
      : config(spec), block(size), weights(context), cache(context, spec.decoder) {
    decode = std::make_unique<ExecutionGraph>(context, store, weights, cache, config, 1);
    if (block != 1) {
      prefill = std::make_unique<ExecutionGraph>(context, store, weights, cache, config, block);
    }
  }

  void consume(const std::vector<std::int32_t> &tokens) {
    if (failed || (tokens.size() != 1 && tokens.size() != block) ||
        position + tokens.size() > config.decoder.capacity ||
        (tokens.size() != 1 && position % block)) {
      throw std::logic_error("invalid fixed-graph append or failed model");
    }
    for (auto token : tokens) {
      config.validate_token(token);
    }
    auto &execution = tokens.size() == 1 ? *decode : *prefill;
    auto controls = tokens;
    controls.push_back(position);
    controls.push_back(position / execution.block);
    failed = true;
    upload_tensor(execution.graph, execution.controls.id, integers(controls));
    if (!vxIsGraphVerified(execution.graph.get()->g)) {
      throw std::runtime_error("fixed graph became unverified; refusing per-token revalidation");
    }
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(execution.graph.get()); }),
          "execute complete OPT graph");
    position += tokens.size();
    ++completed_launches;
    latest = &execution;
    failed = false;
  }

  std::int32_t predict() {
    if (failed || !latest || !position) {
      throw std::logic_error("no complete graph prediction is available");
    }
    failed = true;
    const auto bytes = read_tensor(latest->graph, latest->selected.id);
    if (bytes.size() != sizeof(std::int32_t)) {
      throw std::runtime_error("prediction is not one INT32");
    }
    std::int32_t token;
    std::memcpy(&token, bytes.data(), sizeof(token));
    config.validate_token(token);
    failed = false;
    return token;
  }
};

GraphModel::GraphModel(Context &context, const WeightStore &weights, const ModelConfig &config,
                       unsigned prefill_block) {
  config.validate();
  if (!prefill_block || prefill_block > 8 || config.decoder.capacity % prefill_block) {
    throw std::invalid_argument("prefill block must be 1..8 and divide capacity");
  }
  validate_model_weights(weights, config, config.decoder.layers);
  impl_ = std::make_unique<Impl>(context, weights, config, prefill_block);
}

GraphModel::~GraphModel() = default;

inference::Generation GraphModel::generate(const std::vector<std::int32_t> &prompt,
                                           unsigned maximum) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed graph model");
  }
  const auto &config = impl_->config;
  return inference::generate_tokens(
      {config.vocabulary, config.decoder.capacity, config.bos, config.eos},
      {[&] {
         impl_->position = 0;
         impl_->latest = nullptr;
       },
       [&](std::int32_t token) { impl_->consume({token}); }, [&] { return impl_->predict(); },
       [&](const std::vector<std::int32_t> &tokens) {
         for (const auto chunk : inference::plan_prefill(tokens.size(), impl_->block)) {
           impl_->consume(
               {tokens.begin() + chunk.begin, tokens.begin() + chunk.begin + chunk.count});
         }
       }},
      prompt, maximum);
}

void GraphModel::step(std::int32_t token) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed graph model");
  }
  impl_->consume({token});
}

std::vector<std::uint8_t> GraphModel::read_input(const std::string &name) {
  if (!impl_ || impl_->failed || !impl_->latest) {
    throw std::logic_error("no completed graph input is available");
  }
  auto &execution = *impl_->latest;
  return read_tensor(execution.graph, execution.input_tensors.at(name).id);
}

std::vector<std::uint8_t> GraphModel::read_cache(unsigned layer, bool values) {
  if (!impl_ || impl_->failed || !impl_->position || layer >= impl_->config.decoder.layers) {
    throw std::logic_error("invalid graph-cache read");
  }
  return impl_->cache.read(layer, values);
}

std::size_t GraphModel::launches() const { return impl_ ? impl_->completed_launches : 0; }

std::size_t GraphModel::weight_bytes() const { return impl_ ? impl_->weights.payload_bytes() : 0; }

void GraphModel::close() {
  if (impl_) {
    if (impl_->prefill) {
      impl_->prefill->graph.close();
    }
    impl_->decode->graph.close();
    impl_->prefill.reset();
    impl_->decode.reset();
    impl_->cache.graph.close();
    impl_->weights.close();
    impl_.reset();
  }
}
} // namespace specferry::models::opt
