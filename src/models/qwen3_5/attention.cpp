#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/ops/graph_builder.hpp"
#include "devices/np101/api/ops/mixers.hpp"
#include "devices/np101/api/tensor_spec.hpp"
#include "models/qwen3_5/attention.hpp"
#include "vsi_nn_pub.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

namespace specferry::models::qwen3_5 {
using namespace np101;
using namespace np101::ops;

namespace {
using Shape = std::vector<std::uint32_t>;
constexpr auto f16 = DataType::Float16;
constexpr auto f32 = DataType::Float32;
constexpr auto i32 = DataType::Int32;

std::vector<std::uint8_t> integer_bytes(std::int32_t value) {
  std::vector<std::uint8_t> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

// Qwen weight names and normalization policy stay outside the generic builder.
class AttentionGraph : public GraphBuilder {
public:
  Config config;
  unsigned layer;
  WeightBank *bank;
  bool shared_fcl_projections;

  AttentionGraph(Context &context, const Config &configuration, unsigned index, WeightBank *weights,
                 bool use_shared_fcl, bool fuse_input_norm = false)
      : GraphBuilder(context, fuse_input_norm ? 116 : 100, fuse_input_norm ? 92 : 80, weights),
        config(configuration), layer(index), bank(weights), shared_fcl_projections(use_shared_fcl) {
  }

  Tensor weight(const WeightStore &weights, const std::string &name, Shape shape) {
    return GraphBuilder::weight(weights,
                                weights.find(config.layer_prefix(layer) + "self_attn." + name),
                                {f16, std::move(shape)});
  }

  Tensor project(const WeightStore &weights, Tensor input, const std::string &name, unsigned rows) {
    const auto matrix = weight(weights, name + ".weight", {input.spec.shape[0], rows});
    if (shared_fcl_projections) {
      return fully_connected(input, matrix, true);
    }
    return matmul(input, matrix, {rows, 1}, false, true);
  }

  Tensor rms_norm(const WeightStore &weights, Tensor input, const std::string &name) {
    return GraphBuilder::rms_norm(input, weight(weights, name + ".weight", {config.kv.head_dim}),
                                  config.epsilon, 1.0f, f16);
  }
};

class ProjectionGraph : public AttentionGraph {
public:
  Tensor hidden, position, query, key, value, gate;

  ProjectionGraph(Context &context, const WeightStore &weights, const Config &configuration,
                  unsigned layer, std::optional<TensorBinding> input, WeightBank *bank,
                  bool use_shared_fcl, bool fuse_input_norm)
      : AttentionGraph(context, configuration, layer, bank, use_shared_fcl, fuse_input_norm) {
    const auto d = config.kv.head_dim, qh = config.query_heads, kh = config.kv.heads;
    hidden = input ? bind(*input, config.hidden_spec()) : tensor(config.hidden_spec());
    position = tensor({i32, {1}});
    auto projection_input = hidden;
    if (fuse_input_norm) {
      const auto scale = GraphBuilder::weight(
          weights, weights.find(config.layer_prefix(layer) + "input_layernorm.weight"),
          {f16, {config.hidden}});
      // Preserve the separate decoder norm's FP32 arithmetic and final FP16
      // conversion. Only its graph boundary is removed; QKV still consumes FP16.
      projection_input = GraphBuilder::rms_norm(hidden, scale, config.epsilon, 1.0f, f16);
    }
    auto packed = reshape(project(weights, projection_input, "q_proj", 2 * d * qh), {2 * d, qh});
    query = rms_norm(weights, slice(packed, {0, 0}, {d, qh}), "q_norm");
    gate = slice(packed, {d, 0}, {d, qh});
    key = rms_norm(weights, reshape(project(weights, projection_input, "k_proj", d * kh), {d, kh}),
                   "k_norm");
    value = reshape(project(weights, projection_input, "v_proj", d * kh), {d, 1, kh});
    auto cosine = rope_table(false);
    auto sine = rope_table(true);
    query = reshape(rotate(query, cosine, sine), {d, qh / kh, kh});
    key = reshape(rotate(key, cosine, sine), {d, 1, kh});
    compile({hidden, position}, {query, key, value, gate});
  }

private:
  Tensor rope_table(bool sine) {
    std::vector<float> table(std::size_t(config.kv.capacity) * config.rotary_dim);
    for (unsigned position = 0; position < config.kv.capacity; ++position) {
      for (unsigned dim = 0; dim < config.rotary_dim; ++dim) {
        const float frequency =
            1.0f / std::pow(config.rope_theta,
                            float(2 * (dim % (config.rotary_dim / 2))) / config.rotary_dim);
        const float angle = position * frequency;
        table[std::size_t(position) * config.rotary_dim + dim] =
            sine ? std::sin(angle) : std::cos(angle);
      }
    }
    auto data = floats(table, {config.rotary_dim, config.kv.capacity}, f16);
    auto row = tensor({f16, {config.rotary_dim, 1}});
    node(VSI_NN_OP_GATHER, {data, position}, row)->nn_param.gather.axis = 1;
    return row;
  }

  Tensor rotate(Tensor input, Tensor cosine, Tensor sine) {
    auto heads = input.spec.shape[1];
    auto rotary = slice(input, {0, 0}, {config.rotary_dim, heads});
    auto first = slice(rotary, {0, 0}, {config.rotary_dim / 2, heads});
    auto second = slice(rotary, {config.rotary_dim / 2, 0}, {config.rotary_dim / 2, heads});
    auto turned = concat(binary(VSI_NN_OP_MULTIPLY, second, scalar(-1, f16)), first);
    auto rotated = binary(VSI_NN_OP_ADD, binary(VSI_NN_OP_MULTIPLY, rotary, cosine),
                          binary(VSI_NN_OP_MULTIPLY, turned, sine));
    if (config.rotary_dim == config.kv.head_dim) {
      return rotated;
    }
    return concat(rotated, slice(input, {config.rotary_dim, 0},
                                 {config.kv.head_dim - config.rotary_dim, heads}));
  }
};

class ReaderGraph : public AttentionGraph {
public:
  Tensor valid_length, output, probabilities;

  ReaderGraph(Context &context, const WeightStore &weights, ProjectionGraph &producer,
              KvCache &cache, std::optional<TensorBinding> destination)
      : AttentionGraph(context, producer.config, producer.layer, producer.bank,
                       producer.shared_fcl_projections) {
    auto query = share(producer, producer.query);
    auto gate = share(producer, producer.gate);
    Tensor keys{cache.retain_keys(graph), cache.spec().tensor()};
    Tensor values{cache.retain_values(graph), keys.spec};
    valid_length = tensor({i32, {1}});
    auto core = attention_core(*this, query, keys, values, valid_length,
                               1.0f / std::sqrt(float(config.kv.head_dim)));
    probabilities = core.probabilities;
    auto attended = core.attended;
    auto gated = binary(VSI_NN_OP_MULTIPLY, attended, unary(VSI_NN_OP_SIGMOID, gate, f16));
    output = destination ? bind(*destination, config.hidden_spec()) : tensor(config.hidden_spec());
    const auto width = config.kv.head_dim * config.query_heads;
    auto matrix = weight(weights, "o_proj.weight", {width, config.hidden});
    const auto projection_input = reshape(gated, {width, 1});
    if (shared_fcl_projections) {
      fully_connected(projection_input, matrix, true, output);
    } else {
      auto *projection = node(VSI_NN_OP_MATRIXMUL, {projection_input, matrix}, output);
      projection->nn_param.matrixmul.transpose[0] = false;
      projection->nn_param.matrixmul.transpose[1] = true;
    }
    compile({query, gate, keys, values, valid_length}, {output, probabilities});
  }
};
} // namespace

struct Attention::Impl {
  ProjectionGraph producer;
  KvCache cache;
  ReaderGraph reader;
  unsigned length = 0;
  bool failed = false;
  bool has_output = false;

  Impl(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
       std::optional<TensorBinding> input = {}, std::optional<TensorBinding> output = {},
       WeightBank *bank = nullptr, bool shared_fcl_projections = false,
       KvWriteMode kv_write_mode = KvWriteMode::CopyGraph, bool fuse_input_norm = false)
      : producer(context, weights, config, layer, input, bank, shared_fcl_projections,
                 fuse_input_norm),
        cache(context, producer.graph, producer.key.id, producer.value.id, config.kv,
              kv_write_mode),
        reader(context, weights, producer, cache, output) {}
};

Attention::Attention(Context &context, const WeightStore &weights, const Config &config,
                     unsigned layer) {
  validate_layer_weights(weights, config, layer, false);
  if (config.mixer(layer) != MixerKind::Attention) {
    throw std::invalid_argument("layer is not Attention");
  }
  impl_ = std::make_unique<Impl>(context, weights, config, layer);
}

Attention::Attention(Context &context, const WeightStore &weights, const Config &config,
                     unsigned layer, TensorBinding input, TensorBinding output, WeightBank *bank,
                     bool shared_fcl_projections, KvWriteMode kv_write_mode, bool fuse_input_norm)
    : impl_(nullptr) {
  validate_layer_weights(weights, config, layer, false);
  if (config.mixer(layer) != MixerKind::Attention) {
    throw std::invalid_argument("layer is not Attention");
  }
  if (shared_fcl_projections && !bank) {
    throw std::invalid_argument("Qwen shared FCL projections require a weight bank");
  }
  impl_ = std::make_unique<Impl>(context, weights, config, layer, input, output, bank,
                                 shared_fcl_projections, kv_write_mode, fuse_input_norm);
}

Attention::~Attention() = default;

void Attention::step(const std::vector<std::uint8_t> &hidden_fp16) {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("attention is closed or must be recreated after failure");
  }
  if (impl_->length >= impl_->cache.capacity()) {
    throw std::out_of_range("attention KV capacity exhausted");
  }
  if (hidden_fp16.size() != impl_->producer.config.hidden_spec().bytes()) {
    throw std::invalid_argument("attention expects one FP16 hidden vector");
  }
  impl_->failed = true;
  impl_->has_output = false;
  auto &producer = impl_->producer;
  upload_tensor(producer.graph, producer.hidden.id, hidden_fp16);
  impl_->failed = false;
  step();
}

void Attention::step() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("attention is closed or must be recreated after failure");
  }
  if (impl_->length >= impl_->cache.capacity()) {
    throw std::out_of_range("attention KV capacity exhausted");
  }
  impl_->failed = true;
  impl_->has_output = false;
  auto &producer = impl_->producer;
  upload_tensor(producer.graph, producer.position.id, integer_bytes(impl_->length));
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(producer.graph.get()); }),
        "attention projections");
  impl_->cache.write(impl_->length);
  upload_tensor(impl_->reader.graph, impl_->reader.valid_length.id,
                integer_bytes(impl_->length + 1));
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(impl_->reader.graph.get()); }),
        "attention cache reader");
  ++impl_->length;
  impl_->has_output = true;
  impl_->failed = false;
}

void Attention::reset() {
  if (!impl_ || impl_->failed) {
    throw std::logic_error("closed or failed attention must be recreated");
  }
  // Storage remains allocated. Every reused position is overwritten before it
  // becomes visible; the mask excludes the stale suffix.
  impl_->length = 0;
  impl_->failed = false;
  impl_->has_output = false;
}

void Attention::truncate(unsigned length) {
  if (!impl_ || impl_->failed || length > impl_->length) {
    throw std::out_of_range("cannot extend or truncate an invalid attention cache");
  }
  impl_->length = length;
  impl_->has_output = false;
}

unsigned Attention::length() const { return impl_ ? impl_->length : 0; }

std::size_t Attention::kv_revalidations() const { return impl_ ? impl_->cache.revalidations() : 0; }

std::vector<std::uint8_t> Attention::read(const std::string &name, TensorReadback method) {
  if (!impl_ || impl_->failed || !impl_->has_output) {
    throw std::logic_error("attention has no completed output");
  }
  if (name == "keys") {
    return impl_->cache.read_keys(method);
  }
  if (name == "values") {
    return impl_->cache.read_values(method);
  }
  if (name == "query" || name == "key" || name == "value") {
    auto &producer = impl_->producer;
    auto tensor = name == "query" ? producer.query : name == "key" ? producer.key : producer.value;
    return read_tensor(producer.graph, tensor.id, method);
  }
  if (name == "output" || name == "probabilities") {
    auto &reader = impl_->reader;
    return read_tensor(reader.graph, name == "output" ? reader.output.id : reader.probabilities.id,
                       method);
  }
  throw std::invalid_argument("unknown attention output: " + name);
}

void Attention::close() {
  if (impl_) {
    impl_->reader.graph.close();
    impl_->cache.close();
    impl_->producer.graph.close();
    impl_.reset();
  }
}
} // namespace specferry::models::qwen3_5
