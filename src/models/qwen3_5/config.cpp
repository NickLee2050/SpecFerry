#include "models/qwen3_5/config.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace specferry::models::qwen3_5 {
void Config::validate() const {
  kv.validate();
  delta.validate();
  // Longer-cache experiments currently belong to OPT; retain this adapter's limit.
  if (kv.capacity > 512 || !hidden || !intermediate || !query_heads || query_heads % kv.heads ||
      query_heads > std::numeric_limits<unsigned>::max() / kv.head_dim / 2 || !rotary_dim ||
      rotary_dim % 2 || rotary_dim > kv.head_dim || !std::isfinite(rope_theta) || rope_theta <= 0 ||
      !std::isfinite(epsilon) || epsilon <= 0 || hidden != delta.hidden || layer_types.empty()) {
    throw std::invalid_argument("unsupported Qwen component configuration");
  }
  hidden_spec().bytes();
}

np101::TensorSpec Config::hidden_spec() const { return {np101::DataType::Float16, {hidden, 1}}; }

std::string Config::layer_prefix(unsigned layer) const {
  mixer(layer);
  return "model.layers." + std::to_string(layer) + ".";
}

MixerKind Config::mixer(unsigned layer) const {
  if (layer >= layer_types.size()) {
    throw std::out_of_range("layer index exceeds configured architecture");
  }
  return layer_types[layer];
}

Config read_config(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string header, kind;
  if (!std::getline(input, header) || header != "specferry-qwen-components 1") {
    throw std::invalid_argument("missing or unsupported component configuration");
  }
  std::string line;
  std::getline(input, line);
  std::istringstream parameters(line);
  auto dimension = [&] {
    std::string value;
    parameters >> value;
    const auto shape = np101::parse_shape(value);
    if (shape.size() != 1) {
      throw std::invalid_argument("component dimension must be one positive integer");
    }
    return shape.front();
  };
  Config config{};
  config.hidden = dimension();
  config.intermediate = dimension();
  config.query_heads = dimension();
  config.kv.heads = dimension();
  config.kv.head_dim = dimension();
  config.kv.capacity = dimension();
  config.rotary_dim = dimension();
  if (!(parameters >> config.rope_theta >> config.epsilon)) {
    throw std::invalid_argument("invalid component parameter record");
  }
  config.delta.heads = dimension();
  config.delta.key_dim = dimension();
  config.delta.value_dim = dimension();
  config.delta.convolution_width = dimension();
  if (parameters >> kind) {
    throw std::invalid_argument("unexpected trailing component parameter");
  }
  config.delta.hidden = config.hidden;
  config.delta.epsilon = config.epsilon;
  while (input >> kind) {
    if (kind != "delta" && kind != "attention") {
      throw std::invalid_argument("unknown mixer kind in component configuration");
    }
    config.layer_types.push_back(kind == "delta" ? MixerKind::DeltaNet : MixerKind::Attention);
  }
  config.validate();
  return config;
}

void validate_layer_weights(const np101::WeightStore &weights, const Config &config, unsigned layer,
                            bool include_feed_forward) {
  config.validate();
  const auto prefix = config.layer_prefix(layer);
  auto require = [&](const std::string &name, np101::DataType type,
                     std::vector<std::uint32_t> shape) {
    const auto &record = weights.find(prefix + name);
    if (record.spec.type != type || record.spec.shape != shape) {
      throw std::invalid_argument("weight contract mismatch before graph creation: " + record.name);
    }
  };
  constexpr auto f16 = np101::DataType::Float16, f32 = np101::DataType::Float32;
  if (include_feed_forward) {
    require("input_layernorm.weight", f16, {config.hidden});
    require("post_attention_layernorm.weight", f16, {config.hidden});
    require("mlp.gate_proj.weight", f16, {config.hidden, config.intermediate});
    require("mlp.up_proj.weight", f16, {config.hidden, config.intermediate});
    require("mlp.down_proj.weight", f16, {config.intermediate, config.hidden});
  }
  if (config.mixer(layer) == MixerKind::Attention) {
    const auto dim = config.kv.head_dim, query_width = dim * config.query_heads;
    require("self_attn.q_proj.weight", f16, {config.hidden, 2 * query_width});
    for (const auto &projection : {"k_proj", "v_proj"}) {
      require(std::string("self_attn.") + projection + ".weight", f16,
              {config.hidden, dim * config.kv.heads});
    }
    require("self_attn.o_proj.weight", f16, {query_width, config.hidden});
    require("self_attn.q_norm.weight", f16, {dim});
    require("self_attn.k_norm.weight", f16, {dim});
  } else {
    const auto &delta = config.delta;
    require("linear_attn.in_proj_qkv.weight", f16, {config.hidden, delta.channels()});
    require("linear_attn.in_proj_z.weight", f16, {config.hidden, delta.value_width()});
    require("linear_attn.in_proj_a.weight", f16, {config.hidden, delta.heads});
    require("linear_attn.in_proj_b.weight", f16, {config.hidden, delta.heads});
    require("linear_attn.out_proj.weight", f16, {delta.value_width(), config.hidden});
    require("linear_attn.conv1d.weight", f16, {delta.convolution_width, 1, delta.channels()});
    require("linear_attn.dt_bias", f16, {delta.heads});
    require("linear_attn.A_log", f32, {delta.heads});
    require("linear_attn.norm.weight", f32, {delta.value_dim});
  }
}
} // namespace specferry::models::qwen3_5
