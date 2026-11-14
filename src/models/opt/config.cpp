#include "models/opt/config.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace specferry::models::opt {
void Config::validate() const {
  if (!hidden || !intermediate || !heads || hidden % heads || !layers || !std::isfinite(epsilon) ||
      epsilon <= 0 ||
      std::uint64_t(hidden) * std::max(hidden, intermediate) * 2 > 8 * 1024 * 1024) {
    throw std::invalid_argument("unsupported OPT component dimensions");
  }
  kv_spec().validate();
}

np101::TensorSpec Config::hidden_spec() const { return {np101::DataType::Float16, {hidden, 1}}; }

np101::KvSpec Config::kv_spec() const {
  if (!heads || hidden % heads) {
    throw std::invalid_argument("invalid OPT head grouping");
  }
  return {heads, hidden / heads, capacity};
}

std::string Config::prefix(unsigned layer) const {
  if (layer >= layers) {
    throw std::out_of_range("OPT layer exceeds configuration");
  }
  return "decoder.layers." + std::to_string(layer) + ".";
}

Config read_config(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line) || line != "specferry-opt-components 1") {
    throw std::invalid_argument("missing or unsupported OPT component configuration");
  }
  auto dimension = [&] {
    std::string value;
    input >> value;
    auto shape = np101::parse_shape(value);
    if (shape.size() != 1) {
      throw std::invalid_argument("OPT dimension must be a positive integer");
    }
    return shape.front();
  };
  Config result{dimension(), dimension(), dimension(), dimension(), dimension(), 0};
  if (!(input >> result.epsilon) || input >> line) {
    throw std::invalid_argument("invalid OPT parameter record");
  }
  result.validate();
  return result;
}

void validate_weights(const np101::WeightStore &weights, const Config &config,
                      const std::vector<unsigned> &selected) {
  config.validate();
  if (selected.empty()) {
    throw std::invalid_argument("empty OPT layer selection");
  }
  std::vector<unsigned> seen;
  for (auto layer : selected) {
    if (std::find(seen.begin(), seen.end(), layer) != seen.end()) {
      throw std::invalid_argument("duplicate OPT layer selection");
    }
    seen.push_back(layer);
    auto require = [&](const std::string &name, std::vector<std::uint32_t> shape) {
      const auto &record = weights.find(config.prefix(layer) + name);
      if (record.spec.type != np101::DataType::Float16 || record.spec.shape != shape) {
        throw std::invalid_argument("OPT weight contract mismatch: " + record.name);
      }
    };
    for (const auto *projection : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
      const auto name = std::string("self_attn.") + projection;
      require(name + ".weight", {config.hidden, config.hidden});
      require(name + ".bias", {config.hidden});
    }
    for (const auto *norm : {"self_attn_layer_norm", "final_layer_norm"}) {
      require(std::string(norm) + ".weight", {config.hidden});
      require(std::string(norm) + ".bias", {config.hidden});
    }
    require("fc1.weight", {config.hidden, config.intermediate});
    require("fc1.bias", {config.intermediate});
    require("fc2.weight", {config.intermediate, config.hidden});
    require("fc2.bias", {config.hidden});
  }
}

void ModelConfig::validate() const {
  decoder.validate();
  if (!embedding || !vocabulary || vocabulary > std::numeric_limits<std::int32_t>::max() ||
      positions < decoder.capacity || position_offset != 2 ||
      positions > std::numeric_limits<std::int32_t>::max() - position_offset || !block_rows ||
      block_rows > 4096 || bos >= vocabulary || eos >= vocabulary || pad >= vocabulary ||
      pad == bos || pad == eos ||
      std::uint64_t(embedding) * std::max(block_rows, decoder.hidden) * 2 > 8 * 1024 * 1024 ||
      std::uint64_t(decoder.hidden) * (positions + position_offset) * 2 > 8 * 1024 * 1024) {
    throw std::invalid_argument("unsupported OPT input/output configuration");
  }
}

void ModelConfig::validate_token(std::int32_t token) const {
  if (token < 0 || unsigned(token) >= vocabulary) {
    throw std::invalid_argument("token is outside the vocabulary");
  }
}

ModelConfig read_model_config(const std::filesystem::path &directory) {
  auto decoder = read_config(directory / "components.txt");
  std::ifstream input(directory / "model.txt");
  std::string line;
  if (!std::getline(input, line) || line != "specferry-opt-model 1") {
    throw std::invalid_argument("missing or unsupported OPT model configuration");
  }
  ModelConfig result{decoder, 0, 0, 0, 0, 0, 0, 0, 0};
  if (!(input >> result.embedding >> result.vocabulary >> result.positions >>
        result.position_offset >> result.block_rows >> result.bos >> result.eos >> result.pad) ||
      input >> line) {
    throw std::invalid_argument("invalid OPT model configuration");
  }
  result.validate();
  return result;
}

void validate_model_weights(const np101::WeightStore &weights, const ModelConfig &config,
                            unsigned layers) {
  config.validate();
  if (!layers || layers > config.decoder.layers) {
    throw std::invalid_argument("invalid OPT resident layer count");
  }
  std::vector<unsigned> selected(layers);
  std::iota(selected.begin(), selected.end(), 0);
  validate_weights(weights, config.decoder, selected);
  auto require = [&](const std::string &name, std::vector<std::uint32_t> shape) {
    const auto &record = weights.find("decoder." + name + ".weight");
    if (record.spec.type != np101::DataType::Float16 || record.spec.shape != shape) {
      throw std::invalid_argument("OPT input/output weight mismatch: " + record.name);
    }
  };
  require("embed_tokens", {config.embedding, config.vocabulary});
  require("embed_positions", {config.decoder.hidden, config.positions + config.position_offset});
  require("project_in", {config.embedding, config.decoder.hidden});
  require("project_out", {config.decoder.hidden, config.embedding});
}
} // namespace specferry::models::opt
