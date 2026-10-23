#include "models/qwen3_5/compiled.hpp"
#include "np101/memory_budget.hpp"
#include "np101/sdk_identity.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>

namespace specferry::models::qwen3_5 {
using namespace np101;

namespace {
void validate_block(const CompiledBlock &artifact, const WeightStore &weights, const Config &config,
                    unsigned layer, CompiledBlockKind kind) {
  const bool mixer = kind == CompiledBlockKind::DeltaNet;
  if (config.hidden != 1024 || config.epsilon != 1e-6f || layer >= config.layer_types.size() ||
      artifact.kind != kind || artifact.layer != layer ||
      (artifact.includes_input_norm && !mixer) ||
      artifact.weight_sha256.size() != (mixer ? 9u + artifact.includes_input_norm : 4u) ||
      (artifact.graph.graph_version != std::array<unsigned, 3>{1, 1, 34} &&
       artifact.graph.graph_version != std::array<unsigned, 3>{1, 1, 35})) {
    throw std::invalid_argument("compiled Qwen block geometry/version/source count differs");
  }
  if (mixer &&
      (config.delta.epsilon != 1e-6f || config.delta.heads != 16 || config.delta.key_dim != 128 ||
       config.delta.value_dim != 128 || config.delta.convolution_width != 4 ||
       config.mixer(layer) != MixerKind::DeltaNet)) {
    throw std::invalid_argument("compiled Qwen mixer geometry differs");
  }
  for (bool output : {false, true}) {
    const auto &ports = output ? artifact.graph.outputs : artifact.graph.inputs;
    const auto &order = output ? artifact.output_order : artifact.input_order;
    const std::vector<TensorSpec> expected =
        mixer ? std::vector<TensorSpec>{config.hidden_spec(), config.delta.recurrent(),
                                        config.delta.convolution()}
              : std::vector<TensorSpec>(output ? 1 : 2, config.hidden_spec());
    std::set<unsigned> permutation;
    for (unsigned index = 0; index < expected.size(); ++index) {
      permutation.insert(index);
    }
    if (ports.size() != expected.size() || order.size() != expected.size() ||
        std::set<unsigned>(order.begin(), order.end()) != permutation) {
      throw std::invalid_argument("compiled Qwen block port order differs");
    }
    for (unsigned index = 0; index < expected.size(); ++index) {
      if (ports[order[index]].shape != expected[index].shape ||
          ports[order[index]].type != expected[index].type) {
        throw std::invalid_argument("compiled Qwen block precision/shape differs");
      }
    }
  }
  const auto prefix = config.layer_prefix(layer);
  std::size_t payload = 0, count = 0;
  for (const auto &record : weights.records()) {
    const bool selected =
        mixer
            ? record.name.compare(0, prefix.size() + 12, prefix + "linear_attn.") == 0 ||
                  (artifact.includes_input_norm && record.name == prefix + "input_layernorm.weight")
            : record.name == prefix + "post_attention_layernorm.weight" ||
                  record.name.compare(0, prefix.size() + 4, prefix + "mlp.") == 0;
    if (selected) {
      const auto source = artifact.weight_sha256.find(record.name);
      if (source == artifact.weight_sha256.end() || source->second != record.sha256) {
        throw std::invalid_argument("compiled Qwen block source weight differs");
      }
      payload += record.bytes;
      ++count;
    }
  }
  if (count != (mixer ? 9u + artifact.includes_input_norm : 4u) ||
      artifact.graph.constant_bytes < payload ||
      artifact.graph.constant_bytes > segment_payload_limit) {
    throw std::invalid_argument("compiled Qwen block constant budget differs");
  }
}

std::map<unsigned, CompiledBlock> load_blocks(const std::filesystem::path &package,
                                              const WeightStore &weights, const Config &config,
                                              CompiledBlockKind kind) {
  const bool mixer = kind == CompiledBlockKind::DeltaNet;
  const auto label = mixer ? "delta" : "feed-forward";
  const auto root = std::filesystem::canonical(package);
  std::ifstream file(root / (mixer ? "compiled-mixers.txt" : "compiled-feed-forward.txt"));
  std::string magic, driver, name, digest, extra;
  unsigned version = 0, count = 0;
  file >> magic >> version >> driver >> count;
  if (!file || magic != std::string("specferry-qwen-") + label ||
      (version != 1 && !(mixer && version == 2)) || driver != "F53A3E6EFBB7D0FD7D4617C") {
    throw std::invalid_argument("compiled Qwen block package/driver differs");
  }
  verify_compiled_sdk(file, driver, count);
  file >> count;
  if (!file || count == 0 || count > (mixer ? 18u : 24u)) {
    throw std::invalid_argument("compiled Qwen block package count differs");
  }
  std::map<unsigned, CompiledBlock> result;
  std::size_t total = 0;
  for (unsigned entry = 0; entry < count; ++entry) {
    CompiledBlock artifact;
    artifact.kind = kind;
    file >> artifact.layer >> std::quoted(name) >> digest >> artifact.graph.constant_bytes;
    for (auto &part : artifact.graph.graph_version) {
      file >> part;
    }
    const auto binary = std::filesystem::canonical(root / name);
    if (!file || binary.parent_path() != root || !std::filesystem::file_size(binary) ||
        std::filesystem::file_size(binary) > artifact.graph.constant_bytes ||
        artifact.graph.constant_bytes > segment_payload_limit - total ||
        file_sha256(binary) != digest) {
      throw std::invalid_argument("compiled Qwen block NBG identity/budget differs");
    }
    total += artifact.graph.constant_bytes;
    artifact.graph.path = binary.string();
    if (mixer && version == 2) {
      unsigned normalized = 0;
      file >> normalized;
      if (!file || normalized > 1) {
        throw std::invalid_argument("invalid DeltaNet input normalization contract");
      }
      artifact.includes_input_norm = normalized;
    }
    for (bool output : {false, true}) {
      auto &order = output ? artifact.output_order : artifact.input_order;
      auto &ports = output ? artifact.graph.outputs : artifact.graph.inputs;
      const auto port_count = mixer ? 3u : (output ? 1u : 2u);
      order.resize(port_count);
      for (auto &index : order) {
        file >> index;
      }
      for (unsigned port = 0; port < port_count; ++port) {
        std::string type;
        unsigned rank = 0;
        file >> type >> rank;
        if (!file || (type != "F16" && type != "F32") || rank == 0 || rank > 3) {
          throw std::invalid_argument("compiled Qwen block port declaration differs");
        }
        ops::Shape shape(rank);
        for (auto &dimension : shape) {
          file >> dimension;
          if (!file || !dimension) {
            throw std::invalid_argument("compiled Qwen block port dimension differs");
          }
        }
        ports.push_back({type == "F16" ? DataType::Float16 : DataType::Float32, shape});
      }
    }
    for (unsigned source = 0; source < (mixer ? 9u + artifact.includes_input_norm : 4u); ++source) {
      file >> std::quoted(name) >> digest;
      if (!file || !artifact.weight_sha256.emplace(name, digest).second) {
        throw std::invalid_argument("compiled Qwen block source declarations differ");
      }
    }
    validate_block(artifact, weights, config, artifact.layer, kind);
    if (!result.emplace(artifact.layer, artifact).second) {
      throw std::invalid_argument("compiled Qwen block repeats a layer");
    }
  }
  if (file >> extra) {
    throw std::invalid_argument("trailing compiled Qwen block package data");
  }
  return result;
}
} // namespace

void validate_compiled_mixer(const CompiledBlock &artifact, const WeightStore &weights,
                             const Config &config, unsigned layer) {
  validate_block(artifact, weights, config, layer, CompiledBlockKind::DeltaNet);
}

void validate_compiled_feed_forward(const CompiledBlock &artifact, const WeightStore &weights,
                                    const Config &config, unsigned layer) {
  validate_block(artifact, weights, config, layer, CompiledBlockKind::FeedForward);
}

std::map<unsigned, CompiledBlock> load_compiled_mixers(const std::filesystem::path &package,
                                                       const WeightStore &weights,
                                                       const Config &config) {
  return load_blocks(package, weights, config, CompiledBlockKind::DeltaNet);
}

std::map<unsigned, CompiledBlock> load_compiled_feed_forward(const std::filesystem::path &package,
                                                             const WeightStore &weights,
                                                             const Config &config) {
  return load_blocks(package, weights, config, CompiledBlockKind::FeedForward);
}

} // namespace specferry::models::qwen3_5
