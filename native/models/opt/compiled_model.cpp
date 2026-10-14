#include "models/opt/layer.hpp"
#include "np101/memory_budget.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
using namespace np101;

namespace {
void read_ports(std::istream &file, std::vector<TensorSpec> &ports, std::vector<std::string> &names,
                std::vector<unsigned> &order, unsigned expected_count) {
  unsigned count = 0;
  file >> count;
  if (!file || count != expected_count) {
    throw std::invalid_argument("compiled OPT model port count differs");
  }
  order.resize(count);
  std::set<unsigned> seen;
  for (auto &index : order) {
    file >> index;
    if (!file || index >= count || !seen.insert(index).second) {
      throw std::invalid_argument("compiled OPT model requires a complete port permutation");
    }
  }
  for (unsigned index = 0; index < count; ++index) {
    std::string name, dtype;
    unsigned rank = 0;
    file >> std::quoted(name) >> dtype >> rank;
    if (!file || rank == 0 || rank > 3) {
      throw std::invalid_argument("compiled OPT model port rank differs");
    }
    const auto type = dtype == "F16"   ? DataType::Float16
                      : dtype == "F32" ? DataType::Float32
                                       : DataType::Int32;
    if (dtype != "F16" && dtype != "F32" && dtype != "I32") {
      throw std::invalid_argument("compiled OPT model port dtype differs");
    }
    ops::Shape shape(rank);
    for (auto &dimension : shape) {
      file >> dimension;
      if (!file || !dimension) {
        throw std::invalid_argument("compiled OPT model port dimension differs");
      }
    }
    ports.push_back({type, shape});
    names.push_back(std::move(name));
  }
}
} // namespace

void validate_compiled_full_model(const CompiledFullModel &compiled, const WeightStore &weights,
                                  const ModelConfig &config) {
  const auto &decoder = config.decoder;
  if (decoder.layers != 24 || decoder.hidden != 1024 || decoder.heads != 16 ||
      decoder.intermediate != 4096 || decoder.epsilon != 1e-5f ||
      (decoder.capacity != 64 && decoder.capacity != 128) || config.vocabulary != 50272 ||
      config.embedding != 512 || config.position_offset != 2 ||
      compiled.graph.graph_version != std::array<unsigned, 3>{1, 1, 35}) {
    throw std::invalid_argument("compiled full OPT model geometry/version differs");
  }
  std::vector<std::pair<std::string, TensorSpec>> inputs{
      {"token", {DataType::Int32, {1, 1}}},
      {"position", {DataType::Int32, {1, 1}}},
      {"kv_indices", {DataType::Int32, {2, 16}}},
      {"attention_mask", {DataType::Float32, {decoder.capacity, 1}}}};
  std::vector<std::pair<std::string, TensorSpec>> outputs{
      {"logits", {DataType::Float16, {50272, 1}}}, {"selected", {DataType::Int32, {1}}}};
  const TensorSpec cache{DataType::Float32, {64, decoder.capacity, 16}};
  for (unsigned layer = 0; layer < 24; ++layer) {
    for (const std::string part : {"key", "value"}) {
      const auto prefix = "layer" + std::to_string(layer) + ".";
      inputs.push_back({prefix + "past_" + part, cache});
      outputs.push_back({prefix + "present_" + part, cache});
    }
  }
  const auto validate = [](const auto &ports, const auto &names, const auto &order,
                           const auto &expected) {
    if (ports.size() != expected.size() || names.size() != expected.size() ||
        order.size() != expected.size()) {
      throw std::invalid_argument("compiled full OPT model port count differs");
    }
    std::set<unsigned> seen;
    for (unsigned index = 0; index < order.size(); ++index) {
      const auto slot = order[index];
      if (slot >= ports.size() || !seen.insert(slot).second ||
          names[slot] != expected[index].first || ports[slot].type != expected[index].second.type ||
          ports[slot].shape != expected[index].second.shape) {
        throw std::invalid_argument("compiled full OPT model semantic port differs");
      }
    }
  };
  validate(compiled.graph.inputs, compiled.input_names, compiled.input_order, inputs);
  validate(compiled.graph.outputs, compiled.output_names, compiled.output_order, outputs);
  std::size_t payload = 0;
  if (compiled.weight_sha256.size() != weights.records().size()) {
    throw std::invalid_argument("compiled full OPT model source count differs");
  }
  for (const auto &record : weights.records()) {
    const auto identity = compiled.weight_sha256.find(record.name);
    if (identity == compiled.weight_sha256.end() || identity->second != record.sha256) {
      throw std::invalid_argument("compiled full OPT model source differs: " + record.name);
    }
    payload += record.bytes * (record.name.find("layer_norm") != std::string::npos ? 2 : 1);
  }
  if (compiled.graph.constant_bytes < payload ||
      compiled.graph.constant_bytes > segment_payload_limit) {
    throw std::invalid_argument("compiled full OPT model constant payload is not accounted");
  }
}

CompiledModelPackage load_compiled_model(const std::filesystem::path &package,
                                         const std::filesystem::path &deployment,
                                         const ModelConfig &config, const WeightStore &weights) {
  const auto root = std::filesystem::canonical(package);
  std::ifstream file(root / "compiled-model.txt");
  std::string magic, kind, index_digest, deployment_digest, driver, name, digest, extra;
  unsigned version = 0, capacity = 0, count = 0;
  file >> magic >> version >> kind >> capacity >> index_digest >> deployment_digest >> driver >>
      count;
  if (!file || magic != "specferry-opt-model" || version != 1 ||
      (kind != "decoders" && kind != "fullgraph") || capacity != config.decoder.capacity ||
      (capacity != 64 && capacity != 128) || count != 5 || driver != "F53A3E6EFBB7D0FD7D4617C" ||
      file_sha256(deployment / "weights.index") != index_digest ||
      file_sha256(deployment / "deployment-manifest.json") != deployment_digest) {
    throw std::invalid_argument("compiled OPT model deployment/header differs");
  }
  std::ifstream module("/sys/module/galcore/srcversion");
  std::string loaded_driver;
  module >> loaded_driver;
  if (!module || loaded_driver != driver) {
    throw std::invalid_argument("compiled OPT model loaded driver differs");
  }
  std::set<std::string> libraries{"libovxlib.so", "libOpenVX.so", "libGAL.so", "libArchModelSw.so",
                                  "libNNArchPerf.so"};
  for (unsigned index = 0; index < 5; ++index) {
    file >> name >> digest;
    if (!file || !libraries.erase(name) ||
        file_sha256(std::filesystem::path("/usr/lib/ljmicro") / name) != digest) {
      throw std::invalid_argument("compiled OPT model SDK identity differs");
    }
  }
  file >> count;
  if (!file || count != (kind == "decoders" ? 24u : 1u) || config.decoder.layers != 24) {
    throw std::invalid_argument("compiled OPT model requires complete ordered layers");
  }
  CompiledModelPackage result;
  std::size_t constants = 0;
  for (unsigned index = 0; index < count; ++index) {
    CompiledFullModel artifact;
    file >> std::quoted(name) >> digest >> artifact.graph.constant_bytes;
    for (auto &part : artifact.graph.graph_version) {
      file >> part;
    }
    const auto binary = std::filesystem::canonical(root / name);
    if (!file || binary.parent_path() != root || !std::filesystem::file_size(binary) ||
        std::filesystem::file_size(binary) > artifact.graph.constant_bytes ||
        artifact.graph.constant_bytes > segment_payload_limit - constants ||
        file_sha256(binary) != digest) {
      throw std::invalid_argument("compiled OPT model NBG path/hash/budget differs");
    }
    artifact.graph.path = binary.string();
    constants += artifact.graph.constant_bytes;
    read_ports(file, artifact.graph.inputs, artifact.input_names, artifact.input_order,
               kind == "decoders" ? 5 : 52);
    read_ports(file, artifact.graph.outputs, artifact.output_names, artifact.output_order,
               kind == "decoders" ? 3 : 50);
    unsigned sources = 0;
    file >> sources;
    if (!file || sources != (kind == "decoders" ? 16u : weights.records().size())) {
      throw std::invalid_argument("compiled OPT model source count differs");
    }
    for (unsigned source = 0; source < sources; ++source) {
      file >> std::quoted(name) >> digest;
      if (!file || !artifact.weight_sha256.emplace(name, digest).second) {
        throw std::invalid_argument("compiled OPT model source identities differ");
      }
    }
    if (kind == "fullgraph") {
      validate_compiled_full_model(artifact, weights, config);
      result.full = std::move(artifact);
    } else {
      CompiledDecoder decoder;
      decoder.graph = std::move(artifact.graph);
      decoder.layer = index;
      std::copy(artifact.input_order.begin(), artifact.input_order.end(),
                decoder.input_order.begin());
      std::copy(artifact.output_order.begin(), artifact.output_order.end(),
                decoder.output_order.begin());
      const auto prefix = config.decoder.prefix(index);
      for (const auto &[key, hash] : artifact.weight_sha256) {
        if (key.compare(0, prefix.size(), prefix) != 0) {
          throw std::invalid_argument("compiled OPT decoder source layer differs");
        }
        decoder.weight_sha256.emplace(key.substr(prefix.size()), hash);
      }
      validate_compiled_decoder(decoder, weights, config.decoder, index);
      result.decoders.push_back(std::move(decoder));
    }
  }
  if (file >> extra) {
    throw std::invalid_argument("trailing compiled OPT model package data");
  }
  return result;
}
} // namespace specferry::models::opt
