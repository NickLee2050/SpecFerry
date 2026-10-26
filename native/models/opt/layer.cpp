#include "models/opt/layer.hpp"
#include "np101/memory_budget.hpp"
#include "np101/sdk_identity.hpp"
#include "vsi_nn_pub.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

namespace {
constexpr auto f16 = DataType::Float16;

Tensor linear(GraphBuilder &g, const WeightStore &weights, Tensor input, const std::string &name,
              bool split_output_channels = false, LinearBackend backend = LinearBackend::MatMul) {
  return g.linear(input, weights, weights.find(name + ".weight"), weights.find(name + ".bias"),
                  split_output_channels ? 2u : 1u, backend);
}

Tensor norm(GraphBuilder &g, const WeightStore &weights, Tensor input, const std::string &name,
            float epsilon) {
  const auto width = input.spec.shape[0];
  const auto affine = [&](const std::string &suffix) {
    const auto &record = weights.find(name + suffix);
    if (record.spec.type != f16 || record.spec.shape != Shape{width}) {
      throw std::invalid_argument("unexpected OPT layer norm weight: " + record.name);
    }
    return g.weight_chunk(weights, record, {DataType::Float32, {width, 1}}, 0);
  };
  auto scale = affine(".weight");
  auto bias = affine(".bias");
  return g.layer_norm(input, scale, bias, epsilon);
}
} // namespace

std::vector<CompiledFfn> load_compiled_ffn(const std::filesystem::path &package,
                                           const std::filesystem::path &deployment,
                                           const ModelConfig &config, const WeightStore &weights,
                                           CompiledHead &head,
                                           std::vector<CompiledProjections> *projections) {
  if (config.decoder.layers != 24 ||
      (config.decoder.capacity != 64 && config.decoder.capacity != 128 &&
       config.decoder.capacity != 256) ||
      config.decoder.heads != 16 || config.embedding != 512 || config.vocabulary != 50272) {
    throw std::invalid_argument("compiled FFN package requires OPT-350m C64/C128/C256 geometry");
  }
  const auto root = std::filesystem::canonical(package);
  std::ifstream file(root / "compiled-ffn.txt");
  std::string magic, index_digest, deployment_digest, driver, digest, name, extra;
  unsigned version = 0, count = 0;
  file >> magic >> version >> index_digest >> deployment_digest >> driver >> count;
  if (!file || magic != "specferry-opt-ffn-finish" || version < 1 || version > 3 ||
      file_sha256(deployment / "weights.index") != index_digest ||
      file_sha256(deployment / "deployment-manifest.json") != deployment_digest) {
    throw std::invalid_argument("compiled FFN package deployment/header mismatch");
  }
  verify_compiled_sdk(file, driver, count);
  file >> count;
  if (!file || count != 24) {
    throw std::invalid_argument("compiled FFN package requires 24 ordered layers");
  }
  std::vector<CompiledFfn> result;
  std::size_t total = 0;
  for (unsigned layer = 0; layer < count; ++layer) {
    CompiledFfn artifact;
    file >> artifact.layer >> std::quoted(name) >> digest >> artifact.graph.constant_bytes;
    for (auto &part : artifact.graph.graph_version) {
      file >> part;
    }
    for (auto &part : artifact.weight_sha256) {
      file >> part;
    }
    for (auto &part : artifact.final_norm_sha256) {
      file >> part;
    }
    if (!file || artifact.layer != layer || !artifact.includes_final_norm() ||
        artifact.graph.graph_version != std::array<unsigned, 3>{1, 1, 34}) {
      throw std::invalid_argument("compiled FFN package layer/version mismatch");
    }
    const auto binary = std::filesystem::canonical(root / name);
    if (binary.parent_path() != root || !std::filesystem::file_size(binary) ||
        std::filesystem::file_size(binary) > artifact.graph.constant_bytes ||
        file_sha256(binary) != digest) {
      throw std::invalid_argument("compiled FFN package NBG path/checksum mismatch");
    }
    artifact.graph.path = binary.string();
    artifact.graph.inputs = artifact.graph.outputs = {{f16, {1024, 1}}};
    validate_compiled_ffn(artifact, weights, config.decoder, layer);
    if (artifact.graph.constant_bytes > segment_payload_limit - total) {
      throw std::invalid_argument("compiled FFN package exceeds the constant payload limit");
    }
    total += artifact.graph.constant_bytes;
    result.push_back(std::move(artifact));
  }
  unsigned head_count = version == 2 ? 1 : 0;
  if (version == 3) {
    file >> head_count;
    if (!file || head_count > 1 || !projections) {
      throw std::invalid_argument("compiled projection package requires an explicit receiver");
    }
  }
  if (head_count) {
    file >> std::quoted(name) >> digest >> head.graph.constant_bytes;
    for (auto &part : head.graph.graph_version) {
      file >> part;
    }
    for (auto &part : head.weight_sha256) {
      file >> part;
    }
    const auto binary = std::filesystem::canonical(root / name);
    if (!file || binary.parent_path() != root || !std::filesystem::file_size(binary) ||
        std::filesystem::file_size(binary) > head.graph.constant_bytes ||
        file_sha256(binary) != digest ||
        head.graph.constant_bytes > segment_payload_limit - total) {
      throw std::invalid_argument("compiled head package path/checksum/budget mismatch");
    }
    head.graph.path = binary.string();
    head.graph.inputs = {{f16, {1024, 1}}};
    head.graph.outputs = {{f16, {50272, 1}}};
    validate_compiled_head(head, weights, config);
    total += head.graph.constant_bytes;
  }
  if (version == 3) {
    file >> count;
    if (!file || count != config.decoder.layers) {
      throw std::invalid_argument("compiled projections require all ordered decoder layers");
    }
    for (unsigned layer = 0; layer < count; ++layer) {
      CompiledProjections artifact;
      file >> artifact.layer;
      for (auto *graph : {&artifact.qkv, &artifact.output}) {
        file >> std::quoted(name) >> digest >> graph->constant_bytes;
        for (auto &part : graph->graph_version) {
          file >> part;
        }
        const auto binary = std::filesystem::canonical(root / name);
        if (!file || binary.parent_path() != root || !std::filesystem::file_size(binary) ||
            std::filesystem::file_size(binary) > graph->constant_bytes ||
            file_sha256(binary) != digest ||
            graph->constant_bytes > segment_payload_limit - total) {
          throw std::invalid_argument("compiled projection path/checksum/budget mismatch");
        }
        graph->path = binary.string();
        graph->inputs = {{f16, {config.decoder.hidden, 1}}};
        graph->outputs.assign(graph == &artifact.qkv ? 3 : 1, graph->inputs.front());
        total += graph->constant_bytes;
      }
      for (auto &port : artifact.qkv_order) {
        file >> port;
      }
      for (auto &hash : artifact.weight_sha256) {
        file >> hash;
      }
      if (!file) {
        throw std::invalid_argument("incomplete compiled projection identities");
      }
      validate_compiled_projections(artifact, weights, config.decoder, layer);
      projections->push_back(std::move(artifact));
    }
  }
  if (file >> extra) {
    throw std::invalid_argument("trailing compiled FFN package data");
  }
  return result;
}

std::vector<CompiledFfn> load_compiled_prefill(const std::filesystem::path &package,
                                               const std::filesystem::path &deployment,
                                               const ModelConfig &config,
                                               const WeightStore &weights) {
  const auto root = std::filesystem::canonical(package);
  std::ifstream file(root / "compiled-prefill.txt");
  std::string magic, m1_digest, index_digest, deployment_digest, name, digest, extra;
  unsigned version = 0, tokens = 0, count = 0;
  file >> magic >> version >> m1_digest >> index_digest >> deployment_digest >> tokens >> count;
  const bool full_model = version == 2;
  if (!file || magic != "specferry-opt-prefill" || (version != 1 && version != 2) || tokens != 16 ||
      count != (full_model ? 18u : 20u) || config.decoder.layers != 24 ||
      (config.decoder.capacity != 64 && config.decoder.capacity != 256) ||
      (full_model && config.decoder.capacity != 64) ||
      file_sha256(root / (full_model ? "compiled-model.txt" : "compiled-ffn.txt")) != m1_digest ||
      file_sha256(deployment / "weights.index") != index_digest ||
      file_sha256(deployment / "deployment-manifest.json") != deployment_digest) {
    throw std::invalid_argument("M16 FFN package source/M1/geometry differs");
  }
  std::vector<CompiledFfn> result(24);
  for (unsigned layer = 0; layer < result.size(); ++layer) {
    result[layer].layer = layer;
  }
  std::size_t total = 0;
  for (unsigned layer = full_model ? 6u : 4u; layer < result.size(); ++layer) {
    auto &artifact = result[layer];
    file >> artifact.layer >> std::quoted(name) >> digest >> artifact.graph.constant_bytes;
    for (auto &part : artifact.graph.graph_version) {
      file >> part;
    }
    for (auto &hash : artifact.weight_sha256) {
      file >> hash;
    }
    const auto binary = std::filesystem::canonical(root / name);
    if (!file || artifact.layer != layer || binary.parent_path() != root ||
        !std::filesystem::file_size(binary) || file_sha256(binary) != digest ||
        artifact.graph.graph_version != std::array<unsigned, 3>{1, 1, 34} ||
        std::filesystem::file_size(binary) > artifact.graph.constant_bytes ||
        artifact.graph.constant_bytes > segment_payload_limit - total) {
      throw std::invalid_argument("M16 FFN package path/hash/layer/budget differs");
    }
    artifact.graph.path = binary.string();
    artifact.graph.inputs = artifact.graph.outputs = {{f16, {1024, 16}}};
    validate_compiled_ffn(artifact, weights, config.decoder, layer, 16);
    total += artifact.graph.constant_bytes;
  }
  if (file >> extra) {
    throw std::invalid_argument("trailing M16 FFN package data");
  }
  return result;
}

std::vector<CompiledPrefillLayer>
load_compiled_block_prefill(const std::filesystem::path &package,
                            const std::filesystem::path &deployment, const ModelConfig &config,
                            const WeightStore &weights) {
  const auto root = std::filesystem::canonical(package);
  std::ifstream stream(root / "compiled-block-prefill.txt");
  std::string magic, model_digest, index_digest, deployment_digest, extra;
  unsigned version = 0, capacity = 0, tokens = 0, count = 0;
  stream >> magic >> version >> model_digest >> index_digest >> deployment_digest >> capacity >>
      tokens >> count;
  if (!stream || magic != "specferry-opt-compiled-block-prefill" || version != 1 ||
      (capacity != 64 && capacity != 128) || tokens != 16 || count != 24 ||
      config.decoder.capacity != capacity || config.decoder.layers != count ||
      config.decoder.hidden != 1024 || config.decoder.heads != 16 ||
      config.decoder.intermediate != 4096 || config.decoder.epsilon != 1e-5f ||
      file_sha256(root / "compiled-model.txt") != model_digest ||
      file_sha256(deployment / "weights.index") != index_digest ||
      file_sha256(deployment / "deployment-manifest.json") != deployment_digest) {
    throw std::invalid_argument("compiled block prefill source/geometry differs");
  }
  std::vector<CompiledPrefillLayer> result;
  std::vector<FileChecksum> checksums;
  std::size_t constants = 0;
  for (unsigned layer = 0; layer < count; ++layer) {
    CompiledPrefillLayer item;
    std::string name, digest;
    unsigned full = 0, identities = 0;
    stream >> item.layer >> full >> std::quoted(name) >> digest >> item.graph.constant_bytes;
    for (auto &part : item.graph.graph_version) {
      stream >> part;
    }
    stream >> identities;
    if (!stream || item.layer != layer || full > 1 || identities != (full ? 16u : 10u) ||
        item.graph.graph_version != std::array<unsigned, 3>{1, 1, 35}) {
      throw std::invalid_argument("compiled block prefill layer/version differs");
    }
    item.full_decoder = full;
    std::set<std::string> expected;
    for (const std::string projection : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                         "self_attn.out_proj", "self_attn_layer_norm"}) {
      for (const std::string suffix : {".weight", ".bias"}) {
        expected.insert(config.decoder.prefix(layer) + projection + suffix);
      }
    }
    if (full) {
      for (const std::string part : {"fc1", "fc2", "final_layer_norm"}) {
        for (const std::string suffix : {".weight", ".bias"}) {
          expected.insert(config.decoder.prefix(layer) + part + suffix);
        }
      }
    }
    std::size_t source_bytes = 0;
    for (unsigned index = 0; index < identities; ++index) {
      std::string weight_name, weight_digest;
      stream >> weight_name >> weight_digest;
      if (!stream || !expected.erase(weight_name)) {
        throw std::invalid_argument("compiled block prefill weight name differs");
      }
      const auto &weight = weights.find(weight_name);
      if (weight.sha256 != weight_digest) {
        throw std::invalid_argument("compiled block prefill source weight differs");
      }
      source_bytes += weight.bytes;
    }
    const auto binary = std::filesystem::canonical(root / name);
    if (!expected.empty() || binary.parent_path() != root || !std::filesystem::file_size(binary) ||
        std::filesystem::file_size(binary) > item.graph.constant_bytes ||
        source_bytes > item.graph.constant_bytes ||
        item.graph.constant_bytes > segment_payload_limit - constants) {
      throw std::invalid_argument("compiled block prefill path/hash/budget differs");
    }
    item.graph.path = binary.string();
    checksums.push_back({binary, digest});
    const TensorSpec hidden{f16, {1024, 16}}, state{DataType::Float32, {64, capacity, 16}};
    item.graph.inputs = {
        hidden, state, state, {DataType::Int32, {1}}, {DataType::Float32, {capacity, 16}}};
    item.graph.outputs = {hidden, state, state};
    constants += item.graph.constant_bytes;
    result.push_back(std::move(item));
  }
  if (stream >> extra) {
    throw std::invalid_argument("trailing compiled block prefill data");
  }
  verify_files(checksums);
  return result;
}

void validate_compiled_projections(const CompiledProjections &projections,
                                   const WeightStore &weights, const Config &config,
                                   unsigned layer) {
  const std::set<unsigned> order(projections.qkv_order.begin(), projections.qkv_order.end());
  if (config.hidden != 1024 || config.heads != 16 || projections.layer != layer ||
      order != std::set<unsigned>{0, 1, 2}) {
    throw std::invalid_argument("compiled projection geometry/order differs");
  }
  for (const auto *graph : {&projections.qkv, &projections.output}) {
    if (graph->graph_version != std::array<unsigned, 3>{1, 1, 34} || graph->inputs.size() != 1 ||
        graph->outputs.size() != (graph == &projections.qkv ? 3u : 1u)) {
      throw std::invalid_argument("compiled projection ports/version differ");
    }
    for (const auto &ports : {graph->inputs, graph->outputs}) {
      for (const auto &port : ports) {
        if (port.type != f16 || port.shape != Shape{1024, 1}) {
          throw std::invalid_argument("compiled projection requires FP16 M1 ports");
        }
      }
    }
  }
  std::size_t index = 0, qkv_bytes = 0, output_bytes = 0;
  for (const std::string part : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
    for (const std::string suffix : {".weight", ".bias"}) {
      const auto &record = weights.find(config.prefix(layer) + "self_attn." + part + suffix);
      if (projections.weight_sha256.at(index++) != record.sha256) {
        throw std::invalid_argument("compiled projection source weight differs");
      }
      (part == "out_proj" ? output_bytes : qkv_bytes) += record.bytes;
    }
  }
  if (projections.qkv.constant_bytes < qkv_bytes ||
      projections.output.constant_bytes < output_bytes) {
    throw std::invalid_argument("compiled projection constants are not accounted");
  }
}

void validate_compiled_head(const CompiledHead &head, const WeightStore &weights,
                            const ModelConfig &config) {
  if (config.decoder.hidden != 1024 || config.embedding != 512 || config.vocabulary != 50272 ||
      head.graph.graph_version != std::array<unsigned, 3>{1, 1, 34} ||
      head.graph.inputs.size() != 1 || head.graph.outputs.size() != 1 ||
      head.graph.inputs[0].type != f16 || head.graph.inputs[0].shape != Shape{1024, 1} ||
      head.graph.outputs[0].type != f16 || head.graph.outputs[0].shape != Shape{50272, 1}) {
    throw std::invalid_argument("compiled head differs from the accepted FP16 OPT contract");
  }
  std::size_t payload = 0;
  unsigned index = 0;
  for (const std::string name : {"decoder.project_out.weight", "decoder.embed_tokens.weight"}) {
    const auto &record = weights.find(name);
    if (head.weight_sha256[index++] != record.sha256) {
      throw std::invalid_argument("compiled head source weights differ");
    }
    payload += record.bytes;
  }
  if (head.graph.constant_bytes < payload || head.graph.constant_bytes > segment_payload_limit) {
    throw std::invalid_argument("compiled head constant payload is not accounted");
  }
}

std::vector<TensorSpec> compiled_decoder_inputs(const Config &config) {
  const auto dimension = config.kv_spec().head_dim;
  return {{f16, {config.hidden, 1}},
          {DataType::Float32, {dimension, config.capacity, config.heads}},
          {DataType::Float32, {dimension, config.capacity, config.heads}},
          {DataType::Int32, {2, config.heads}},
          {DataType::Float32, {config.capacity, 1}}};
}

void validate_compiled_decoder(const CompiledDecoder &compiled, const WeightStore &weights,
                               const Config &config, unsigned layer) {
  // Only this export geometry and precision policy have hardware evidence.
  if (config.hidden != 1024 || config.intermediate != 4096 || config.heads != 16 ||
      (config.capacity != 64 && config.capacity != 128) || config.epsilon != 1e-5f ||
      compiled.layer != layer || compiled.weight_sha256.size() != 16 ||
      (compiled.graph.graph_version != std::array<unsigned, 3>{1, 1, 34} &&
       compiled.graph.graph_version != std::array<unsigned, 3>{1, 1, 35})) {
    throw std::invalid_argument("compiled OPT decoder differs from the accepted export contract");
  }
  const auto inputs = compiled_decoder_inputs(config);
  const std::vector<TensorSpec> outputs{inputs[0], inputs[1], inputs[2]};
  const auto matches = [](const auto &actual, const auto &expected, const auto &order) {
    if (actual.size() != expected.size()) {
      return false;
    }
    const std::set<unsigned> seen(order.begin(), order.end());
    if (seen.size() != actual.size()) {
      return false;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (order[i] >= actual.size() || actual[order[i]].type != expected[i].type ||
          actual[order[i]].shape != expected[i].shape) {
        return false;
      }
    }
    return true;
  };
  if (!matches(compiled.graph.inputs, inputs, compiled.input_order) ||
      !matches(compiled.graph.outputs, outputs, compiled.output_order)) {
    throw std::invalid_argument("compiled OPT decoder port contract mismatch");
  }
  std::size_t payload = 0;
  for (const std::string name :
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
        "self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"}) {
    for (const std::string part : {"weight", "bias"}) {
      const auto key = name + "." + part;
      const auto &record = weights.find(config.prefix(layer) + key);
      const auto identity = compiled.weight_sha256.find(key);
      if (identity == compiled.weight_sha256.end() || identity->second != record.sha256) {
        throw std::invalid_argument("compiled OPT decoder source weight differs: " + record.name);
      }
      // Export widens normalization affine parameters to FP32 once.
      payload += record.spec.bytes() * (name.find("layer_norm") != std::string::npos ? 2 : 1);
    }
  }
  if (compiled.graph.constant_bytes < payload ||
      compiled.graph.constant_bytes > segment_payload_limit) {
    throw std::invalid_argument("compiled OPT decoder does not account for its constant payload");
  }
}

void validate_compiled_ffn(const CompiledFfn &compiled, const WeightStore &weights,
                           const Config &config, unsigned layer, unsigned tokens) {
  const TensorSpec port{f16, {config.hidden, tokens}};
  const auto matches = [&](const std::vector<TensorSpec> &ports) {
    return ports.size() == 1 && ports.front().type == port.type &&
           ports.front().shape == port.shape;
  };
  if ((tokens != 1 && tokens != 16) || compiled.layer != layer || !matches(compiled.graph.inputs) ||
      !matches(compiled.graph.outputs)) {
    throw std::invalid_argument("compiled OPT FFN requires matching layer and FP16 block ports");
  }
  const std::array<std::string, 4> names{"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"};
  std::size_t payload = 0;
  for (std::size_t index = 0; index < names.size(); ++index) {
    const auto &weight = weights.find(config.prefix(layer) + names[index]);
    if (compiled.weight_sha256[index] != weight.sha256) {
      throw std::invalid_argument("compiled OPT FFN source weight differs: " + weight.name);
    }
    payload += weight.spec.bytes();
  }
  if (compiled.final_norm_sha256[0].empty() != compiled.final_norm_sha256[1].empty()) {
    throw std::invalid_argument("compiled FFN normalization identities are incomplete");
  }
  if (compiled.includes_final_norm()) {
    if (tokens != 1 || config.hidden != 1024 || config.intermediate != 4096 ||
        config.epsilon != 1e-5f) {
      throw std::invalid_argument("compiled FFN finish requires the fixed M1 OPT export contract");
    }
    for (unsigned index = 0; index < 2; ++index) {
      const auto &weight =
          weights.find(config.prefix(layer) + "final_layer_norm." + (index ? "bias" : "weight"));
      if (weight.spec.type != f16 || weight.spec.shape != Shape{config.hidden} ||
          weight.sha256 != compiled.final_norm_sha256[index]) {
        throw std::invalid_argument("compiled FFN normalization source differs: " + weight.name);
      }
      // The mixed precision package widens the checkpoint affine to FP32.
      payload += 2 * weight.spec.bytes();
    }
  }
  // Compiled packages can include padded or packed constants beyond the source weights.
  if (compiled.graph.constant_bytes < payload ||
      compiled.graph.constant_bytes > segment_payload_limit) {
    throw std::invalid_argument("compiled OPT FFN constant payload does not cover source weights");
  }
}

Projections build_projections(GraphBuilder &g, const WeightStore &weights, const Config &config,
                              unsigned layer, Tensor hidden, LinearBackend backend) {
  const auto prefix = config.prefix(layer);
  auto query = linear(g, weights, hidden, prefix + "self_attn.q_proj", false, backend);
  query = g.binary(VSI_NN_OP_MULTIPLY, query,
                   g.scalar(1.0f / std::sqrt(float(config.kv_spec().head_dim)), f16));
  auto key = linear(g, weights, hidden, prefix + "self_attn.k_proj", false, backend);
  auto value = linear(g, weights, hidden, prefix + "self_attn.v_proj", false, backend);
  return {query, key, value};
}

AttentionTail build_attention_tail(GraphBuilder &g, const WeightStore &weights,
                                   const Config &config, unsigned layer, Tensor hidden,
                                   Tensor attended, LinearBackend backend) {
  const auto prefix = config.prefix(layer);
  auto mixer = linear(g, weights, attended, prefix + "self_attn.out_proj", false, backend);
  return build_attention_finish(g, weights, config, layer, hidden, mixer);
}

AttentionTail build_attention_finish(GraphBuilder &g, const WeightStore &weights,
                                     const Config &config, unsigned layer, Tensor hidden,
                                     Tensor mixer) {
  const auto prefix = config.prefix(layer);
  auto residual = g.binary(VSI_NN_OP_ADD, hidden, mixer);
  auto normalized = norm(g, weights, residual, prefix + "self_attn_layer_norm", config.epsilon);
  return {mixer, residual, normalized};
}

FfnFinish build_ffn_finish(GraphBuilder &g, const WeightStore &weights, const Config &config,
                           unsigned layer, Tensor normalized, Tensor mlp) {
  auto residual = g.binary(VSI_NN_OP_ADD, normalized, mlp);
  auto output =
      norm(g, weights, residual, config.prefix(layer) + "final_layer_norm", config.epsilon);
  return {residual, output};
}

std::map<std::string, Tensor> build_decoder_tail(GraphBuilder &g, const WeightStore &weights,
                                                 const Config &config, unsigned layer,
                                                 Tensor hidden, Tensor attended,
                                                 bool split_fc1_output) {
  const auto prefix = config.prefix(layer);
  const auto attention = build_attention_tail(g, weights, config, layer, hidden, attended);
  auto fc1 = linear(g, weights, attention.normalized, prefix + "fc1", split_fc1_output);
  auto activation = g.unary(VSI_NN_OP_RELU, fc1, f16);
  auto mlp = linear(g, weights, activation, prefix + "fc2");
  const auto finish = build_ffn_finish(g, weights, config, layer, attention.normalized, mlp);
  return {{"mixer", attention.mixer},
          {"attention_residual", attention.residual},
          {"attention_norm", attention.normalized},
          {"fc1", fc1},
          {"activation", activation},
          {"mlp", mlp},
          {"ffn_residual", finish.residual},
          {"output", finish.output}};
}
} // namespace specferry::models::opt
