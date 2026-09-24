#include "binary_io.hpp"
#include "models/opt/config.hpp"
#include "models/opt/graph_model.hpp"
#include "models/opt/layer.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
namespace opt = specferry::models::opt;
namespace fs = std::filesystem;
constexpr auto f16 = DataType::Float16;

std::vector<std::uint8_t> integers(const std::vector<std::int32_t> &values) {
  std::vector<std::uint8_t> result(values.size() * sizeof(std::int32_t));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

class Report {
public:
  const fs::path directory;
  unsigned checks = 0, failures = 0;

  explicit Report(const fs::path &output) : directory(output), rows_(output / "checks.tsv") {
    rows_ << "check\tpassed\telements\tnonzero\tmax_abs_error\n";
    phase("initialize");
  }

  void phase(const std::string &name, bool released = false) {
    std::ofstream stream(directory / "execution.json");
    stream << "{\"phase\":\"" << name << "\",\"released\":" << (released ? "true" : "false")
           << ",\"checks\":" << checks << ",\"failures\":" << failures << "}\n";
    if (!stream) {
      throw std::runtime_error("cannot save pipeline progress");
    }
    std::cout << name << std::endl;
  }

  bool compare(const std::string &name, const std::vector<std::uint8_t> &actual,
               const std::vector<std::uint8_t> &expected, DataType type, float atol = 0,
               float rtol = 0) {
    if (actual.size() != expected.size() || actual.size() % dtype_bytes(type)) {
      throw std::runtime_error("comparison shape mismatch: " + name);
    }
    bool passed = true;
    unsigned nonzero = 0;
    float maximum = 0;
    for (std::size_t offset = 0; offset < actual.size(); offset += dtype_bytes(type)) {
      if (type == f16) {
        std::uint16_t a, b;
        std::memcpy(&a, actual.data() + offset, 2);
        std::memcpy(&b, expected.data() + offset, 2);
        const auto value = vsi_nn_Fp16ToFp32(a), reference = vsi_nn_Fp16ToFp32(b);
        const auto error = std::abs(value - reference);
        maximum = std::max(maximum, error);
        nonzero += value != 0;
        passed &= std::isfinite(value) && error <= atol + rtol * std::abs(reference);
      } else if (type == DataType::Int32) {
        std::int32_t a, b;
        std::memcpy(&a, actual.data() + offset, 4);
        std::memcpy(&b, expected.data() + offset, 4);
        nonzero += a != 0;
        passed &= a == b;
        maximum = std::max(maximum, std::abs(float(a) - float(b)));
      } else {
        throw std::invalid_argument("diagnostic comparison requires FP16 or INT32");
      }
    }
    ++checks;
    failures += !passed;
    rows_ << name << '\t' << passed << '\t' << actual.size() / dtype_bytes(type) << '\t' << nonzero
          << '\t' << maximum << std::endl;
    std::ofstream bytes(directory / (name + ".bin"), std::ios::binary);
    if (!bytes.write(reinterpret_cast<const char *>(actual.data()), actual.size()) || !rows_) {
      throw std::runtime_error("cannot save pipeline diagnostic");
    }
    if (!passed) {
      std::cout << "FAIL " << name << ": max_abs_error=" << maximum << std::endl;
    }
    return passed;
  }

private:
  std::ofstream rows_;
};

void lookup_check(Context &context, const WeightStore &weights, const fs::path &fixture,
                  Report &report) {
  const auto config = opt::read_model_config(fixture);
  WeightBank bank(context);
  GraphBuilder g(context, 512, 256, &bank);
  auto controls = g.tensor({DataType::Int32, {3}});
  auto token = g.slice(controls, {0}, {1});
  auto position = g.slice(controls, {1}, {1});
  auto slot = g.slice(controls, {2}, {1});
  const auto &table = weights.find("decoder.embed_tokens.weight");
  std::vector<Tensor> blocks;
  for (unsigned first = 0; first < config.vocabulary; first += config.block_rows) {
    const auto rows = std::min(config.block_rows, config.vocabulary - first);
    blocks.push_back(g.weight_chunk(weights, table, {f16, {config.embedding, rows}},
                                    std::size_t(first) * config.embedding * 2));
  }
  const auto first_block = weights.read(table, 0, blocks.front().spec.bytes());
  report.compare("weight.before_setup", read_tensor(g.graph, blocks.front().id), first_block, f16);
  auto embedding = lookup_blocks(g, blocks, token);
  auto projected = g.project(embedding, weights, weights.find("decoder.project_in.weight"));
  const auto &positions = weights.find("decoder.embed_positions.weight");
  auto learned = g.gather(g.weight(weights, positions, positions.spec),
                          g.binary(VSI_NN_OP_ADD, position, g.integer(config.position_offset)), 1);
  auto hidden = g.binary(VSI_NN_OP_ADD, projected, learned);
  report.phase("setup");
  g.compile({controls}, {token, position, slot, embedding, hidden});
  report.compare("weight.after_setup", read_tensor(g.graph, blocks.front().id), first_block, f16);

  std::ifstream cases(fixture / "cases.txt");
  std::int32_t id, index;
  unsigned step = 0;
  while (cases >> id >> index) {
    config.validate_token(id);
    if (index < 0 || unsigned(index) >= config.decoder.capacity) {
      throw std::out_of_range("lookup fixture position exceeds capacity");
    }
    const auto prefix = "case." + std::to_string(step);

    report.phase(prefix);
    upload_tensor(g.graph, controls.id, integers({id, index, index}));
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(g.graph.get()); }),
          "run packed lookup graph");
    report.compare(prefix + ".token", read_tensor(g.graph, token.id), integers({id}),
                   DataType::Int32);
    report.compare(prefix + ".position", read_tensor(g.graph, position.id), integers({index}),
                   DataType::Int32);
    report.compare(prefix + ".slot", read_tensor(g.graph, slot.id), integers({index}),
                   DataType::Int32);
    report.compare(
        prefix + ".lookup", read_tensor(g.graph, embedding.id),
        weights.read(table, std::size_t(id) * config.embedding * 2, embedding.spec.bytes()), f16);
    report.compare(prefix + ".embedding", read_tensor(g.graph, hidden.id),
                   specferry::testing::read_bytes(
                       fixture, "embedding." + std::to_string(step) + ".bin", hidden.spec.bytes()),
                   f16, 0.08f, 0.02f);
    ++step;
    if (report.failures) {
      break;
    }
  }
  if (!step) {
    throw std::runtime_error("lookup fixture has no cases");
  }
  report.compare("weight.after_run", read_tensor(g.graph, blocks.front().id), first_block, f16);
  bank.verify();
  report.phase("release");
  g.graph.close();
  bank.close();
}

void layer_check(Context &context, const WeightStore &weights, const fs::path &fixture,
                 Report &report) {
  const auto config = opt::read_config(fixture / "components.txt");
  if (config.capacity < 2) {
    throw std::invalid_argument("layer diagnostic requires two cache slots");
  }
  opt::validate_weights(weights, config, {0});
  WeightBank bank(context);
  Graph storage(context, config.heads * 2, 0);
  GraphBuilder g(context, 2048, 1024, &bank);
  auto hidden = g.tensor(config.hidden_spec());
  auto controls = g.tensor({DataType::Int32, {2}});
  auto slot = g.slice(controls, {0}, {1});
  auto limit = g.slice(controls, {1}, {1});
  auto qkv = opt::build_projections(g, weights, config, 0, hidden);
  const unsigned dimension = config.kv_spec().head_dim;
  std::vector<Tensor> parents, probabilities;
  Tensor attended{};
  for (unsigned head = 0; head < config.heads; ++head) {
    std::vector<Tensor> caches;
    for (unsigned kind = 0; kind < 2; ++kind) {
      const TensorSpec shape{f16, {dimension, config.capacity}};
      auto id = add_tensor(storage, shape, false, std::vector<std::uint8_t>(shape.bytes()));
      parents.push_back({id, shape});
      auto parent = g.bind({storage, id}, shape);
      auto column = g.slice(kind ? qkv.value : qkv.key, {head * dimension, 0}, {dimension, 1});
      auto updated = indexed_append(g, g.reshape(column, {dimension, 1}), slot, parent);
      caches.push_back(storage_reshape(g, updated, {dimension, config.capacity, 1}));
    }
    auto query = g.slice(qkv.query, {head * dimension, 0}, {dimension, 1});
    auto attention =
        attention_core(g, g.reshape(query, {dimension, 1, 1}), caches[0], caches[1], limit, 1);
    probabilities.push_back(attention.probabilities);
    auto current = g.reshape(attention.attended, {dimension, 1});
    attended = head ? g.concat(attended, current) : current;
  }
  auto outputs = opt::build_decoder_tail(g, weights, config, 0, hidden, attended);
  outputs.emplace("query", qkv.query);
  outputs.emplace("key", qkv.key);
  outputs.emplace("value", qkv.value);
  report.phase("setup");
  // Match production: only the final value is a graph output. Read ordinary
  // intermediate tensors diagnostically after execution, without extra nodes.
  g.compile({hidden, controls}, {outputs.at("output")});

  for (unsigned step = 0; step < 2; ++step) {
    const auto label = "step." + std::to_string(step);

    report.phase(label);
    auto input = specferry::testing::read_bytes(fixture, "input." + std::to_string(step) + ".bin",
                                                hidden.spec.bytes());
    upload_tensor(g.graph, hidden.id, input);
    upload_tensor(g.graph, controls.id, integers({std::int32_t(step), std::int32_t(step + 1)}));
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(g.graph.get()); }),
          "run integrated decoder layer");
    report.compare(label + ".input", read_tensor(g.graph, hidden.id), input, f16);
    for (const auto &[name, tensor] : outputs) {
      report.compare(label + "." + name, read_tensor(g.graph, tensor.id),
                     specferry::testing::read_bytes(
                         fixture, "zero.layer.0." + name + "." + std::to_string(step) + ".bin",
                         tensor.spec.bytes()),
                     f16, 0.08f, 0.02f);
    }
    for (unsigned kind = 0; kind < 3; ++kind) {
      std::vector<std::uint8_t> data;
      for (unsigned head = 0; head < config.heads; ++head) {
        const auto tensor = kind == 2 ? probabilities[head] : parents[head * 2 + kind];
        auto part = read_tensor(kind == 2 ? g.graph : storage, tensor.id);
        data.insert(data.end(), part.begin(), part.end());
      }
      const std::string name = kind == 0 ? "keys" : kind == 1 ? "values" : "probabilities";
      report.compare(
          label + "." + name, data,
          specferry::testing::read_bytes(
              fixture, "zero.layer.0." + name + "." + std::to_string(step) + ".bin", data.size()),
          f16, 0.08f, 0.02f);
    }
    if (report.failures) {
      break;
    }
  }
  bank.verify();
  report.phase("release");
  g.graph.close();
  storage.close();
  bank.close();
}

void prefix_check(Context &context, const WeightStore &weights, const fs::path &fixture,
                  Report &report, unsigned layers) {
  auto config = opt::read_model_config(fixture);
  if (!layers || layers > config.decoder.layers) {
    throw std::invalid_argument("diagnostic prefix exceeds decoder layer count");
  }
  // This is a diagnostic truncation, not a supported text-generation model.
  // KV in a prefix is independent of later layers and can use the full CPU oracle.
  config.decoder.layers = layers;
  std::ifstream source(fixture / "tokens.txt");
  std::vector<std::int32_t> prompt(2);
  if (!(source >> prompt[0] >> prompt[1])) {
    throw std::invalid_argument("prefix fixture needs two teacher tokens");
  }
  report.phase("setup");
  opt::GraphModel model(context, weights, config, 1);
  report.phase("execute");
  const auto generated = model.generate(prompt, 1);
  std::cout << "Diagnostic prefix prediction (not a model acceptance): " << generated.tokens.at(0)
            << std::endl;
  for (unsigned layer = 0; layer < layers; ++layer) {
    for (bool values : {false, true}) {
      const auto name = "layer." + std::to_string(layer) + (values ? ".values" : ".keys");
      const auto actual = model.read_cache(layer, values);
      report.compare(
          name, actual,
          specferry::testing::read_bytes(fixture, "teacher.1." + name + ".bin", actual.size()), f16,
          0.08f, 0.02f);
    }
  }
  report.phase("release");
  model.close();
}
} // namespace

int main(int argc, char **argv) {
  if ((argc != 5 && argc != 6) ||
      (std::string(argv[4]) != "lookup" && std::string(argv[4]) != "layer" &&
       std::string(argv[4]) != "prefix")) {
    std::cerr << "usage: np101_graph_pipeline_check DEPLOYMENT FIXTURE OUTPUT lookup|layer|prefix "
                 "[LAYERS]\n";
    return 2;
  }
  try {
    fs::create_directories(argv[3]);
    Report report(argv[3]);
    SdkTimings timings(argv[3]);
    WeightStore weights(argv[1]);
    weights.verify();
    Context context;
    if (std::string(argv[4]) == "lookup") {
      lookup_check(context, weights, argv[2], report);
    } else if (std::string(argv[4]) == "layer") {
      layer_check(context, weights, argv[2], report);
    } else {
      const auto layers = parse_shape(argc == 6 ? argv[5] : "1");
      if (layers.size() != 1) {
        throw std::invalid_argument("expected one layer count");
      }
      prefix_check(context, weights, argv[2], report, layers[0]);
    }
    context.close();
    report.phase(report.failures ? "numerical_failure" : "complete", true);
    std::cout << report.checks << " checks; " << report.failures << " failures" << std::endl;
    return report.failures ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
