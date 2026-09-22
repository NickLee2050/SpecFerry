#include "case_file.hpp"
#include "models/opt/config.hpp"
#include "models/qwen3_5/config.hpp"
#include "np101/component_spec.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
using namespace specferry::np101;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void rejects(const std::function<void()> &operation, const std::string &message) {
  try {
    operation();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected rejection: " + message);
}

struct TemporaryDirectory {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("specferry-data-test-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

  TemporaryDirectory() { std::filesystem::create_directory(path); }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

void write(const std::filesystem::path &path, const std::string &data) {
  std::ofstream output(path, std::ios::binary);
  output.write(data.data(), data.size());
  require(bool(output), "test fixture write failed");
}

void tensor_boundaries() {
  require(TensorSpec{DataType::Float32, {128, 128, 16}}.bytes() == 1024 * 1024,
          "FP32 recurrent state byte count");
  require(parse_shape("256,2,512") == std::vector<std::uint32_t>({256, 2, 512}), "SDK shape order");
  for (const auto &shape : {"", "0,4", "-1,2", "3,", "2,,3", "4294967296", " 2"}) {
    rejects([&] { parse_shape(shape); }, shape);
  }
  rejects([] { TensorSpec{DataType::Float32, {UINT32_MAX, UINT32_MAX, 8}}.bytes(); }, "overflow");
  rejects([] { parse_dtype("BF16"); }, "unconverted weight dtype");
}

void weight_integrity() {
  TemporaryDirectory directory;
  const std::string payload("\1\2\3\4\5\6\7\10", 8);
  const std::string record = "block.projection.weight F16 2,2 0 8 "
                             "66840dda154e8a113c31dd0ad32f7f3a366a80e8136979d8f5a101d3d29d6f72\n";
  write(directory.path / "weights.bin", payload);
  write(directory.path / "weights.index", "specferry-np101-weights 1\n" + record);
  WeightStore store(directory.path);
  store.verify();
  const auto &embedding = store.find("block.projection.weight");
  rejects([&] { store.find("lm_head.weight"); }, "reader must not infer model aliases");
  require(store.read(embedding, 4, 4) == std::vector<std::uint8_t>({5, 6, 7, 8}),
          "bounded row read");
  rejects([&] { store.read(embedding, 7, 2); }, "cross-record read");
  rejects([&] { store.read(embedding, UINT64_MAX, 1); }, "overflowing read offset");
  auto forged = embedding;
  rejects([&] { store.read(forged, 0, 1); }, "foreign record");
  write(directory.path / "weights.bin", std::string(8, '\0'));
  rejects([&] { store.verify(); }, "same-size corruption");
  write(directory.path / "weights.bin", payload.substr(0, 7));
  rejects([&] { WeightStore truncated(directory.path); }, "truncated payload");
  write(directory.path / "weights.bin", payload);
  write(directory.path / "weights.index", "specferry-np101-weights 1\n" + record + record);
  rejects([&] { WeightStore duplicate(directory.path); }, "duplicate tensor and overlap");
}

void component_contracts() {
  KvSpec cache{3, 16, 8};
  cache.validate();
  require(cache.tensor().bytes() == 768 && cache.token_bytes() == 192,
          "alternate KV dimensions and slot accounting");
  rejects([] { KvSpec{2, 16, 513}.validate(); }, "unsupported KV capacity");
  rejects([] { DeltaSpec{64, UINT32_MAX, 8, 8, 3, 1e-6f}.validate(); },
          "overflowing DeltaNet dimensions");

  TemporaryDirectory directory;
  const auto path = directory.path / "components.txt";
  const std::string parameters = "64 96 4 2 16 8 8 10000 1e-6 2 8 8 3";
  write(path, "specferry-qwen-components 1\n" + parameters + "\ndelta attention\n");
  auto config = specferry::models::qwen3_5::read_config(path);
  require(config.hidden_spec().bytes() == 128 && config.delta.channels() == 48,
          "component parser must preserve dimensions");
  require(config.mixer(1) == specferry::models::qwen3_5::MixerKind::Attention,
          "explicit layer order");
  rejects([&] { config.mixer(2); }, "unconfigured layer");
  for (const auto &record : {"-1" + parameters.substr(2), parameters + " extra",
                             std::string("64 96 3 2 16 8 8 10000 1e-6 2 8 8 3")}) {
    write(path, "specferry-qwen-components 1\n" + record + "\ndelta attention\n");
    rejects([&] { specferry::models::qwen3_5::read_config(path); }, "invalid component record");
  }
  write(path, "specferry-qwen-components 1\n" + parameters + "\nunknown\n");
  rejects([&] { specferry::models::qwen3_5::read_config(path); }, "unknown mixer");
}

void opt_contracts() {
  TemporaryDirectory directory;
  const auto path = directory.path / "components.txt";
  const std::string header = "specferry-opt-components 1\n";
  write(path, header + "1024 4096 16 24 512 1e-5\n");
  const auto config = specferry::models::opt::read_config(path);
  require(config.kv_spec().head_dim == 64, "OPT heads must retain their configured width");
  rejects([&] { config.prefix(24); }, "OPT layer outside configured range");
  write(directory.path / "model.txt", "specferry-opt-model 1\n512 50272 2048 2 4096 2 2 1\n");
  auto model = specferry::models::opt::read_model_config(directory.path);
  model.validate_token(0);
  model.validate_token(50271);
  rejects([&] { model.validate_token(-1); }, "negative model token");
  rejects([&] { model.validate_token(50272); }, "out-of-vocabulary model token");
  for (const auto *record : {"512 50272 2048 0 4096 2 2 1", "512 50272 511 2 4096 2 2 1",
                             "512 50272 2048 2 0 2 2 1", "512 50272 2048 2 4096 50272 2 1",
                             "512 50272 2048 2 4096 2 2 2", "512 50272 2048 2 4096 2 2 1 extra"}) {
    write(directory.path / "model.txt", std::string("specferry-opt-model 1\n") + record + "\n");
    rejects([&] { specferry::models::opt::read_model_config(directory.path); },
            "invalid model IO configuration rejected before device initialization");
  }
  for (const auto *record :
       {"1024 4096 0 24 512 1e-5", "1024 4096 16 24 513 1e-5", "1024 4096 16 24 512 -1",
        "1024 4096 16 24 512 1e-5 extra", "-1 4096 16 24 512 1e-5", "1024 4097 16 24 512 1e-5"}) {
    write(path, header + record + "\n");
    rejects([&] { specferry::models::opt::read_config(path); }, "invalid OPT contract");
  }
  // An otherwise well-formed package lacking projection biases cannot build a layer.
  write(directory.path / "weights.bin", std::string("\1\2\3\4\5\6\7\10", 8));
  write(directory.path / "weights.index",
        "specferry-np101-weights 1\ndecoder.layers.0.self_attn.q_proj.weight F16 2,2 0 8 "
        "66840dda154e8a113c31dd0ad32f7f3a366a80e8136979d8f5a101d3d29d6f72\n");
  WeightStore weights(directory.path);
  const specferry::models::opt::Config small{2, 4, 1, 1, 2, 1e-5f};
  rejects([&] { specferry::models::opt::validate_weights(weights, small, {0}); },
          "incomplete OPT weights rejected before SDK initialization");
}

void case_validation() {
  TemporaryDirectory directory;
  const auto path = directory.path / "graph.txt";
  const std::string valid = "specferry-np101-case 1\nsteps 2\n"
                            "tensor x F32 mutable 4,1 -\ntensor y F32 mutable 4,1 -\n"
                            "node ADD x,x y -\ninput x first.bin,second.bin\noutput y\n";
  write(path, valid);
  auto test = specferry::testing::load_case(path);
  require(test.steps == 2 && test.nodes.size() == 1, "fixture must retain graph and steps");
  write(path, valid + "tensor x F32 mutable 4,1 -\n");
  rejects([&] { specferry::testing::load_case(path); }, "duplicate tensor");
  write(path, valid + "output unknown\n");
  rejects([&] { specferry::testing::load_case(path); }, "unknown output");
  write(path, valid + "ignored_directive\n");
  rejects([&] { specferry::testing::load_case(path); }, "unknown directive");
  write(directory.path / "data.bin", std::string(16, '\0'));
  rejects([&] { specferry::testing::read_bytes(directory.path, "data.bin", 15); },
          "wrong file size");
  rejects([&] { specferry::testing::read_bytes(directory.path, "../data.bin", 16); },
          "path escape");

  // Archived feedback fixtures must fail before device initialization, rather
  // than silently running independent steps after their feedback path is removed.
  const std::string legacy = "specferry-np101-case 2\nsteps 2\n"
                             "tensor x F32 mutable 4,1 data.bin\n"
                             "tensor y F32 mutable 4,1 -\nnode ADD x,x y -\noutput y\n";
  for (const auto &directive : {"feedback y x\n", "feedback y x\nreset_after 1\n",
                                "tensor old F32 handle 4,1 data.bin\n"}) {
    write(path, legacy + directive);
    rejects([&] { specferry::testing::load_case(path); }, "retired feedback fixture");
  }
  write(path, valid + "bounds x 0 511\n");
  rejects([&] { specferry::testing::load_case(path); }, "bounds on non-integer data");

  const std::string bounded =
      "specferry-np101-case 2\nsteps 1\n"
      "tensor index I32 mutable 1 -\ntensor result I32 mutable 1 -\n"
      "node ADD index,index result -\ninput index index.bin\noutput result\n"
      "bounds index 0 511\n";
  write(path, bounded);
  auto indexed = specferry::testing::load_case(path);
  for (std::int32_t index : {-1, 512, 513}) {
    write(directory.path / "index.bin",
          std::string(reinterpret_cast<const char *>(&index), sizeof(index)));
    rejects([&] { specferry::testing::validate_case_data(indexed); },
            "out-of-capacity index rejected before SDK initialization");
  }
  std::int32_t last = 511;
  write(directory.path / "index.bin",
        std::string(reinterpret_cast<const char *>(&last), sizeof(last)));
  specferry::testing::validate_case_data(indexed);
}
} // namespace

int main() {
  try {
    tensor_boundaries();
    weight_integrity();
    component_contracts();
    opt_contracts();
    case_validation();
    std::cout << "Tensor boundaries, weight integrity, and fixture validation passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
