#include "case_file.hpp"
#include "np101/weights.hpp"

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

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
  const std::string record = "model.embed_tokens.weight F16 2,2 0 8 "
                             "66840dda154e8a113c31dd0ad32f7f3a366a80e8136979d8f5a101d3d29d6f72\n";
  write(directory.path / "weights.bin", payload);
  write(directory.path / "weights.index", "specferry-np101-weights 1\n" + record);
  WeightStore store(directory.path);
  store.verify();
  const auto &embedding = store.find("model.embed_tokens.weight");
  require(&embedding == &store.find("lm_head.weight"), "tied head must resolve to the same record");
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
}
} // namespace

int main() {
  try {
    tensor_boundaries();
    weight_integrity();
    case_validation();
    std::cout << "Tensor boundaries, weight integrity, and fixture validation passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
