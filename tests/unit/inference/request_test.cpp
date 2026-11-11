#include "io/files.hpp"
#include "models/opt/request.hpp"
#include "models/qwen3_5/request.hpp"

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <system_error>

namespace {
namespace fs = std::filesystem;

void require(bool value) {
  if (!value) {
    throw std::runtime_error("native request contract failed");
  }
}

template <class Action> void rejects(Action action) {
  try {
    action();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid request was accepted");
}

struct Directory {
  fs::path path = fs::temp_directory_path() /
                  ("specferry-request-" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

  Directory() { fs::create_directory(path); }

  ~Directory() {
    std::error_code ignored;
    fs::remove_all(path, ignored);
  }

  void write(const char *name, const std::string &text) const {
    std::ofstream output(path / name);
    output << text;
    require(bool(output));
  }
};

void opt_request() {
  using namespace specferry::models::opt;
  Directory directory;
  directory.write("components.txt", "specferry-opt-components 1\n1024 4096 16 24 256 1e-5\n");
  directory.write("model.txt", "specferry-opt-model 1\n512 50272 2048 2 4096 2 2 1\n");
  directory.write("requests.txt", "2 7 8\n2 9\n");
  directory.write("expected.txt", "11 12\n13\n");
  auto request = read_graph_request(directory.path);
  require(request.requests.size() == 2 && request.expected[0].size() == 2);
  require(request.model_options.variant == GraphVariant::MergedKv);
  directory.write("run-options.txt", "variant compiled-fullgraph\nkv_transfer mapped-slot\n"
                                     "block 16\nprefill_head last-token\nmax_new_tokens 16\n");
  request = read_graph_request(directory.path);
  require(request.model_options.kv_transfer == KvTransfer::MappedSlot);
  require(request.model_options.prefill_block == 16);
  directory.write("expected.txt", "11\n");
  rejects([&] { read_graph_request(directory.path); });
  directory.write("expected.txt", "11\n50272\n");
  rejects([&] { read_graph_request(directory.path); });
}

void qwen_request() {
  using namespace specferry::models::qwen3_5;
  Directory directory;
  std::string config = "specferry-qwen-components 1\n"
                       "1024 3584 8 2 256 128 64 10000000 1e-6 16 128 128 4\n";
  for (unsigned layer = 0; layer < 24; ++layer) {
    config += layer % 4 == 3 ? "attention\n" : "delta\n";
  }
  directory.write("components.txt", config);
  directory.write("policy.txt", "248320 128 0 248044 16 1\n");
  directory.write("tokens.txt", "1 7 9\n");
  auto request = read_execution_request(directory.path, "ep");
  require(request.maximum == 16 && request.matmul_layer == 9);
  require(request.ffn_batch_tokens == 1);
  directory.write("projections.txt", "shared-fcl\n");
  directory.write("head-backend.txt", "shared-fcl-split\n");
  directory.write("state-commit.txt", "ffn\n");
  directory.write("ffn-batch-tokens.txt", "8\n");
  directory.write("kv-write-mode.txt", "mapped\n");
  directory.write("decoder-layout.txt", "fused-attention-norm\n");
  request = read_execution_request(directory.path, "soc");
  require(request.ffn_batch_tokens == 8 && request.matmul_layer == -1);
  require(request.model_options.head_blocks_per_graph == 8);
  require(request.model_options.state_commit == StateCommit::FeedForward);
  rejects([&] { read_execution_request(directory.path, "ep"); });
  directory.write("ffn-batch-tokens.txt", "4\n");
  rejects([&] { read_execution_request(directory.path, "soc"); });
  directory.write("ffn-batch-tokens.txt", "16\n");
  require(read_execution_request(directory.path, "soc").ffn_batch_tokens == 16);
  directory.write("tokens.txt", "1 248320\n");
  rejects([&] { read_execution_request(directory.path, "soc"); });
}

void binary_output() {
  Directory directory;
  const std::vector<std::uint8_t> bytes{0, 1, 127, 255};
  specferry::io::write_bytes(directory.path / "state.bin", bytes, "cannot save state");
  std::ifstream input(directory.path / "state.bin", std::ios::binary);
  const std::vector<std::uint8_t> read((std::istreambuf_iterator<char>(input)), {});
  require(read == bytes);
}
} // namespace

int main() {
  try {
    opt_request();
    qwen_request();
    binary_output();
    std::cout << "Native request and output contracts passed without opening a Context.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
