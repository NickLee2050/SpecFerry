#include "inference/generation_io.hpp"
#include "models/qwen3_5/model.hpp"
#include "np101/diagnostics.hpp"
#include "np101/memory_budget.hpp"
#include "np101/weights.hpp"
#include "platform/runtime_profile.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
namespace fs = std::filesystem;
using namespace specferry;

void phase(const fs::path &output, const std::string &name) {
  const auto now =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  std::ofstream file(output / "phase.json");
  file << std::setprecision(17) << "{\"phase\":\"" << name << "\",\"unix_seconds\":" << now
       << "}\n";
  if (!file) {
    throw std::runtime_error("cannot save Qwen progress");
  }
}

void save(const fs::path &path, const std::vector<std::uint8_t> &bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  if (!file) {
    throw std::runtime_error("cannot save Qwen coarse diagnostic");
  }
}

std::string read_option(const fs::path &request, const char *name, const char *fallback) {
  const auto path = request / name;
  if (!fs::exists(path)) {
    return fallback;
  }
  std::ifstream input(path);
  std::string value, extra;
  if (!(input >> value) || input >> extra) {
    throw std::invalid_argument("invalid Qwen option: " + path.string());
  }
  return value;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: specferry_qwen_generate DEPLOYMENT MIXERS REQUEST OUTPUT\n";
    return 2;
  }
  try {
    const fs::path request(argv[3]), output(argv[4]);
    const auto config = models::qwen3_5::read_config(request / "components.txt");
    std::ifstream policy_file(request / "policy.txt");
    inference::GenerationPolicy policy{};
    std::string extra;
    unsigned maximum = 0, repeats = 0;
    policy_file >> policy.vocabulary >> policy.capacity >> policy.bos >> policy.eos >> maximum >>
        repeats;
    if (!policy_file || policy_file >> extra || maximum == 0 || maximum > 32 || repeats == 0 ||
        repeats > 3) {
      throw std::invalid_argument("invalid Qwen generation policy");
    }
    std::ifstream input(request / "tokens.txt");
    std::vector<std::int32_t> prompt;
    std::int32_t token;
    while (input >> token) {
      if (token < 0 || unsigned(token) >= policy.vocabulary) {
        throw std::invalid_argument("Qwen prompt token exceeds vocabulary");
      }
      prompt.push_back(token);
    }
    if (!input.eof() || prompt.empty() || prompt.size() > policy.capacity) {
      throw std::invalid_argument("invalid Qwen rendered prompt");
    }
    const auto lookup_name = read_option(request, "embedding-lookup.txt", "gather-nd");
    if (lookup_name != "gather" && lookup_name != "gather-nd") {
      throw std::invalid_argument("unsupported Qwen embedding lookup operator");
    }
    const auto lookup = lookup_name == "gather-nd" ? np101::ops::VocabularyLookup::GatherNd
                                                   : np101::ops::VocabularyLookup::Gather;
    const auto token_readback_name = read_option(request, "token-readback.txt", "mapped");
    if (token_readback_name != "sdk" && token_readback_name != "mapped") {
      throw std::invalid_argument("unsupported Qwen token readback transport");
    }
    const auto token_readback = token_readback_name == "mapped" ? np101::TensorReadback::MappedHost
                                                                : np101::TensorReadback::SdkCopy;
    const auto head_backend_name = read_option(request, "head-backend.txt", "matmul");
    if (head_backend_name != "matmul" && head_backend_name != "shared-fcl") {
      throw std::invalid_argument("unsupported Qwen head projection");
    }
    const auto head_backend = head_backend_name == "shared-fcl"
                                  ? np101::ops::LinearBackend::FullyConnected
                                  : np101::ops::LinearBackend::MatMul;
    const auto state_commit = read_option(request, "state-commit.txt", "per-layer");
    if (state_commit != "per-layer" && state_commit != "combined") {
      throw std::invalid_argument("unsupported Qwen state commit policy");
    }
    if (fs::exists(request / "compiled-head-package.txt")) {
      throw std::invalid_argument(
          "compiled vocabulary-prefix requests are retired; use shared-fcl");
    }
    unsigned matmul_layer = 9;
    if (fs::exists(request / "matmul-feed-forward-layer.txt")) {
      std::ifstream fallback(request / "matmul-feed-forward-layer.txt");
      unsigned layer = 0;
      fallback >> layer;
      if (!fallback || fallback >> extra) {
        throw std::invalid_argument("invalid Qwen MatMul feed-forward layer");
      }
      matmul_layer = layer;
    }
    if (matmul_layer >= config.layer_types.size()) {
      throw std::invalid_argument("Qwen MatMul feed-forward layer exceeds model depth");
    }
    np101::WeightStore weights(argv[1]);
    weights.verify();
    const auto artifacts = models::qwen3_5::load_compiled_mixers(argv[2], weights, config);
    auto feed_forward = models::qwen3_5::load_compiled_feed_forward(argv[2], weights, config);
    // Validate the complete package before selecting the existing device MatMul FFN.
    feed_forward.erase(matmul_layer);
    fs::create_directories(output);
    platform::write_runtime_profile(output);
    np101::SdkTimings timings(output);
    phase(output, "initialize");
    const auto start = std::chrono::steady_clock::now();
    np101::Context context;
    models::qwen3_5::Model model(
        context, weights, config, policy, artifacts, feed_forward,
        [&](const std::string &name) {
          phase(output, name.compare(0, 6, "token.") == 0 ? name : "setup." + name);
        },
        lookup, head_backend, state_commit == "combined");
    const auto initialize =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    {
      // Keep completed preparation evidence even if a later request fails.
      const auto &memory = context.memory();
      unsigned compiled_input_norm_layers = 0;
      for (const auto &entry : artifacts) {
        compiled_input_norm_layers += entry.second.includes_input_norm;
      }
      std::ofstream initialized(output / "initialization.json");
      initialized << std::setprecision(17) << "{\"initialize_seconds\":" << initialize
                  << ",\"const_peak_bytes\":" << memory.peak(true)
                  << ",\"nonconst_peak_bytes\":" << memory.peak(false)
                  << ",\"compiled_mixers\":" << artifacts.size() << ",\"embedding_lookup\":\""
                  << lookup_name << '"' << ",\"matmul_feed_forward_layer\":" << matmul_layer
                  << ",\"token_readback\":\"" << token_readback_name << '"'
                  << ",\"head_backend\":\"" << head_backend_name << '"' << ",\"state_commit\":\""
                  << state_commit << '"'
                  << ",\"compiled_input_norm_layers\":" << compiled_input_norm_layers
                  << ",\"compiled_feed_forward\":" << feed_forward.size() << "}\n";
      if (!initialized) {
        throw std::runtime_error("cannot save Qwen initialization evidence");
      }
    }
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
      phase(output, "request." + std::to_string(repeat));
      const auto result = model.generate(prompt, maximum, token_readback);
      std::ofstream file(output / ("measured." + std::to_string(repeat) + ".json"));
      file << '{';
      inference::write_generation(file, result);
      file << "}\n";
      if (!file) {
        throw std::runtime_error("cannot save Qwen generation");
      }
      // Coarse snapshots are outside generation wall time and reused numerical oracles.
      save(output / ("measured." + std::to_string(repeat) + ".hidden.bin"),
           model.read(23, "output"));
      save(output / ("measured." + std::to_string(repeat) + ".recurrent.bin"),
           model.read(22, "recurrent"));
      save(output / ("measured." + std::to_string(repeat) + ".keys.bin"), model.read(23, "keys"));
      save(output / ("measured." + std::to_string(repeat) + ".values.bin"),
           model.read(23, "values"));
    }
    phase(output, "release");
    model.close();
    const auto &memory = context.memory();
    if (memory.live(true) || memory.live(false)) {
      throw std::runtime_error("Qwen retains tensor payload after release");
    }
    std::ofstream file(output / "execution.json");
    file << "{\"status\":\"executed\",\"phase\":\"complete\",\"released\":true,\"initialize_"
            "seconds\":"
         << initialize << ",\"const_live_bytes\":0,\"nonconst_live_bytes\":0,\"const_peak_bytes\":"
         << memory.peak(true) << ",\"nonconst_peak_bytes\":" << memory.peak(false) << "}\n";
    context.close();
    return file ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
