#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/memory_budget.hpp"
#include "devices/np101/api/runtime_profile.hpp"
#include "devices/np101/api/weights.hpp"
#include "inference/generation_io.hpp"
#include "models/qwen3_5/model.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

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
    const auto prompt = inference::read_prompt(input, policy.vocabulary, policy.capacity);
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
    if (head_backend_name != "matmul" && head_backend_name != "shared-fcl" &&
        head_backend_name != "shared-fcl-split") {
      throw std::invalid_argument("unsupported Qwen head projection");
    }
    const auto head_backend = head_backend_name != "matmul"
                                  ? np101::ops::LinearBackend::FullyConnected
                                  : np101::ops::LinearBackend::MatMul;
    const auto state_commit = read_option(request, "state-commit.txt", "per-layer");
    if (state_commit != "per-layer" && state_commit != "combined" && state_commit != "ffn") {
      throw std::invalid_argument("unsupported Qwen state commit policy");
    }
    if (fs::exists(request / "compiled-head-package.txt")) {
      throw std::invalid_argument(
          "compiled vocabulary-prefix requests are retired; use shared-fcl");
    }
    const auto batch_name = read_option(request, "ffn-batch-tokens.txt", "1");
    if (batch_name != "1" && batch_name != "8" && batch_name != "16") {
      throw std::invalid_argument("unsupported Qwen FFN batch width");
    }
    const unsigned ffn_batch_tokens = std::stoul(batch_name);
    if (state_commit == "ffn" && ffn_batch_tokens == 1) {
      throw std::invalid_argument("FFN state commit requires batched FFN");
    }
    const auto layout_name = read_option(request, "decoder-layout.txt", "separate");
    using DecoderLayout = models::qwen3_5::DecoderLayout;
    DecoderLayout layout;
    if (layout_name == "separate") {
      layout = DecoderLayout::Separate;
    } else if (layout_name == "fused-attention-norm") {
      layout = DecoderLayout::FusedAttentionNorm;
    } else {
      throw std::invalid_argument("unsupported Qwen decoder layout");
    }
    if (layout != DecoderLayout::Separate && ffn_batch_tokens == 1) {
      throw std::invalid_argument("optimized Qwen decoder layout requires batched FFN");
    }
    int matmul_layer = ffn_batch_tokens == 1 ? 9 : -1;
    if (fs::exists(request / "matmul-feed-forward-layer.txt")) {
      std::ifstream fallback(request / "matmul-feed-forward-layer.txt");
      if (!(fallback >> matmul_layer) || fallback >> extra) {
        throw std::invalid_argument("invalid Qwen MatMul feed-forward layer");
      }
    }
    if ((ffn_batch_tokens == 1 &&
         (matmul_layer < 0 || unsigned(matmul_layer) >= config.layer_types.size())) ||
        (ffn_batch_tokens > 1 && matmul_layer != -1)) {
      throw std::invalid_argument("Qwen FFN batch/fallback layer contract differs");
    }
    const auto projections = read_option(request, "projections.txt", "matmul");
    const bool shared_fcl = projections == "shared-fcl";
    if (projections != "matmul" && !shared_fcl) {
      throw std::invalid_argument("unsupported Qwen projection backend");
    }
    if (shared_fcl &&
        (std::string_view(platform::runtime_profile().target) != "soc" ||
         (policy.capacity != 64 && policy.capacity != 128 && policy.capacity != 512) ||
         matmul_layer != (ffn_batch_tokens == 1 ? 9 : -1) ||
         head_backend_name != "shared-fcl-split" || state_commit == "per-layer" ||
         lookup_name != "gather-nd" || token_readback_name != "mapped")) {
      throw std::invalid_argument("Qwen shared FCL requires the validated SoC M1 composition");
    }
    const auto kv_write_name = read_option(request, "kv-write-mode.txt", "copy-graph");
    if (kv_write_name != "copy-graph" && kv_write_name != "mapped") {
      throw std::invalid_argument("unsupported Qwen KV write mode");
    }
    if (kv_write_name == "mapped" &&
        (std::string_view(platform::runtime_profile().target) != "soc" || !shared_fcl)) {
      throw std::invalid_argument("mapped Qwen KV requires the shared FCL SoC composition");
    }
    const auto kv_write_mode =
        kv_write_name == "mapped" ? np101::KvWriteMode::MappedHost : np101::KvWriteMode::CopyGraph;
    if (ffn_batch_tokens > 1 &&
        (!shared_fcl || (policy.capacity != 128 && policy.capacity != 512) ||
         kv_write_name != "mapped")) {
      throw std::invalid_argument("batched Qwen FFN requires a C128/C512 mapped shared FCL chain");
    }
    np101::WeightStore weights(argv[1]);
    if (platform::verify_model_payloads()) {
      weights.verify();
    }
    const auto artifacts = models::qwen3_5::load_compiled_mixers(argv[2], weights, config);
    auto feed_forward = models::qwen3_5::load_compiled_feed_forward(argv[2], weights, config);
    for (const auto &entry : feed_forward) {
      if (models::qwen3_5::compiled_feed_forward_batch_tokens(entry.second) != ffn_batch_tokens) {
        throw std::invalid_argument("Qwen FFN package does not match selected batch width");
      }
    }
    if (ffn_batch_tokens > 1 && feed_forward.size() != config.layer_types.size()) {
      throw std::invalid_argument("batched Qwen FFN requires all decoder artifacts");
    }
    // The default M1 chain retains its validated shared FCL replacement at layer 9.
    if (matmul_layer >= 0) {
      feed_forward.erase(unsigned(matmul_layer));
    }
    fs::create_directories(output);
    platform::write_runtime_profile(output);
    np101::SdkTimings timings(output);
    phase(output, "initialize");
    const auto start = std::chrono::steady_clock::now();
    np101::Context context;
    models::qwen3_5::ModelOptions model_options;
    model_options.lookup = lookup;
    model_options.head = head_backend;
    model_options.head_blocks_per_graph = head_backend_name == "shared-fcl-split" ? 8 : 0;
    model_options.projections = shared_fcl ? np101::ops::LinearBackend::SharedFullyConnected
                                           : np101::ops::LinearBackend::MatMul;
    model_options.kv_write = kv_write_mode;
    model_options.layout = layout;
    using StateCommit = models::qwen3_5::StateCommit;
    model_options.state_commit = state_commit == "ffn"        ? StateCommit::FeedForward
                                 : state_commit == "combined" ? StateCommit::Combined
                                                              : StateCommit::PerLayer;
    models::qwen3_5::Model model(
        context, weights, config, policy, artifacts, feed_forward, model_options,
        [&](const std::string &name) {
          phase(output, name.compare(0, 6, "token.") == 0 ? name : "setup." + name);
        });
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
                  << ",\"capacity\":" << policy.capacity
                  << ",\"const_peak_bytes\":" << memory.peak(true)
                  << ",\"nonconst_peak_bytes\":" << memory.peak(false)
                  << ",\"compiled_mixers\":" << artifacts.size() << ",\"embedding_lookup\":\""
                  << lookup_name << '"' << ",\"matmul_feed_forward_layer\":" << matmul_layer
                  << ",\"token_readback\":\"" << token_readback_name << '"'
                  << ",\"head_backend\":\"" << head_backend_name << '"' << ",\"state_commit\":\""
                  << state_commit << '"' << ",\"projections\":\"" << projections << '"'
                  << ",\"kv_write_mode\":\"" << kv_write_name << '"'
                  << ",\"compiled_input_norm_layers\":" << compiled_input_norm_layers
                  << ",\"ffn_batch_tokens\":" << ffn_batch_tokens << ",\"decoder_layout\":\""
                  << layout_name << '"' << ",\"compiled_feed_forward\":" << feed_forward.size()
                  << "}\n";
      if (!initialized) {
        throw std::runtime_error("cannot save Qwen initialization evidence");
      }
    }
    platform::notify_model_ready();
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
    const auto kv_revalidations = model.kv_revalidations();
    model.close();
    const auto &memory = context.memory();
    if (memory.live(true) || memory.live(false)) {
      throw std::runtime_error("Qwen retains tensor payload after release");
    }
    context.close();
    // Publish completion only after model storage and its Context are released.
    // Both platforms use the same record; the SoC supervisor also requires run.json.
    for (const auto *name : {"execution.json", "run.json"}) {
      std::ofstream file(output / name);
      file << std::setprecision(17)
           << "{\"status\":\"executed\",\"phase\":\"complete\",\"released\":true,\"initialize_"
              "seconds\":"
           << initialize << ",\"kv_revalidations\":" << kv_revalidations
           << ",\"const_live_bytes\":0,\"nonconst_live_bytes\":0,\"const_peak_bytes\":"
           << memory.peak(true) << ",\"nonconst_peak_bytes\":" << memory.peak(false) << "}\n";
      if (!file) {
        throw std::runtime_error("cannot save Qwen release evidence");
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
