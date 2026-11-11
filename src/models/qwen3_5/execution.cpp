#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/memory_budget.hpp"
#include "devices/np101/api/runtime_profile.hpp"
#include "devices/np101/api/weights.hpp"
#include "inference/report.hpp"
#include "io/files.hpp"
#include "models/qwen3_5/execution.hpp"
#include "models/qwen3_5/request.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
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

} // namespace

namespace specferry::models::qwen3_5 {
int execute(const inference::ExecutionPaths &paths, const std::filesystem::path &compiled) {
  const fs::path request(paths.request), output(paths.output);
  const auto settings =
      models::qwen3_5::read_execution_request(request, platform::runtime_profile().target);
  const auto &config = settings.config;
  const auto &policy = settings.policy;
  const auto &prompt = settings.prompt;
  const auto &maximum = settings.maximum;
  const auto &repeats = settings.repeats;
  const auto &lookup_name = settings.lookup_name;
  const auto &token_readback_name = settings.token_readback_name;
  const auto &head_backend_name = settings.head_backend_name;
  const auto &state_commit = settings.state_commit;
  const auto &ffn_batch_tokens = settings.ffn_batch_tokens;
  const auto &layout_name = settings.layout_name;
  const auto &matmul_layer = settings.matmul_layer;
  const auto &projections = settings.projections;
  const auto &kv_write_name = settings.kv_write_name;
  const auto &token_readback = settings.token_readback;
  const auto &model_options = settings.model_options;
  np101::WeightStore weights(paths.deployment);
  if (platform::verify_model_payloads()) {
    weights.verify();
  }
  const auto artifacts = models::qwen3_5::load_compiled_mixers(compiled, weights, config);
  auto feed_forward = models::qwen3_5::load_compiled_feed_forward(compiled, weights, config);
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
  models::qwen3_5::Model model(context, weights, config, policy, artifacts, feed_forward,
                               model_options, [&](const std::string &name) {
                                 phase(output,
                                       name.compare(0, 6, "token.") == 0 ? name : "setup." + name);
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
                << ",\"token_readback\":\"" << token_readback_name << '"' << ",\"head_backend\":\""
                << head_backend_name << '"' << ",\"state_commit\":\"" << state_commit << '"'
                << ",\"projections\":\"" << projections << '"' << ",\"kv_write_mode\":\""
                << kv_write_name << '"'
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
    io::write_bytes(output / ("measured." + std::to_string(repeat) + ".hidden.bin"),
                    model.read(23, "output"), "cannot save Qwen coarse diagnostic");
    io::write_bytes(output / ("measured." + std::to_string(repeat) + ".recurrent.bin"),
                    model.read(22, "recurrent"), "cannot save Qwen coarse diagnostic");
    io::write_bytes(output / ("measured." + std::to_string(repeat) + ".keys.bin"),
                    model.read(23, "keys"), "cannot save Qwen coarse diagnostic");
    io::write_bytes(output / ("measured." + std::to_string(repeat) + ".values.bin"),
                    model.read(23, "values"), "cannot save Qwen coarse diagnostic");
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
}
} // namespace specferry::models::qwen3_5
