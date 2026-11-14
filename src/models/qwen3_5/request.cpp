#include "inference/request.hpp"
#include "models/qwen3_5/request.hpp"

#include <fstream>
#include <stdexcept>

namespace specferry::models::qwen3_5 {
namespace fs = std::filesystem;

namespace {
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

ExecutionRequest read_execution_request(const fs::path &request, std::string_view target) {
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
    throw std::invalid_argument("compiled vocabulary-prefix requests are retired; use shared-fcl");
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
  if (shared_fcl && (target != "soc" ||
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
  if (kv_write_name == "mapped" && (target != "soc" || !shared_fcl)) {
    throw std::invalid_argument("mapped Qwen KV requires the shared FCL SoC composition");
  }
  const auto kv_write_mode =
      kv_write_name == "mapped" ? np101::KvWriteMode::MappedHost : np101::KvWriteMode::CopyGraph;
  if (ffn_batch_tokens > 1 && (!shared_fcl || (policy.capacity != 128 && policy.capacity != 512) ||
                               kv_write_name != "mapped")) {
    throw std::invalid_argument("batched Qwen FFN requires a C128/C512 mapped shared FCL chain");
  }
  const auto delta_batch = read_option(request, "delta-batch-tokens.txt", "1");
  if (delta_batch != "1" && delta_batch != "8") {
    throw std::invalid_argument("unsupported Qwen DeltaNet batch width");
  }
  if (delta_batch == "8" &&
      (target != "soc" || ffn_batch_tokens != 8 || state_commit != "ffn" ||
       layout != DecoderLayout::FusedAttentionNorm || !shared_fcl || kv_write_name != "mapped")) {
    throw std::invalid_argument(
        "block DeltaNet requires the SoC M8 fused FFN-commit configuration");
  }
  models::qwen3_5::ModelOptions model_options;
  model_options.delta_batch_tokens = std::stoul(delta_batch);
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
  return {config,
          policy,
          prompt,
          maximum,
          repeats,
          lookup_name,
          token_readback_name,
          head_backend_name,
          state_commit,
          ffn_batch_tokens,
          layout_name,
          matmul_layer,
          projections,
          kv_write_name,
          token_readback,
          model_options};
}
} // namespace specferry::models::qwen3_5
