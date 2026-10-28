#include "models/opt/prefill_graph_cache.hpp"
#include "platform/runtime_profile.hpp"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
void attach_prefill_graph_caches(const std::filesystem::path &package, const ModelConfig &config,
                                 const np101::WeightStore &weights,
                                 std::vector<CompiledPrefillLayer> &layers) {
  namespace fs = std::filesystem;
  const auto manifest = package / "verified-prefill-cache.txt";
  if (!fs::exists(manifest)) {
    return;
  }
  for (const auto *name :
       {"VIV_VX_ENABLE_CACHE_GRAPH_BINARY", "VIV_VX_CACHE_BINARY_GRAPH_DIR",
        "VIV_VX_CACHE_BINARY_GRAPH_MAX_SIZEKB", "VIV_VX_CACHE_BINARY_GRAPH_MAX_COUNT"}) {
    if (std::getenv(name)) {
      throw std::invalid_argument("package-controlled cache rejects inherited SDK cache settings");
    }
  }
  std::ifstream stream(manifest);
  std::string magic, recipe, consumer, producer, producer_digest, sdk, extra;
  unsigned schema = 0, count = 0, hidden = 0, intermediate = 0, tokens = 0, capacity = 0,
           libraries = 0;
  stream >> magic >> schema >> recipe >> count >> hidden >> intermediate >> tokens >> capacity;
  stream >> consumer >> std::quoted(producer) >> producer_digest >> sdk >> libraries;
  if (!stream || magic != "specferry-opt-verify-cache" || schema != 1 ||
      recipe != "batch-pointwise-f16-fp32-norm-v1" || (count != 12 && count != 19) ||
      platform::pointwise_prefill_layers(count) != count || config.decoder.hidden != hidden ||
      hidden != 1024 || config.decoder.intermediate != intermediate || intermediate != 4096 ||
      tokens != 16 || capacity != config.decoder.capacity || (capacity != 128 && capacity != 256) ||
      config.decoder.epsilon != 1e-5f || layers.size() != 24 ||
      np101::file_sha256("/proc/self/exe") != consumer) {
    throw std::invalid_argument("verified prefill cache program/recipe/geometry differs");
  }
  const auto &profile = platform::runtime_profile();
  if (sdk != profile.sdk_identity) {
    throw std::invalid_argument("verified cache SDK profile differs");
  }
  platform::verify_driver();
  const fs::path sdk_directory(profile.sdk_library_directory);
  std::set<std::string> expected_libraries{"galcore.ko"};
  for (const auto &entry : fs::directory_iterator(sdk_directory)) {
    if (entry.path().extension() == ".so") {
      expected_libraries.insert(entry.path().filename().string());
    }
  }
  if (libraries != expected_libraries.size()) {
    throw std::invalid_argument("verified cache SDK library set differs");
  }
  for (unsigned index = 0; index < libraries; ++index) {
    std::string name, digest;
    stream >> name >> digest;
    if (!stream || !expected_libraries.erase(name) ||
        np101::file_sha256(sdk_directory / name) != digest) {
      throw std::invalid_argument("verified cache SDK library differs: " + name);
    }
  }
  const auto producer_path = fs::canonical(package / producer);
  if (fs::path(producer).filename() != producer || producer_path.parent_path() != package ||
      fs::is_symlink(package / producer)) {
    throw std::invalid_argument("verified cache producer path differs");
  }
  std::vector<np101::FileChecksum> checksums{{producer_path, producer_digest}};
  const auto root = fs::canonical(package / "verify-cache");
  if (root.parent_path() != package || fs::is_symlink(package / "verify-cache")) {
    throw std::invalid_argument("verified cache root escapes its package");
  }
  const std::array<std::string, 6> names{
      "fc1.weight",           "fc1.bias", "fc2.weight", "fc2.bias", "final_layer_norm.weight",
      "final_layer_norm.bias"};
  for (unsigned index = 0; index < count; ++index) {
    unsigned layer = 0;
    std::string filename;
    np101::ops::VerifiedGraphCache cache;
    stream >> layer >> std::quoted(filename) >> cache.bytes >> cache.sha256;
    const auto directory = root / ("layer-" + std::to_string(index));
    if (!stream || layer != index || layers[index].layer != index || layers[index].full_decoder ||
        layers[index].graph.graph_version != std::array<unsigned, 3>{1, 1, 35} ||
        fs::is_symlink(directory) || fs::canonical(directory).parent_path() != root ||
        fs::path(filename).filename() != filename || fs::path(filename).extension() != ".nb") {
      throw std::invalid_argument("verified cache layer/version/path differs");
    }
    for (const auto &suffix : names) {
      std::string digest;
      stream >> digest;
      if (!stream || digest != weights.find(config.decoder.prefix(layer) + suffix).sha256) {
        throw std::invalid_argument("verified cache source weight differs: " + suffix);
      }
    }
    cache.path = directory / filename;
    np101::ops::validate_graph_cache(cache);
    checksums.push_back({cache.path, cache.sha256});
    layers[index].verify_cache = std::move(cache);
  }
  if (stream >> extra) {
    throw std::invalid_argument("trailing verified prefill cache metadata");
  }
  np101::verify_files(checksums);
}
} // namespace specferry::models::opt
