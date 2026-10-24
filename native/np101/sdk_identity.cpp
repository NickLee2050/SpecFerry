#include "np101/sdk_identity.hpp"
#include "np101/weights.hpp"
#include "platform/runtime_profile.hpp"

#include <filesystem>
#include <set>
#include <stdexcept>

namespace specferry::np101 {
void verify_compiled_sdk(std::istream &manifest, const std::string &profile, unsigned libraries) {
  namespace fs = std::filesystem;
  std::set<std::string> expected{"libovxlib.so", "libOpenVX.so", "libGAL.so", "libArchModelSw.so",
                                 "libNNArchPerf.so"};
  if (libraries != expected.size()) {
    throw std::invalid_argument("compiled package SDK library count differs");
  }
  const auto &target = platform::runtime_profile();
  if (profile != target.sdk_identity) {
    throw std::invalid_argument("compiled package SDK profile differs from build target: " +
                                std::string(target.target));
  }
  platform::verify_driver();
  const fs::path directory(target.sdk_library_directory);
  for (unsigned index = 0; index < libraries; ++index) {
    std::string name, digest;
    manifest >> name >> digest;
    if (!manifest || !expected.erase(name) || file_sha256(directory / name) != digest) {
      throw std::invalid_argument("compiled package SDK identity differs: " + name);
    }
  }
}
} // namespace specferry::np101
