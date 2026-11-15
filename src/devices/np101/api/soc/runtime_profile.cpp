#include "devices/np101/api/runtime_profile.hpp"
#include "devices/np101/api/weights.hpp"

#include <stdexcept>

namespace specferry::platform {
bool reuse_prefill_cache_outputs() { return true; }

unsigned prefill_embedding_block_rows(unsigned configured_rows) {
  return configured_rows == 4096 ? 8192 : configured_rows;
}

bool verify_primary_model_last() { return true; }

bool verify_model_payloads() { return false; }

unsigned pointwise_prefill_layers() { return 4; }

const RuntimeProfile &runtime_profile() {
  static const RuntimeProfile profile{"soc", "riscv64", "np101-soc-riscv64",
                                      "/np101_driver/build/sdk/drivers"};
  return profile;
}

void verify_driver() {
  // These vendor builds omit sysfs srcversion. Keep the accepted July image and
  // replacement April image explicit; packages still pin all userspace libraries.
  // The run guard also records the active boot/load and enforces recovery controls.
  const std::filesystem::path directory(runtime_profile().sdk_library_directory);
  if (!std::filesystem::is_directory("/sys/module/galcore")) {
    throw std::invalid_argument("compiled package requires a loaded SoC driver");
  }
  const auto digest = np101::file_sha256(directory / "galcore.ko");
  if (digest != "4c9a6ca280791ec23b13759ebee67759dd8a40bdb465536ba0f1dff3f78449bc" &&
      digest != "d19f6d06ba1b8017cb44fb3280f62f9c1a0a0ef04c299c00867040cc9ae8de77") {
    throw std::invalid_argument("compiled package requires the pinned SoC driver");
  }
}
} // namespace specferry::platform
