#include "np101/weights.hpp"
#include "platform/runtime_profile.hpp"

#include <stdexcept>

namespace specferry::platform {
const RuntimeProfile &runtime_profile() {
  static const RuntimeProfile profile{"soc", "riscv64", "np101-soc-riscv64",
                                      "/np101_driver/build/sdk/drivers"};
  return profile;
}

void verify_driver() {
  // This vendor build omits sysfs srcversion. Pin its module file as well as
  // the userspace libraries; the run guard also records the active boot/load.
  const std::filesystem::path directory(runtime_profile().sdk_library_directory);
  if (!std::filesystem::is_directory("/sys/module/galcore") ||
      np101::file_sha256(directory / "galcore.ko") !=
          "4c9a6ca280791ec23b13759ebee67759dd8a40bdb465536ba0f1dff3f78449bc") {
    throw std::invalid_argument("compiled package requires the pinned SoC driver");
  }
}
} // namespace specferry::platform
