#include "np101/sdk_identity.hpp"
#include "np101/weights.hpp"

#include <filesystem>
#include <fstream>
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
  fs::path directory;
  if (profile == "F53A3E6EFBB7D0FD7D4617C") {
    std::ifstream module("/sys/module/galcore/srcversion");
    std::string loaded;
    module >> loaded;
    if (!module || loaded != profile) {
      throw std::invalid_argument("compiled package requires its accepted EP driver");
    }
    directory = "/usr/lib/ljmicro";
  } else if (profile == "np101-soc-riscv64") {
    directory = "/np101_driver/build/sdk/drivers";
    // This vendor build omits sysfs srcversion. Pin its module file as well as
    // the userspace libraries; the run guard also records the active boot/load.
    if (!fs::is_directory("/sys/module/galcore") ||
        file_sha256(directory / "galcore.ko") !=
            "4c9a6ca280791ec23b13759ebee67759dd8a40bdb465536ba0f1dff3f78449bc") {
      throw std::invalid_argument("compiled package requires the pinned SoC driver");
    }
  } else {
    throw std::invalid_argument("unsupported compiled package SDK profile");
  }
  for (unsigned index = 0; index < libraries; ++index) {
    std::string name, digest;
    manifest >> name >> digest;
    if (!manifest || !expected.erase(name) || file_sha256(directory / name) != digest) {
      throw std::invalid_argument("compiled package SDK identity differs: " + name);
    }
  }
}
} // namespace specferry::np101
