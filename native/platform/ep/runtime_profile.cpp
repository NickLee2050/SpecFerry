#include "platform/runtime_profile.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

namespace specferry::platform {
const RuntimeProfile &runtime_profile() {
  static const RuntimeProfile profile{"ep", "x86_64", "F53A3E6EFBB7D0FD7D4617C",
                                      "/usr/lib/ljmicro"};
  return profile;
}

void verify_driver() {
  std::ifstream module("/sys/module/galcore/srcversion");
  std::string loaded;
  module >> loaded;
  if (!module || loaded != runtime_profile().sdk_identity) {
    throw std::invalid_argument("compiled package requires its accepted EP driver");
  }
}
} // namespace specferry::platform
