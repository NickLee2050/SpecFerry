#include "platform/runtime_profile.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace specferry::platform {
void write_runtime_profile(const std::filesystem::path &output_directory) {
  const auto &profile = runtime_profile();
  std::ofstream output(output_directory / "platform.json");
  output << "{\"target\":\"" << profile.target << "\",\"architecture\":\"" << profile.architecture
         << "\",\"sdk_identity\":\"" << profile.sdk_identity << "\",\"sdk_library_directory\":\""
         << profile.sdk_library_directory << "\"}\n";
  output.close();
  if (!output) {
    throw std::runtime_error("cannot save runtime platform profile");
  }
  std::clog << "SpecFerry target=" << profile.target << " architecture=" << profile.architecture
            << " sdk=" << profile.sdk_identity << '\n';
}
} // namespace specferry::platform
