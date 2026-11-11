#include "devices/np101/api/runtime_profile.hpp"
#include "unistd.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace specferry::platform {
void notify_model_ready() {
  const auto *value = std::getenv("SPECFERRY_READY_FD");
  if (!value) {
    return;
  }
  const std::string descriptor(value);
  if (descriptor.empty() || descriptor.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("invalid supervisor readiness descriptor");
  }
  const auto now =
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  const auto record = std::to_string(now) + '\n';
  const auto fd = std::stoi(descriptor);
  ssize_t written;
  do {
    written = ::write(fd, record.data(), record.size());
  } while (written < 0 && errno == EINTR);
  ::close(fd);
  if (written != static_cast<ssize_t>(record.size())) {
    throw std::runtime_error("cannot notify supervisor of model readiness");
  }
}

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
