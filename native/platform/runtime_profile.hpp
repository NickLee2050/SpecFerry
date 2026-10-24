#pragma once

#include <filesystem>

namespace specferry::platform {
struct RuntimeProfile {
  const char *target;
  const char *architecture;
  const char *sdk_identity;
  const char *sdk_library_directory;
};

// CMake links exactly one platform implementation. Neither call initializes the SDK.
const RuntimeProfile &runtime_profile();
void verify_driver();

// Record the linked platform before initialization, outside inference timings.
void write_runtime_profile(const std::filesystem::path &output_directory);
} // namespace specferry::platform
