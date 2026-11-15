#pragma once

#include <filesystem>

namespace specferry::inference {
struct ExecutionPaths {
  std::filesystem::path deployment;
  std::filesystem::path request;
  std::filesystem::path output;
};
} // namespace specferry::inference
