#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace specferry::supervision {
struct ProcessOptions {
  std::filesystem::path output;
  std::filesystem::path working_directory;
  std::optional<double> preparation_seconds;
  double execution_seconds = 300;
  unsigned execution_limit = 300;
  std::vector<std::string> command;
};

// stdin is a private, nonblocking stop channel from the policy adapter. The model gets /dev/null.
// Budgets are enforced independently of policy sampling and filesystem publication.
int supervise(const ProcessOptions &options);
} // namespace specferry::supervision
