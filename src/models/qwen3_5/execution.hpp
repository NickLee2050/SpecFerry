#pragma once

#include "inference/execution.hpp"

namespace specferry::models::qwen3_5 {
int execute(const inference::ExecutionPaths &paths, const std::filesystem::path &compiled);
} // namespace specferry::models::qwen3_5
