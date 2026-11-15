#pragma once

#include "inference/execution.hpp"

namespace specferry::models::opt {
int execute(const inference::ExecutionPaths &paths);
int execute_partitioned(const inference::ExecutionPaths &paths);
} // namespace specferry::models::opt
