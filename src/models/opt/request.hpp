#pragma once

#include "inference/request.hpp"
#include "models/opt/config.hpp"
#include "models/opt/partitioned_graph_model.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace specferry::models::opt {
struct GraphRequest {
  inference::RunOptions options;
  ModelConfig config;
  GraphOptions model_options;
  np101::TensorReadback readback_method;
  std::vector<std::vector<std::int32_t>> requests;
  std::vector<std::vector<std::int32_t>> expected;
};

GraphRequest read_graph_request(const std::filesystem::path &request);
} // namespace specferry::models::opt
