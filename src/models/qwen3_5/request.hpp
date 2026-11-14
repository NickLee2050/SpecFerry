#pragma once

#include "models/qwen3_5/model.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace specferry::models::qwen3_5 {
// Parsed before Context creation; artifact validation remains with the package loaders.
struct ExecutionRequest {
  Config config;
  inference::GenerationPolicy policy;
  std::vector<std::int32_t> prompt;
  unsigned maximum;
  unsigned repeats;
  std::string lookup_name;
  std::string token_readback_name;
  std::string head_backend_name;
  std::string state_commit;
  unsigned ffn_batch_tokens;
  std::string layout_name;
  int matmul_layer;
  std::string projections;
  std::string kv_write_name;
  np101::TensorReadback token_readback;
  ModelOptions model_options;
};

ExecutionRequest read_execution_request(const std::filesystem::path &request,
                                        std::string_view target);
} // namespace specferry::models::qwen3_5
