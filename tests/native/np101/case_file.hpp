#pragma once

#include "np101/tensor_spec.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace specferry::testing {
struct TensorDefinition {
  std::string name;
  np101::TensorSpec spec;
  std::string storage;
  std::string initial_file;
};

struct NodeDefinition {
  std::string operation;
  std::vector<std::string> inputs;
  std::string output;
  std::vector<std::uint32_t> parameters;
};

struct InputSequence {
  std::string tensor;
  std::vector<std::string> files;
};

struct IntegerBounds {
  std::string tensor;
  std::int32_t minimum;
  std::int32_t maximum;
};

struct CaseDefinition {
  std::filesystem::path root;
  std::vector<TensorDefinition> tensors;
  std::vector<NodeDefinition> nodes;
  std::vector<InputSequence> inputs;
  std::vector<std::string> outputs;
  std::size_t steps = 1;
  std::vector<IntegerBounds> integer_bounds;
};

CaseDefinition load_case(const std::filesystem::path &path);

// Validate every supplied byte buffer and dynamic index before opening the device.
void validate_case_data(const CaseDefinition &test);

std::vector<std::uint8_t> read_bytes(const std::filesystem::path &root, const std::string &name,
                                     std::size_t expected_bytes);
} // namespace specferry::testing
