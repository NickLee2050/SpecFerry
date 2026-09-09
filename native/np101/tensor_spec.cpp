#include "np101/tensor_spec.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace specferry::np101 {
DataType parse_dtype(const std::string &name) {
  if (name == "F16") {
    return DataType::Float16;
  }
  if (name == "F32") {
    return DataType::Float32;
  }
  if (name == "I32") {
    return DataType::Int32;
  }
  if (name == "BOOL") {
    return DataType::Bool8;
  }
  throw std::invalid_argument("unsupported dtype: " + name);
}

std::string dtype_name(DataType type) {
  switch (type) {
  case DataType::Float16:
    return "F16";
  case DataType::Float32:
    return "F32";
  case DataType::Int32:
    return "I32";
  case DataType::Bool8:
    return "BOOL";
  }
  throw std::invalid_argument("invalid dtype");
}

std::size_t dtype_bytes(DataType type) {
  switch (type) {
  case DataType::Float16:
    return 2;
  case DataType::Float32:
  case DataType::Int32:
    return 4;
  case DataType::Bool8:
    return 1;
  }
  throw std::invalid_argument("invalid dtype");
}

std::size_t TensorSpec::elements() const {
  if (shape.empty() || shape.size() > 8) {
    throw std::invalid_argument("tensor rank must be in [1, 8]");
  }
  std::size_t count = 1;
  for (auto dimension : shape) {
    if (dimension == 0 || count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::invalid_argument("zero dimension or tensor size overflow");
    }
    count *= dimension;
  }
  return count;
}

std::size_t TensorSpec::bytes() const {
  auto count = elements();
  auto width = dtype_bytes(type);
  if (count > std::numeric_limits<std::size_t>::max() / width) {
    throw std::invalid_argument("tensor byte size overflow");
  }
  return count * width;
}

std::vector<std::uint32_t> parse_shape(const std::string &text) {
  std::vector<std::uint32_t> shape;
  std::istringstream stream(text);
  std::string dimension;
  if (text.empty() || text.back() == ',') {
    throw std::invalid_argument("empty tensor dimension");
  }
  while (std::getline(stream, dimension, ',')) {
    if (dimension.empty() || dimension.find_first_not_of("0123456789") != std::string::npos) {
      throw std::invalid_argument("invalid tensor dimension: " + dimension);
    }
    auto value = std::stoull(dimension);
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("tensor dimension outside uint32 range");
    }
    shape.push_back(static_cast<std::uint32_t>(value));
  }
  TensorSpec{DataType::Bool8, shape}.bytes();
  return shape;
}
} // namespace specferry::np101
