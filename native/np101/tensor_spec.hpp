#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace specferry::np101 {
enum class DataType { Float16, Float32, Int32, Bool8 };

DataType parse_dtype(const std::string &name);

std::string dtype_name(DataType type);

std::size_t dtype_bytes(DataType type);

// SDK dimensions are ordered from the contiguous axis to the outermost axis.
struct TensorSpec {
  DataType type;
  std::vector<std::uint32_t> shape;

  std::size_t elements() const;

  std::size_t bytes() const;
};

std::vector<std::uint32_t> parse_shape(const std::string &text);
} // namespace specferry::np101
