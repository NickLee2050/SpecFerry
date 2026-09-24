#include "np101/component_spec.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace specferry::np101 {
namespace {
unsigned product(unsigned left, unsigned right) {
  if (!left || !right || left > std::numeric_limits<unsigned>::max() / right) {
    throw std::invalid_argument("component dimension is zero or overflows");
  }
  return left * right;
}
} // namespace

void validate_cache_append(const TensorSpec &column, const TensorSpec &index,
                           const TensorSpec &storage) {
  if (column.type != DataType::Float16 || storage.type != column.type || column.shape.size() != 2 ||
      column.shape[1] != 1 || storage.shape.size() != 2 || storage.shape[0] != column.shape[0] ||
      index.type != DataType::Int32 || index.shape.size() != 1 || index.shape[0] != 1) {
    throw std::invalid_argument("indexed append requires FP16 [width,1] -> [width,slots], I32[1]");
  }
  column.bytes();
  storage.bytes();
}

void KvSpec::validate() const {
  product(heads, head_dim);
  if (!capacity || capacity > 2048) {
    throw std::invalid_argument("KV capacity exceeds experimental limit [1,2048]");
  }
  tensor().bytes();
}

TensorSpec KvSpec::tensor() const { return {DataType::Float16, {head_dim, capacity, heads}}; }

TensorSpec KvSpec::slot() const { return {DataType::Float16, {head_dim, 1, heads}}; }

std::size_t KvSpec::token_bytes() const { return 2 * slot().bytes(); }

void DeltaSpec::validate() const {
  if (!hidden || convolution_width < 2 || !std::isfinite(epsilon) || epsilon <= 0) {
    throw std::invalid_argument("invalid DeltaNet dimensions or epsilon");
  }
  recurrent().bytes();
  convolution().bytes();
}

unsigned DeltaSpec::key_width() const { return product(heads, key_dim); }

unsigned DeltaSpec::value_width() const { return product(heads, value_dim); }

unsigned DeltaSpec::channels() const {
  const auto keys = product(2, key_width());
  if (value_width() > std::numeric_limits<unsigned>::max() - keys) {
    throw std::invalid_argument("DeltaNet channel count overflows");
  }
  return keys + value_width();
}

TensorSpec DeltaSpec::recurrent() const { return {DataType::Float32, {value_dim, key_dim, heads}}; }

TensorSpec DeltaSpec::convolution() const {
  return {DataType::Float16, {convolution_width, channels()}};
}
} // namespace specferry::np101
