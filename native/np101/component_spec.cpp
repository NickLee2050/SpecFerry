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

void KvSpec::validate() const {
  product(heads, head_dim);
  if (!capacity || capacity > 512) {
    throw std::invalid_argument("KV capacity exceeds validated range [1,512]");
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
