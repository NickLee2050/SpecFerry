#pragma once

#include "np101/tensor_spec.hpp"

#include <cstddef>

namespace specferry::np101 {
struct KvSpec {
  unsigned heads;
  unsigned head_dim;
  unsigned capacity;

  void validate() const;
  TensorSpec tensor() const;
  TensorSpec slot() const;
  std::size_t token_bytes() const;
};

struct DeltaSpec {
  unsigned hidden;
  unsigned heads;
  unsigned key_dim;
  unsigned value_dim;
  unsigned convolution_width;
  float epsilon;

  void validate() const;
  unsigned key_width() const;
  unsigned value_width() const;
  unsigned channels() const;
  TensorSpec recurrent() const;
  TensorSpec convolution() const;
};
} // namespace specferry::np101
