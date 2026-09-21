#pragma once

#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace specferry::np101 {
// Qwen3.5-0.8B mixer, excluding decoder input norm, residual and MLP.
// Context must outlive this object. All graph execution and state routing is C++.
class DeltaNet {
public:
  static constexpr unsigned hidden_size = 1024;
  static constexpr unsigned heads = 16;
  static constexpr unsigned head_size = 128;
  static constexpr unsigned convolution_channels = 6144;
  static constexpr unsigned convolution_width = 4;

  DeltaNet(Context &context, const WeightStore &weights);
  DeltaNet(Context &context, const WeightStore &weights, unsigned layer, TensorBinding input,
           TensorBinding output);
  ~DeltaNet();

  DeltaNet(const DeltaNet &) = delete;
  DeltaNet &operator=(const DeltaNet &) = delete;

  // Empty initial buffers mean zero. Reset is an explicit host initialization;
  // step never downloads/reuploads recurrent state or convolution history.
  void reset(const std::vector<std::uint8_t> &recurrent = {},
             const std::vector<std::uint8_t> &convolution = {});
  void step(const std::vector<std::uint8_t> &hidden_fp16);
  // Execute the construction-time input/output bindings without host activation IO.
  void step();
  std::vector<std::uint8_t> read(const std::string &name);
  std::size_t steps() const;
  void close();

  static TensorSpec recurrent_spec();
  static TensorSpec convolution_spec();
  static TensorSpec output_spec(const std::string &name);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::np101
