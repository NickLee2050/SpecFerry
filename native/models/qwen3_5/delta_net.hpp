#pragma once

#include "models/qwen3_5/config.hpp"
#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
// Configured Qwen DeltaNet mixer, excluding decoder input norm, residual and MLP.
// Context must outlive this object. All graph execution and state routing is C++.
using np101::Context;
using np101::TensorBinding;
using np101::TensorSpec;
using np101::WeightStore;

class DeltaNet {
public:
  DeltaNet(Context &context, const WeightStore &weights, const Config &config, unsigned layer);
  DeltaNet(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
           TensorBinding input, TensorBinding output);
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

  TensorSpec recurrent_spec() const;
  TensorSpec convolution_spec() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::qwen3_5
