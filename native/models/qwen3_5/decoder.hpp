#pragma once

#include "models/qwen3_5/compiled.hpp"
#include "np101/context.hpp"
#include "np101/ops/linear_backend.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace specferry::np101 {
class WeightBank;
}

namespace specferry::models::qwen3_5 {
using np101::Context;
using np101::TensorBinding;
using np101::WeightStore;

// One complete layer, with an immutable input binding and fixed output storage.
// Context and input owner outlive the layer. Execution is synchronous.
class DecoderLayer {
public:
  DecoderLayer(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
               TensorBinding input, np101::WeightBank *shared_weights = nullptr,
               np101::ops::LinearBackend feed_forward = np101::ops::LinearBackend::MatMul,
               const CompiledBlock *compiled_mixer = nullptr,
               const CompiledBlock *compiled_feed_forward = nullptr);
  ~DecoderLayer();
  DecoderLayer(const DecoderLayer &) = delete;
  DecoderLayer &operator=(const DecoderLayer &) = delete;

  void step();
  void reset();
  std::size_t length() const;
  TensorBinding output();
  std::vector<std::uint8_t> read(const std::string &name,
                                 np101::TensorReadback method = np101::TensorReadback::SdkCopy);
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Runs the explicitly selected, ordered Qwen layer slice.
// Only the group boundary accepts host activations. Mid-step failure invalidates
// the entire group: individual recurrent layers cannot be rolled back by length.
class DecoderGroup {
public:
  DecoderGroup(Context &context, const WeightStore &weights, const Config &config,
               const std::vector<unsigned> &layers,
               np101::ops::LinearBackend feed_forward = np101::ops::LinearBackend::MatMul,
               np101::TensorReadback readback = np101::TensorReadback::SdkCopy,
               const std::map<unsigned, CompiledBlock> &compiled_mixers = {});
  ~DecoderGroup();
  DecoderGroup(const DecoderGroup &) = delete;
  DecoderGroup &operator=(const DecoderGroup &) = delete;

  void step(const std::vector<std::uint8_t> &hidden_fp16);
  void reset();
  std::size_t length() const;
  std::vector<std::uint8_t> read(unsigned layer, const std::string &name);
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::qwen3_5
