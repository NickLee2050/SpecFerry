#pragma once

#include "models/qwen3_5/compiled.hpp"
#include "np101/context.hpp"
#include "np101/ops/linear_backend.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
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
               const CompiledBlock *compiled_feed_forward = nullptr,
               bool deferred_delta_commit = false);
  ~DecoderLayer();
  DecoderLayer(const DecoderLayer &) = delete;
  DecoderLayer &operator=(const DecoderLayer &) = delete;

  void step();
  void reset();
  std::size_t length() const;
  TensorBinding output();
  std::vector<np101::TensorCopyBinding> state_copies();
  void confirm_state_commit();
  std::vector<std::uint8_t> read(const std::string &name,
                                 np101::TensorReadback method = np101::TensorReadback::SdkCopy);
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace specferry::models::qwen3_5
