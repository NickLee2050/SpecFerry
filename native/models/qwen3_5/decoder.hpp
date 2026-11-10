#pragma once

#include "models/qwen3_5/compiled.hpp"
#include "np101/context.hpp"
#include "np101/kv_cache.hpp"
#include "np101/ops/linear_backend.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
               bool deferred_delta_commit = false,
               np101::KvWriteMode kv_write_mode = np101::KvWriteMode::CopyGraph,
               bool fuse_attention_input_norm = false,
               std::optional<TensorBinding> residual_block = {}, bool commit_state_in_ffn = false);
  ~DecoderLayer();
  DecoderLayer(const DecoderLayer &) = delete;
  DecoderLayer &operator=(const DecoderLayer &) = delete;

  void step();
  unsigned feed_forward_batch_tokens() const;
  // Packed FP16 [count, hidden] rows. Only real rows advance mixer state.
  // With a shared residual block, these rows must match its current valid rows.
  // Returns valid output rows. Unless publication is explicitly deferred by the
  // model scheduler, publishes the last row through output().
  std::vector<std::uint8_t> prefill_block(const std::vector<std::uint8_t> &packed_rows,
                                          unsigned count, bool publish_output = true);
  void reset();
  std::size_t length() const;
  std::size_t kv_revalidations() const;
  TensorBinding output();
  // Fixed full-batch output for the next layer's residual; owner outlives consumers.
  TensorBinding batch_output();
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
