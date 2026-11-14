#pragma once

#include "devices/np101/api/context.hpp"
#include "devices/np101/api/kv_cache.hpp"
#include "devices/np101/api/ops/linear_backend.hpp"
#include "devices/np101/api/ops/vocabulary.hpp"
#include "inference/generation.hpp"
#include "models/qwen3_5/compiled.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
enum class DecoderLayout { Separate, FusedAttentionNorm };

// Graph construction choices are explicit and fixed for the model lifetime.
// Batch width and tensor contracts come from the source-bound NBG package.
enum class StateCommit { PerLayer, Combined, FeedForward };
enum class HistoryMode { Reset, Retain };

struct ModelOptions {
  np101::ops::VocabularyLookup lookup = np101::ops::VocabularyLookup::Gather;
  np101::ops::LinearBackend head = np101::ops::LinearBackend::MatMul;
  unsigned head_blocks_per_graph = 0;
  np101::ops::LinearBackend projections = np101::ops::LinearBackend::MatMul;
  np101::KvWriteMode kv_write = np101::KvWriteMode::CopyGraph;
  DecoderLayout layout = DecoderLayout::Separate;
  StateCommit state_commit = StateCommit::PerLayer;
  unsigned delta_batch_tokens = 1;
};

// Complete text-only model: shared tied vocabulary, 18 compiled DeltaNet mixers,
// six handwritten gated-attention layers, SwiGLU and device greedy selection.
// Missing compiled FFN entries use device projections from the shared weight bank.
// FeedForward state commit replaces combined decode and block-final local copies;
// it requires a batched FFN package. Graph storage outlives all consumers.
class Model {
public:
  Model(np101::Context &context, const np101::WeightStore &weights, const Config &config,
        const inference::GenerationPolicy &policy,
        const std::map<unsigned, CompiledBlock> &compiled,
        const std::map<unsigned, CompiledBlock> &compiled_feed_forward,
        const ModelOptions &options = {},
        const std::function<void(const std::string &)> &phase = {},
        const std::map<unsigned, CompiledBlock> &compiled_prefill = {});
  ~Model();
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  // Retain requires the complete, unchanged token prefix from the previous call,
  // including its final prediction. KV and recurrent/convolution state stay owned here.
  // Reset is the default; an executor failure invalidates retained history.
  inference::Generation
  generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
           np101::TensorReadback token_readback = np101::TensorReadback::SdkCopy,
           HistoryMode history = HistoryMode::Reset);
  std::vector<std::uint8_t> read(unsigned layer, const std::string &field);
  std::size_t kv_revalidations() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::qwen3_5
