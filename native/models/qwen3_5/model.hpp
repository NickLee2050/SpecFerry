#pragma once

#include "inference/generation.hpp"
#include "models/qwen3_5/compiled.hpp"
#include "np101/context.hpp"
#include "np101/kv_cache.hpp"
#include "np101/ops/linear_backend.hpp"
#include "np101/ops/vocabulary.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
// SharedResidual also fuses attention normalization and omits unconsumed scalar publications.
enum class DecoderLayout { Separate, FusedAttentionNorm, SharedResidual };

// Complete text-only model: shared tied vocabulary, 18 compiled DeltaNet mixers,
// six handwritten gated-attention layers, SwiGLU and device greedy selection.
// Missing compiled FFN entries use device projections from the shared weight bank.
// commit_state_in_ffn replaces the combined decode commit and each block's final
// local commit; it requires deferred (combined_state_commit) batched execution.
class Model {
public:
  Model(np101::Context &context, const np101::WeightStore &weights, const Config &config,
        const inference::GenerationPolicy &policy,
        const std::map<unsigned, CompiledBlock> &compiled,
        const std::map<unsigned, CompiledBlock> &compiled_feed_forward,
        const std::function<void(const std::string &)> &phase = {},
        np101::ops::VocabularyLookup lookup = np101::ops::VocabularyLookup::Gather,
        np101::ops::LinearBackend head_backend = np101::ops::LinearBackend::MatMul,
        bool combined_state_commit = false, unsigned head_blocks_per_graph = 0,
        np101::ops::LinearBackend projection_backend = np101::ops::LinearBackend::MatMul,
        np101::KvWriteMode kv_write_mode = np101::KvWriteMode::CopyGraph,
        DecoderLayout layout = DecoderLayout::Separate, bool commit_state_in_ffn = false);
  ~Model();
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  inference::Generation
  generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
           np101::TensorReadback token_readback = np101::TensorReadback::SdkCopy);
  std::vector<std::uint8_t> read(unsigned layer, const std::string &field);
  std::size_t kv_revalidations() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::qwen3_5
