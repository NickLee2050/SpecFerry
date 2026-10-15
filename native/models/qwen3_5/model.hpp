#pragma once

#include "inference/generation.hpp"
#include "models/qwen3_5/compiled.hpp"
#include "np101/context.hpp"
#include "np101/ops/vocabulary.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
// Complete text-only model: shared tied vocabulary, 18 compiled DeltaNet mixers,
// six handwritten gated-attention layers, SwiGLU and device greedy selection.
// Missing compiled FFN entries use the existing device MatMul path and shared weight bank.
class Model {
public:
  Model(np101::Context &context, const np101::WeightStore &weights, const Config &config,
        const inference::GenerationPolicy &policy,
        const std::map<unsigned, CompiledBlock> &compiled,
        const std::map<unsigned, CompiledBlock> &compiled_feed_forward,
        const std::function<void(const std::string &)> &phase = {},
        np101::ops::VocabularyLookup lookup = np101::ops::VocabularyLookup::Gather);
  ~Model();
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  inference::Generation
  generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
           np101::TensorReadback token_readback = np101::TensorReadback::SdkCopy);
  std::vector<std::uint8_t> read(unsigned layer, const std::string &field);
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::qwen3_5
