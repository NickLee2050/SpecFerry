#pragma once

#include "models/opt/layer.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::opt {
// Convert packed FP16 [slot,head,width] into FP32 [head,slot,width]. The exact
// widening preserves half subnormals and signed zero without SDK conversion.
std::vector<std::uint8_t> widen_prefill_cache(const std::vector<std::uint8_t> &packed,
                                              unsigned positions, unsigned heads, unsigned width);

// Resident, headless M16 execution for an initial prompt. Its independent cache
// is FP16 for handwritten attention or FP32 for compiled attention. The prefix
// is handed to M1 once; this chain does not consume later decode state.
class BlockPrefill {
public:
  using PhaseObserver = std::function<void(const std::string &)>;

  BlockPrefill(np101::Context &context, const np101::WeightStore &weights,
               const ModelConfig &config, std::vector<CompiledFfn> compiled_ffn,
               PhaseObserver phase = {}, std::vector<CompiledPrefillLayer> compiled_layers = {});
  ~BlockPrefill();
  BlockPrefill(const BlockPrefill &) = delete;
  BlockPrefill &operator=(const BlockPrefill &) = delete;

  // Exactly sixteen tokens per call; callers retain at least one prompt token for M1.
  void consume(const std::vector<std::int32_t> &tokens);
  void reset();
  unsigned position() const;
  // Packed FP32 [head,position,width], containing only the populated prefix.
  std::vector<std::uint8_t> read_cache(unsigned layer, bool values);
  std::size_t constant_weight_bytes() const;
  std::size_t weight_bytes() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Impl &require_impl() const;
};
} // namespace specferry::models::opt
