#pragma once

#include "models/opt/layer.hpp"
#include "np101/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::opt {
// Headless M16 execution shares the M1 KV inputs.
class BlockPrefill {
public:
  using PhaseObserver = std::function<void(const std::string &)>;

  BlockPrefill(np101::Context &context, const np101::WeightStore &weights,
               const ModelConfig &config, const std::vector<CompiledPrefillLayer> &compiled_layers,
               PhaseObserver phase, const std::vector<np101::TensorBinding> &shared_cache);
  ~BlockPrefill();
  BlockPrefill(const BlockPrefill &) = delete;
  BlockPrefill &operator=(const BlockPrefill &) = delete;

  // Exactly sixteen tokens per call; callers retain at least one prompt token for M1.
  void consume(const std::vector<std::int32_t> &tokens);
  void reset();
  std::size_t weight_bytes() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Impl &require_impl() const;
};
} // namespace specferry::models::opt
