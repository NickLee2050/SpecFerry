#pragma once

#include "inference/generation.hpp"
#include "models/opt/config.hpp"
#include "np101/context.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::opt {
// Fixed embedding/head graphs and one or two decoder layers per graph. Optional
// four-token prefill uses a second graph set with the same weights and KV backing.
class PartitionedGraphModel {
public:
  using PhaseObserver = std::function<void(const std::string &)>;

  // All variants use merged KV and retain shared mutable vocabulary blocks.
  enum class Variant { MergedKv, MergedKvLocalConst, MergedKvLocalMutable };

  // Generation may skip unused prompt predictions; public step() always runs the head.
  // Block-4 supports EveryToken only: each block runs the head for its final column.
  enum class PrefillHeadPolicy { EveryToken, LastToken };

  PartitionedGraphModel(np101::Context &context, const np101::WeightStore &weights,
                        const ModelConfig &config, PhaseObserver phase = {},
                        unsigned decoder_group_size = 1, Variant variant = Variant::MergedKv,
                        PrefillHeadPolicy prefill_head = PrefillHeadPolicy::EveryToken,
                        unsigned prefill_block = 1);
  ~PartitionedGraphModel();
  PartitionedGraphModel(const PartitionedGraphModel &) = delete;
  PartitionedGraphModel &operator=(const PartitionedGraphModel &) = delete;

  void step(std::int32_t token);
  inference::Generation generate(const std::vector<std::int32_t> &prompt, unsigned maximum);
  // Stage reads use the most recently executed graph set; block outputs retain all columns.
  std::vector<std::uint8_t> read_layer(unsigned layer, const std::string &name);
  std::vector<std::uint8_t> read_cache(unsigned layer, bool values);
  // Head outputs describe only the final consumed token, including after a full prefill block.
  std::vector<std::uint8_t> read_output(const std::string &name);
  // Counts successful block/single-token executions, including skipped prompt heads.
  std::size_t launches() const;
  // Single-token executions include prefill tails; prefill_steps counts full blocks only.
  std::size_t decode_steps() const;
  std::size_t prefill_steps() const;
  // Counts successful head executions over the model lifetime; reset does not clear it.
  std::size_t head_runs() const;
  // Uploads required to reset a nonempty cache; the initial empty state needs none.
  std::size_t cache_reset_uploads_per_request() const;
  std::size_t weight_bytes() const;
  Variant variant() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Impl &require_impl() const;
};
} // namespace specferry::models::opt
