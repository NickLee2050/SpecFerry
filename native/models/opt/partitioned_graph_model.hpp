#pragma once

#include "inference/generation.hpp"
#include "models/opt/layer.hpp"
#include "np101/context.hpp"
#include "np101/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::opt {
struct ProjectionPolicy {
  np101::ops::LinearBackend attention = np101::ops::LinearBackend::MatMul;
  np101::ops::LinearBackend head = np101::ops::LinearBackend::MatMul;
  bool compiled_query = false, compiled_output = false;
};

// Fixed embedding/head graphs with device buffers between decoder stages. Compiled
// FFNs use two or three graphs per layer; full decoder NBGs use one plus a shared KV copy graph.
// Block prefill shares ordinary weights and KV. Compiled FFNs require separately
// budgeted fixed-shape artifacts for the prefill and decode execution sets.
// The compiled block-prefill candidate clears KV with a separate fixed copy graph.
// Full-model mapped-slot M1 instead accepts a headless M16 chain with an explicit
// FP16-to-FP32 prefix handoff. It remains resident and uses reset requests only.
class PartitionedGraphModel {
public:
  using PhaseObserver = std::function<void(const std::string &)>;

  // Except full-model NBGs, variants retain shared mutable vocabulary blocks. Compiled decoders use
  // FP32 head-major KV; handwritten decoders use merged FP16 KV. Compiled paths are explicit
  // candidates; the source-bound FFN finish package is available through an explicit CLI option.
  enum class Variant {
    MergedKv,
    MergedKvFc1Split2,
    MergedKvCompiledFfn,
    CompiledDecoder,
    CompiledFullModel
  };

  // Generation may skip unused prompt predictions; public step() always runs the head.
  // EveryToken runs the head for each consumed block's final column.
  enum class PrefillHeadPolicy { EveryToken, LastToken };

  // Independent of EP/SoC. MappedSlot keeps full-model ports in one graph and
  // copies only the current FP32 KV slot through a reusable host buffer.
  enum class KvTransfer { DeviceCopy, MappedSlot };

  PartitionedGraphModel(np101::Context &context, const np101::WeightStore &weights,
                        const ModelConfig &config, PhaseObserver phase = {},
                        Variant variant = Variant::MergedKv,
                        PrefillHeadPolicy prefill_head = PrefillHeadPolicy::EveryToken,
                        unsigned prefill_block = 1, std::vector<CompiledFfn> compiled_ffn = {},
                        std::vector<CompiledDecoder> compiled_decoders = {},
                        std::vector<CompiledFfn> compiled_prefill_ffn = {},
                        ProjectionPolicy projections = {}, CompiledHead compiled_head = {},
                        std::vector<CompiledProjections> compiled_projections = {},
                        CompiledFullModel compiled_full_model = {},
                        KvTransfer kv_transfer = KvTransfer::DeviceCopy,
                        std::vector<CompiledPrefillLayer> compiled_prefill_layers = {});
  ~PartitionedGraphModel();
  PartitionedGraphModel(const PartitionedGraphModel &) = delete;
  PartitionedGraphModel &operator=(const PartitionedGraphModel &) = delete;

  void step(std::int32_t token);
  inference::Generation generate(const std::vector<std::int32_t> &prompt, unsigned maximum,
                                 const inference::GenerationPhaseObserver &on_phase = {});
  // Full token prefix must extend the preceding prompt and every returned output token.
  inference::Generation
  continue_generation(const std::vector<std::int32_t> &prompt, unsigned maximum,
                      const inference::GenerationPhaseObserver &on_phase = {});
  std::vector<std::uint8_t>
  read_cache(unsigned layer, bool values,
             np101::TensorReadback method = np101::TensorReadback::SdkCopy);
  // Diagnostic layout is [head, slot, dimension]; compiled decoders retain FP32 KV.
  np101::TensorSpec cache_spec() const;
  // Head outputs describe only the final consumed token, including after a full prefill block.
  std::vector<std::uint8_t>
  read_output(const std::string &name,
              np101::TensorReadback method = np101::TensorReadback::SdkCopy);
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
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Impl &require_impl() const;
};
} // namespace specferry::models::opt
