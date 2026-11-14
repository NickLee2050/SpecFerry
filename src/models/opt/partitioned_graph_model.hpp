#pragma once

#include "devices/np101/api/context.hpp"
#include "devices/np101/api/tensor.hpp"
#include "inference/generation.hpp"
#include "models/opt/layer.hpp"

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

// Except full-model NBGs, variants retain shared mutable vocabulary blocks. Compiled decoders use
// FP32 head-major KV; handwritten decoders use merged FP16 KV. Compiled paths are explicit
// candidates; the source-bound FFN finish package is available through an explicit CLI option.
enum class GraphVariant {
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

// Execution choices are independent of artifact identities and runtime diagnostics.
struct GraphOptions {
  GraphVariant variant = GraphVariant::MergedKv;
  PrefillHeadPolicy prefill_head = PrefillHeadPolicy::EveryToken;
  unsigned prefill_block = 1;
  ProjectionPolicy projections;
  KvTransfer kv_transfer = KvTransfer::DeviceCopy;
};

// Source-validated artifact metadata, owned by the model after construction.
// This contains paths and contracts, not another copy of device weight storage.
struct CompiledArtifacts {
  std::vector<CompiledFfn> feed_forward;
  std::vector<CompiledDecoder> decoders;
  std::vector<CompiledFfn> prefill_feed_forward;
  CompiledHead head;
  std::vector<CompiledProjections> projections;
  CompiledFullModel full_model;
  std::vector<CompiledPrefillLayer> prefill_layers;
};

// Fixed embedding/head graphs with device buffers between decoder stages. Compiled
// FFNs use two or three graphs per layer; full decoder NBGs use one plus a shared KV copy graph.
// Block prefill shares ordinary weights and KV. Compiled FFNs require separately
// budgeted fixed-shape artifacts for the prefill and decode execution sets.
// The compiled block-prefill candidate clears KV with a separate fixed copy graph.
// Full-model mapped-slot M1 accepts a headless M16 chain. Compiled attention shares
// its FP32 inputs; handwritten attention widens and transfers its independent KV.
// Both remain resident and use reset requests only.
class PartitionedGraphModel {
public:
  using PhaseObserver = std::function<void(const std::string &)>;

  using Variant = GraphVariant;
  using PrefillHeadPolicy = opt::PrefillHeadPolicy;
  using KvTransfer = opt::KvTransfer;

  PartitionedGraphModel(np101::Context &context, const np101::WeightStore &weights,
                        const ModelConfig &config, const GraphOptions &options = {},
                        CompiledArtifacts artifacts = {}, PhaseObserver phase = {});
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
  bool shares_prefill_cache() const;
  std::size_t weight_bytes() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Impl &require_impl() const;
};
} // namespace specferry::models::opt
