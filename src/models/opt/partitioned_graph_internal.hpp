#pragma once

#include "devices/np101/api/ops/graph_builder.hpp"
#include "devices/np101/api/ops/mixers.hpp"
#include "devices/np101/api/weight_bank.hpp"
#include "models/opt/block_prefill.hpp"
#include "models/opt/partitioned_graph_model.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace specferry::models::opt {
using namespace np101;
using namespace np101::ops;

struct PartitionedGraphModel::Impl {
  struct ExecutionSet {
    explicit ExecutionSet(unsigned tokens)
        : block(tokens), timing_prefix("block" + std::to_string(tokens)) {}

    const unsigned block;
    const std::string timing_prefix;
    std::unique_ptr<GraphBuilder> embedding;
    Tensor controls{}, embedded{}, full_position{};
    std::vector<std::unique_ptr<GraphBuilder>> decoder_graphs;
    std::vector<std::string> decoder_roles;
    std::unique_ptr<GraphBuilder> cache_commit;
    std::unique_ptr<GraphBuilder> head;
    std::unique_ptr<GraphBuilder> selection;
    Tensor head_input{}, projected{}, selected{};
    std::vector<Tensor> logits;
    // The M16 bridge writes the M1 head's fixed input after the decoder stack.
    // Prefill closes before decode, which owns the shared head and input storage.
    ExecutionSet *shared_head = nullptr;

    ExecutionSet &prediction() { return shared_head ? *shared_head : *this; }

    std::string prefix() const { return block == 1 ? "" : "prefill" + std::to_string(block) + "."; }

    void close() {
      if (selection) {
        selection->graph.close();
        selection.reset();
      }
      if (cache_commit) {
        cache_commit->graph.close();
        cache_commit.reset();
      }
      if (head) {
        head->graph.close();
        head.reset();
      }
      while (!decoder_graphs.empty()) {
        decoder_graphs.back()->graph.close();
        decoder_graphs.pop_back();
      }
      if (embedding) {
        embedding->graph.close();
        embedding.reset();
      }
    }
  };

  ModelConfig config;
  Variant variant;
  KvTransfer kv_transfer;
  PrefillHeadPolicy prefill_head_policy;
  unsigned prefill_block;
  const std::vector<CompiledFfn> compiled_ffn;
  const std::vector<CompiledDecoder> compiled_decoders;
  const std::vector<CompiledFfn> compiled_prefill_ffn;
  const std::vector<CompiledPrefillLayer> compiled_prefill_layers;
  const ProjectionPolicy projections;
  const CompiledHead compiled_head;
  const std::vector<CompiledProjections> compiled_projections;
  const CompiledFullModel compiled_full_model;
  Tensor compiled_indices{}, compiled_mask{};
  std::unique_ptr<WeightBank> bank;
  std::unique_ptr<Graph> cache;
  std::vector<Tensor> cache_tensors;
  std::vector<Tensor> mapped_cache_outputs;
  std::vector<std::uint8_t> cache_zeroes, cache_slot;
  Shape slot_start{0, 0, 0}, slot_end;
  inference::GenerationHistory history;
  std::unique_ptr<GraphBuilder> cache_reset;
  std::unique_ptr<ExecutionSet> decode, prefill;
  std::unique_ptr<BlockPrefill> full_model_prefill;
  ExecutionSet *latest = nullptr;
  unsigned position = 0;
  std::size_t completed_steps = 0;
  std::size_t completed_decode_steps = 0;
  std::size_t completed_prefill_steps = 0;
  std::size_t completed_head_runs = 0;
  std::size_t constant_weight_bytes = 0;
  bool prediction_valid = false;
  bool failed = false;
  bool closed = false;
  double single_prefill_seconds = 0, block_prefill_seconds = 0;

  Impl(Context &context, const WeightStore &weights, const ModelConfig &spec,
       const GraphOptions &options, CompiledArtifacts artifacts, const PhaseObserver &phase);
  ~Impl();
  std::pair<Tensor, Tensor> update_merged_cache(GraphBuilder &builder, unsigned layer, Tensor key,
                                                Tensor value, Tensor slot_index, unsigned block);
  Tensor build_attention(GraphBuilder &builder, const WeightStore &weights, unsigned layer,
                         Tensor hidden, Tensor shared_controls, unsigned block,
                         const Projections *provided = nullptr);
  Tensor build_decoder(GraphBuilder &builder, const WeightStore &weights, unsigned layer,
                       Tensor hidden, Tensor shared_controls, unsigned block);
  void build(Context &context, const WeightStore &weights, const PhaseObserver &phase);
  void build_cache_reset(Context &context, const PhaseObserver &phase);
  void build_execution_set(ExecutionSet &execution, Context &context, const WeightStore &weights,
                           const PhaseObserver &phase);
  void build_full_execution(ExecutionSet &execution, Context &context, const PhaseObserver &phase);
  Graph &cache_storage();
  void clear_mapped_cache();
  void commit_mapped_slot();
  void require_open() const;
  void run(GraphBuilder &builder, const std::string &role);
  void consume(ExecutionSet &execution, const std::vector<std::int32_t> &tokens, bool run_head);
  void step(std::int32_t token);
  void consume_prompt(const std::vector<std::int32_t> &tokens);
  void reset();
  std::int32_t predict();
  void close();
};
} // namespace specferry::models::opt
