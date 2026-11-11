#pragma once

#include "devices/np101/api/ops/graph_builder.hpp"
#include "devices/np101/api/tensor.hpp"
#include "devices/np101/api/weight_bank.hpp"
#include "models/qwen3_5/decoder.hpp"
#include "models/qwen3_5/model.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
using namespace np101;
using namespace np101::ops;

struct Model::Impl {
  Config config;
  inference::GenerationPolicy policy;
  WeightBank bank;
  std::unique_ptr<Vocabulary> vocabulary;
  std::vector<std::unique_ptr<DecoderLayer>> layers;
  std::unique_ptr<GraphBuilder> state_commit;
  std::unique_ptr<GraphBuilder> normalization;
  std::unique_ptr<VocabularyHead> head;
  unsigned position = 0, batch_tokens = 1;
  bool failed = false, prediction_ready = false;
  std::function<void(const std::string &)> progress;
  bool trace_steps = false;

  Impl(Context &context, const WeightStore &weights, const Config &spec,
       const inference::GenerationPolicy &selection,
       const std::map<unsigned, CompiledBlock> &compiled,
       const std::map<unsigned, CompiledBlock> &compiled_feed_forward, const ModelOptions &options,
       const std::function<void(const std::string &)> &phase);
  ~Impl();
  void require_live() const;
  void consume(std::int32_t token);
  void prefill(const std::vector<std::int32_t> &tokens);
  std::int32_t predict(TensorReadback token_readback);
  void reset();
  void close();
};
} // namespace specferry::models::qwen3_5
