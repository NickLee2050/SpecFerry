#pragma once

#include "inference/generation.hpp"
#include "models/opt/config.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/vocabulary.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::opt {
class InputEmbedding : public np101::ops::GraphBuilder {
public:
  InputEmbedding(np101::Context &context, const np101::WeightStore &weights,
                 const ModelConfig &config, np101::ops::Vocabulary &table);
  void run(std::int32_t token, unsigned position);
  np101::TensorBinding binding();
  std::vector<std::uint8_t> read();

private:
  ModelConfig config_;
  np101::ops::Vocabulary &table_;
  np101::ops::Tensor position_, output_;
};

class OutputProjection : public np101::ops::GraphBuilder {
public:
  OutputProjection(np101::Context &context, const np101::WeightStore &weights,
                   const ModelConfig &config, np101::TensorBinding hidden);
  void run();
  np101::TensorBinding binding();

private:
  np101::ops::Tensor output_;
};

using Generation = inference::Generation;

// Fixed-capacity, batch-one resident model. Host code supplies token IDs only.
// Partial layer prefixes exist for diagnostics; generation requires every layer.
class Model {
public:
  Model(np101::Context &context, const np101::WeightStore &weights, const ModelConfig &config,
        unsigned layers, np101::ops::SamplingOptions sampling = {});
  ~Model();
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  void consume(std::int32_t token);
  std::int32_t predict();
  void reset();
  unsigned length() const;
  std::size_t cache_writes() const;
  double cache_write_seconds() const;
  double cache_revalidation_seconds() const;
  std::vector<std::uint8_t> read(const std::string &name, unsigned layer = 0);
  Generation generate(const std::vector<std::int32_t> &prompt, unsigned maximum_new_tokens,
                      const std::function<void(std::int32_t)> &on_token = {});
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void require_live() const;
};
} // namespace specferry::models::opt
