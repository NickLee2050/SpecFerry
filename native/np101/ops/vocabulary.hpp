#pragma once

#include "np101/ops/graph_builder.hpp"

#include <memory>

namespace specferry::np101::ops {
struct SamplingOptions {
  bool enabled = false;
  std::uint32_t seed = 0;
};

// One resident table, split into bounded row blocks. Lookup and head graphs retain
// the same ordinary tensors; they never reload weights during token execution.
class Vocabulary {
public:
  Vocabulary(Context &context, const WeightStore &store, const WeightRecord &table,
             unsigned block_rows = 4096);
  ~Vocabulary();
  Vocabulary(const Vocabulary &) = delete;
  Vocabulary &operator=(const Vocabulary &) = delete;

  void lookup(std::int32_t token);
  TensorBinding embedding();
  TensorBinding block(unsigned index);
  TensorSpec block_spec(unsigned index) const;
  unsigned blocks() const;
  unsigned rows() const;
  unsigned width() const;
  unsigned block_rows() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Greedy mode reduces ordered block maxima. Sampling mode concatenates all valid
// logits and samples softmax(logits) through the verified FP32 SDK path.
class VocabularyHead : public GraphBuilder {
public:
  VocabularyHead(Context &context, Vocabulary &table, TensorBinding input,
                 SamplingOptions sampling = {});
  std::int32_t select();
  void reset();
  std::vector<std::uint8_t> read_logits(); // Diagnostic only.

private:
  unsigned vocabulary_size_;
  SamplingOptions sampling_;
  std::uint32_t draws_ = 0;
  Tensor seed_;
  Tensor token_;
  std::vector<Tensor> logits_;
};
} // namespace specferry::np101::ops
