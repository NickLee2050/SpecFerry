#pragma once

#include "models/qwen3_5/config.hpp"
#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::qwen3_5 {
using np101::Context;
using np101::TensorBinding;
using np101::WeightStore;

struct DecoderMetrics {
  std::size_t steps = 0;
  std::size_t cache_writes = 0;
  std::size_t cache_revalidations = 0;
  double cache_write_seconds = 0;
  double cache_revalidation_seconds = 0;
  double normalization_seconds = 0;
  double mixer_seconds = 0;
  double feed_forward_seconds = 0;
};

// One complete layer, with an immutable input binding and fixed output storage.
// Context and input owner outlive the layer. Execution is synchronous.
class DecoderLayer {
public:
  DecoderLayer(Context &context, const WeightStore &weights, const Config &config, unsigned layer,
               TensorBinding input);
  ~DecoderLayer();
  DecoderLayer(const DecoderLayer &) = delete;
  DecoderLayer &operator=(const DecoderLayer &) = delete;

  void step();
  void reset();
  std::size_t length() const;
  TensorBinding output();
  std::vector<std::uint8_t> read(const std::string &name);
  DecoderMetrics metrics() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Runs the explicitly selected, ordered Qwen layer slice.
// Only the group boundary accepts host activations. Mid-step failure invalidates
// the entire group: individual recurrent layers cannot be rolled back by length.
class DecoderGroup {
public:
  DecoderGroup(Context &context, const WeightStore &weights, const Config &config,
               const std::vector<unsigned> &layers);
  ~DecoderGroup();
  DecoderGroup(const DecoderGroup &) = delete;
  DecoderGroup &operator=(const DecoderGroup &) = delete;

  void step(const std::vector<std::uint8_t> &hidden_fp16);
  void reset();
  std::size_t length() const;
  std::vector<std::uint8_t> read(unsigned layer, const std::string &name);
  std::vector<DecoderMetrics> metrics() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::qwen3_5
