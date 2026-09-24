#pragma once

#include "inference/generation.hpp"
#include "models/opt/config.hpp"
#include "np101/context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace specferry::models::opt {
// Experimental: compilation and host contracts are separate from SDK validation.
// Exactly one graph for decode, and optionally one fixed block-prefill graph.
// Context and WeightStore outlive the model. Both graphs share weight/cache owners.
class GraphModel {
public:
  GraphModel(np101::Context &context, const np101::WeightStore &weights, const ModelConfig &config,
             unsigned prefill_block);
  ~GraphModel();
  GraphModel(const GraphModel &) = delete;
  GraphModel &operator=(const GraphModel &) = delete;

  inference::Generation generate(const std::vector<std::int32_t> &prompt, unsigned maximum);
  std::vector<std::uint8_t> read_cache(unsigned layer, bool values);
  std::size_t launches() const;
  std::size_t weight_bytes() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::opt
