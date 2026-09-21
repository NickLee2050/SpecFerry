#pragma once

#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace specferry::np101 {
// Qwen3.5-0.8B layer-3 attention mixer, excluding decoder input norm/residual.
// A single sequence owns one fixed-capacity KV allocation. Context outlives it.
// Calls are synchronous; reset/truncate change the valid prefix without clearing
// storage. A failed SDK step invalidates the instance, which must be recreated.
class Attention {
public:
  Attention(Context &context, const WeightStore &weights);
  Attention(Context &context, const WeightStore &weights, TensorBinding input,
            TensorBinding output);
  ~Attention();
  Attention(const Attention &) = delete;
  Attention &operator=(const Attention &) = delete;

  void step(const std::vector<std::uint8_t> &hidden_fp16);
  void step();
  void reset();
  void truncate(unsigned length);
  unsigned length() const;
  std::vector<std::uint8_t> read(const std::string &name);
  std::size_t cache_writes() const;
  std::size_t cache_revalidations() const;
  double cache_write_seconds() const;
  double cache_revalidation_seconds() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::np101
