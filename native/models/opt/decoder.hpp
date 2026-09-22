#pragma once

#include "models/opt/config.hpp"
#include "np101/context.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace specferry::models::opt {
// A post-norm/ReLU OPT slice. Context outlives this object; calls are synchronous.
// Each layer keeps one K/V allocation and shares its output with the next layer.
// Partial execution failure invalidates the slice; reset is not a rollback.
class DecoderSlice {
public:
  DecoderSlice(np101::Context &context, const np101::WeightStore &weights, const Config &config,
               const std::vector<unsigned> &layers);
  ~DecoderSlice();
  DecoderSlice(const DecoderSlice &) = delete;
  DecoderSlice &operator=(const DecoderSlice &) = delete;

  void step(const std::vector<std::uint8_t> &hidden);
  void reset();
  unsigned length() const;
  std::size_t cache_writes() const;
  std::size_t cache_revalidations() const;
  std::vector<std::uint8_t> read(unsigned layer, const std::string &name);
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::models::opt
