#pragma once

#include "np101/context.hpp"
#include "vsi_nn_pub.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace specferry::np101 {
// Fixed layer storage: one FP16 K and V tensor, each [2, capacity, 256].
// The caller manages the valid prefix and executes the producer before write().
// Context and the two producer tensors must outlive this cache.
class KvCache {
public:
  static constexpr unsigned maximum_capacity = 512;
  static constexpr unsigned token_bytes = 2 * 2 * 256 * 2;

  KvCache(Context &context, Graph &producer, vsi_nn_tensor_id_t key, vsi_nn_tensor_id_t value,
          unsigned capacity = maximum_capacity);
  ~KvCache();
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;

  void write(unsigned position);
  vsi_nn_tensor_id_t retain_keys(Graph &reader);
  vsi_nn_tensor_id_t retain_values(Graph &reader);
  std::vector<std::uint8_t> read_keys();
  std::vector<std::uint8_t> read_values();
  unsigned capacity() const;
  std::size_t writes() const;
  std::size_t revalidations() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::np101
