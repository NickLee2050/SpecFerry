#pragma once

#include "devices/np101/api/component_spec.hpp"
#include "devices/np101/api/context.hpp"
#include "devices/np101/api/tensor.hpp"
#include "vsi_nn_pub.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace specferry::np101 {
// MappedHost requires a completed synchronous producer and ordinary mappable tensors.
enum class KvWriteMode { CopyGraph, MappedHost };

// Fixed layer storage: one FP16 K and V parent tensor using the supplied KvSpec.
// The caller manages the valid prefix and executes the producer before write().
// Context, producer graph and both producer tensors must outlive this cache.
class KvCache {
public:
  KvCache(Context &context, Graph &producer, vsi_nn_tensor_id_t key, vsi_nn_tensor_id_t value,
          KvSpec spec, KvWriteMode mode = KvWriteMode::CopyGraph);
  ~KvCache();
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;

  void write(unsigned position);
  vsi_nn_tensor_id_t retain_keys(Graph &reader);
  vsi_nn_tensor_id_t retain_values(Graph &reader);
  std::vector<std::uint8_t> read_keys(TensorReadback method = TensorReadback::SdkCopy);
  std::vector<std::uint8_t> read_values(TensorReadback method = TensorReadback::SdkCopy);
  unsigned capacity() const;
  KvSpec spec() const;
  std::size_t writes() const;
  std::size_t revalidations() const;
  void close();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace specferry::np101
