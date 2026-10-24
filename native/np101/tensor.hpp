#pragma once

#include "np101/context.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace specferry::np101 {
// A borrowed graph/tensor pair. The owner must outlive every bound consumer.
struct TensorBinding {
  Graph &owner;
  vsi_nn_tensor_id_t id;
};

// A fixed copy contract. Both owners must outlive the graph retaining these bindings.
struct TensorCopyBinding {
  TensorBinding source, destination;
  TensorSpec spec;
};

// Successful explicit helper transfers on this thread, excluding SDK-internal IO
// and AddTensor initialization. Diagnostics use deltas around a complete step.
struct TensorTransfers {
  std::size_t uploads = 0;
  std::size_t upload_bytes = 0;
  std::size_t reads = 0;
  std::size_t read_bytes = 0;
};

TensorTransfers tensor_transfers();

// Validate the caller's shape and dtype before retaining its storage.
vsi_nn_tensor_id_t bind_tensor(TensorBinding binding, Graph &receiver, const TensorSpec &expected);

// Graph owns the SDK tensor. These helpers never release an individual tensor.
vsi_nn_tensor_id_t add_tensor(Graph &graph, const TensorSpec &spec, bool constant = false,
                              const std::vector<std::uint8_t> &initial = {});

void upload_tensor(Graph &graph, vsi_nn_tensor_id_t id, const std::vector<std::uint8_t> &data);

enum class TensorReadback { SdkCopy, MappedHost };

// MappedHost is an explicit diagnostic path for ordinary, unquantized rank-1/2/3 tensors.
// Call with no graph executing, including after initialization or synchronous completion.
// There is no fallback to SDK DMA copy on failure.
std::vector<std::uint8_t> read_tensor(Graph &graph, vsi_nn_tensor_id_t id,
                                      TensorReadback method = TensorReadback::SdkCopy);

// Explicit mapped diagnostic; half-open coordinates, ordinary unquantized rank-1/2/3 only.
std::vector<std::uint8_t> read_tensor_region(Graph &graph, vsi_nn_tensor_id_t id,
                                             const std::vector<std::uint32_t> &start,
                                             const std::vector<std::uint32_t> &end);

// Reuse the caller's packed host buffer; resize only when the region size changes.
void read_tensor_region_into(Graph &graph, vsi_nn_tensor_id_t id,
                             const std::vector<std::uint32_t> &start,
                             const std::vector<std::uint32_t> &end,
                             std::vector<std::uint8_t> &data);

// Explicit mapped write with the same region/layout restrictions as mapped readback.
// Write every element inside the half-open patch, preserving storage outside it.
// Call only with no graph executing; unmap completes before the helper returns.
void upload_tensor_region(Graph &graph, vsi_nn_tensor_id_t id,
                          const std::vector<std::uint32_t> &start,
                          const std::vector<std::uint32_t> &end,
                          const std::vector<std::uint8_t> &data);

// Retain one materialized ordinary tensor in a second SDK wrapper before setup.
// Each graph owns its own OpenVX reference. This does not copy tensor contents.
vsi_nn_tensor_id_t retain_tensor(Graph &owner, vsi_nn_tensor_id_t id, Graph &receiver);

} // namespace specferry::np101
