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

// Successful explicit helper transfers on this thread, excluding SDK-internal IO
// and AddTensor initialization. Diagnostics use deltas around a complete step.
struct TensorTransfers {
  std::size_t uploads = 0;
  std::size_t upload_bytes = 0;
  std::size_t reads = 0;
  std::size_t read_bytes = 0;
};

TensorTransfers tensor_transfers();

// Validate the fixed hidden-vector contract before retaining its storage.
vsi_nn_tensor_id_t bind_hidden(TensorBinding binding, Graph &receiver);

// Graph owns the SDK tensor. These helpers never release an individual tensor.
vsi_nn_tensor_id_t add_tensor(Graph &graph, const TensorSpec &spec, bool constant = false,
                              const std::vector<std::uint8_t> &initial = {});

void upload_tensor(Graph &graph, vsi_nn_tensor_id_t id, const std::vector<std::uint8_t> &data);

std::vector<std::uint8_t> read_tensor(Graph &graph, vsi_nn_tensor_id_t id);

// Retain one materialized ordinary tensor in a second SDK wrapper before setup.
// Each graph owns its own OpenVX reference. This does not copy tensor contents.
vsi_nn_tensor_id_t retain_tensor(Graph &owner, vsi_nn_tensor_id_t id, Graph &receiver);

// Attach a graph-owned tensor without taking ownership of its C wrapper. This
// object must die before either graph. Attachment alone does not prove zero-copy.
class TensorAttachment {
public:
  TensorAttachment(Graph &owner, vsi_nn_tensor_id_t tensor, Graph &receiver);

  ~TensorAttachment();

  TensorAttachment(const TensorAttachment &) = delete;

  TensorAttachment &operator=(const TensorAttachment &) = delete;

  vsi_nn_tensor_id_t id() const { return id_; }

private:
  Graph &receiver_;
  vsi_nn_tensor_id_t id_;
};
} // namespace specferry::np101
