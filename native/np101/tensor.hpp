#pragma once

#include "np101/context.hpp"
#include "np101/tensor_spec.hpp"

#include <vector>

namespace specferry::np101 {
// Graph owns the SDK tensor. These helpers never release an individual tensor.
vsi_nn_tensor_id_t add_tensor(Graph &graph, const TensorSpec &spec, bool constant = false,
                              const std::vector<std::uint8_t> &initial = {},
                              bool from_handle = false);

void upload_tensor(Graph &graph, vsi_nn_tensor_id_t id, const std::vector<std::uint8_t> &data);

std::vector<std::uint8_t> read_tensor(Graph &graph, vsi_nn_tensor_id_t id);

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
