#include "np101/tensor.hpp"

#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>

namespace specferry::np101 {
namespace {
vsi_nn_tensor_t *get_tensor(Graph &graph, vsi_nn_tensor_id_t id) {
  auto *tensor = vsi_nn_GetTensor(graph.get(), id);
  if (!tensor) {
    throw std::runtime_error("missing tensor id=" + std::to_string(id));
  }
  return tensor;
}

std::size_t tensor_bytes(vsi_nn_tensor_t *tensor) {
  DataType type;
  switch (tensor->attr.dtype.vx_type) {
  case VSI_NN_TYPE_FLOAT16:
    type = DataType::Float16;
    break;
  case VSI_NN_TYPE_FLOAT32:
    type = DataType::Float32;
    break;
  case VSI_NN_TYPE_INT32:
    type = DataType::Int32;
    break;
  case VSI_NN_TYPE_BOOL8:
    type = DataType::Bool8;
    break;
  default:
    throw std::runtime_error("unexpected SDK tensor dtype");
  }
  if (tensor->attr.dim_num == 0 || tensor->attr.dim_num > VSI_NN_MAX_DIM_NUM) {
    throw std::runtime_error("unresolved SDK tensor shape");
  }
  std::vector<std::uint32_t> shape(tensor->attr.size, tensor->attr.size + tensor->attr.dim_num);
  return TensorSpec{type, shape}.bytes();
}
} // namespace

vsi_nn_tensor_id_t add_tensor(Graph &graph, const TensorSpec &spec, bool constant,
                              const std::vector<std::uint8_t> &initial, bool from_handle) {
  auto bytes = spec.bytes();
  if ((!initial.empty() && initial.size() != bytes) || (constant && initial.empty())) {
    throw std::invalid_argument("tensor initialization size mismatch");
  }
  vsi_nn_tensor_attr_t attr{};
  attr.dim_num = spec.shape.size();
  std::copy(spec.shape.begin(), spec.shape.end(), attr.size);
  attr.is_const = constant;
  attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
  attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
  switch (spec.type) {
  case DataType::Float16:
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    break;
  case DataType::Float32:
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    break;
  case DataType::Int32:
    attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    break;
  case DataType::Bool8:
    attr.dtype.vx_type = VSI_NN_TYPE_BOOL8;
    break;
  }

  // AddTensor copies initialization bytes. FromHandle owns its aligned allocation;
  // never hand it a vector's storage or infer device residency from this API.
  auto id = from_handle
                ? vsi_nn_AddTensorFromHandle(graph.get(), VSI_NN_TENSOR_ID_AUTO, &attr, nullptr)
                : vsi_nn_AddTensor(graph.get(), VSI_NN_TENSOR_ID_AUTO, &attr,
                                   initial.empty() ? nullptr
                                                   : const_cast<std::uint8_t *>(initial.data()));
  if (id == VSI_NN_TENSOR_ID_NA || !vsi_nn_GetTensor(graph.get(), id)) {
    throw std::runtime_error("AddTensor failed: dtype=" + dtype_name(spec.type) +
                             " bytes=" + std::to_string(bytes));
  }
  if (from_handle && !initial.empty()) {
    upload_tensor(graph, id, initial);
  }
  return id;
}

void upload_tensor(Graph &graph, vsi_nn_tensor_id_t id, const std::vector<std::uint8_t> &data) {
  auto *tensor = get_tensor(graph, id);
  if (data.size() != tensor_bytes(tensor)) {
    throw std::invalid_argument("upload byte count does not match tensor");
  }
  check(vsi_nn_CopyDataToTensor(graph.get(), tensor, const_cast<std::uint8_t *>(data.data())),
        "CopyDataToTensor");
}

std::vector<std::uint8_t> read_tensor(Graph &graph, vsi_nn_tensor_id_t id) {
  auto *tensor = get_tensor(graph, id);
  auto bytes = tensor_bytes(tensor);
  std::unique_ptr<std::uint8_t, decltype(&std::free)> data(
      vsi_nn_ConvertTensorToData(graph.get(), tensor), std::free);
  if (!data) {
    throw std::runtime_error("ConvertTensorToData returned null");
  }
  return {data.get(), data.get() + bytes};
}

TensorAttachment::TensorAttachment(Graph &owner, vsi_nn_tensor_id_t tensor, Graph &receiver)
    : receiver_(receiver), id_(VSI_NN_TENSOR_ID_NA) {
  if (owner.get() == receiver.get() || owner.get()->ctx != receiver.get()->ctx) {
    throw std::invalid_argument("tensor attachment requires separate graphs in one context");
  }
  // Some SDK distributions declare this function but hide it from the dynamic
  // symbol table. Probe availability without making every tensor user unlinkable.
  using Attach = vsi_nn_tensor_id_t (*)(vsi_nn_graph_t *, vsi_nn_tensor_id_t, vsi_nn_tensor_t *);
  auto attach = reinterpret_cast<Attach>(dlsym(RTLD_DEFAULT, "vsi_nn_AttachTensorToGraph"));
  if (!attach) {
    throw std::runtime_error("SDK does not export vsi_nn_AttachTensorToGraph");
  }
  id_ = attach(receiver.get(), VSI_NN_TENSOR_ID_AUTO, get_tensor(owner, tensor));
  if (id_ == VSI_NN_TENSOR_ID_NA) {
    throw std::runtime_error("AttachTensorToGraph failed");
  }
}

TensorAttachment::~TensorAttachment() {
  // AttachTensorToGraph inserts the existing wrapper into the receiver's map
  // without retaining it. Remove only that map entry: RemoveTensor would free the
  // owner's wrapper and cause a second release when the owner graph is destroyed.
  if (receiver_.get()) {
    vsi_nn_MapRemove(receiver_.get()->tensor_table, id_);
  }
}
} // namespace specferry::np101
