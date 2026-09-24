#include "np101/diagnostics.hpp"
#include "np101/tensor.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace specferry::np101 {
namespace {
thread_local TensorTransfers transfers;

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
                              const std::vector<std::uint8_t> &initial) {
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

  // AddTensor copies initialization bytes; the graph owns the resulting tensor.
  auto allocation = graph.reserve(constant, bytes);
  auto id = sdk_call("vsi_nn_AddTensor", [&] {
    return vsi_nn_AddTensor(graph.get(), VSI_NN_TENSOR_ID_AUTO, &attr,
                            initial.empty() ? nullptr : const_cast<std::uint8_t *>(initial.data()));
  });
  if (id == VSI_NN_TENSOR_ID_NA || !vsi_nn_GetTensor(graph.get(), id)) {
    throw std::runtime_error("AddTensor failed: dtype=" + dtype_name(spec.type) +
                             " bytes=" + std::to_string(bytes));
  }
  graph.record_storage(id, std::move(allocation));
  return id;
}

void upload_tensor(Graph &graph, vsi_nn_tensor_id_t id, const std::vector<std::uint8_t> &data) {
  auto *tensor = get_tensor(graph, id);
  if (data.size() != tensor_bytes(tensor)) {
    throw std::invalid_argument("upload byte count does not match tensor");
  }
  check(sdk_call("vsi_nn_CopyDataToTensor",
                 [&] {
                   return vsi_nn_CopyDataToTensor(graph.get(), tensor,
                                                  const_cast<std::uint8_t *>(data.data()));
                 }),
        "CopyDataToTensor");
  ++transfers.uploads;
  transfers.upload_bytes += data.size();
}

std::vector<std::uint8_t> read_tensor(Graph &graph, vsi_nn_tensor_id_t id) {
  auto *tensor = get_tensor(graph, id);
  auto bytes = tensor_bytes(tensor);
  std::unique_ptr<std::uint8_t, decltype(&std::free)> data(
      sdk_call("vsi_nn_ConvertTensorToData",
               [&] { return vsi_nn_ConvertTensorToData(graph.get(), tensor); }),
      std::free);
  if (!data) {
    throw std::runtime_error("ConvertTensorToData returned null");
  }
  ++transfers.reads;
  transfers.read_bytes += bytes;
  return {data.get(), data.get() + bytes};
}

TensorTransfers tensor_transfers() { return transfers; }

vsi_nn_tensor_id_t retain_tensor(Graph &owner, vsi_nn_tensor_id_t id, Graph &receiver) {
  auto *source = get_tensor(owner, id);
  if (owner.get()->ctx != receiver.get()->ctx || !source->t || source->attr.vtl ||
      source->attr.is_created_from_handle || source->attr.is_const) {
    throw std::invalid_argument("sharing requires an ordinary mutable tensor in one context");
  }
  // The SDK's AttachTensorToGraph symbol is unavailable. As in DeltaNet, use
  // public tensor wrappers and explicit OpenVX reference ownership instead.
  auto attr = source->attr;
  // Count the wrapper's temporary backing until it is replaced by the alias.
  auto temporary = receiver.reserve(false, tensor_bytes(source));
  auto shared = sdk_call("vsi_nn_AddTensor", [&] {
    return vsi_nn_AddTensor(receiver.get(), VSI_NN_TENSOR_ID_AUTO, &attr, nullptr);
  });
  if (shared == VSI_NN_TENSOR_ID_NA) {
    throw std::runtime_error("AddTensor failed for retained tensor");
  }
  auto *destination = get_tensor(receiver, shared);
  receiver.record_storage(shared, std::move(temporary));
  check(vxRetainReference(reinterpret_cast<vx_reference>(source->t)), "retain tensor");
  if (destination->t) {
    auto status = vxReleaseTensor(&destination->t);
    if (status != VX_SUCCESS) {
      auto retained = source->t;
      vxReleaseTensor(&retained);
      check(status, "release unused retained-tensor allocation");
    }
  }
  destination->t = source->t;
  receiver.alias_storage(shared, owner, id);
  return shared;
}

vsi_nn_tensor_id_t bind_tensor(TensorBinding binding, Graph &receiver, const TensorSpec &expected) {
  auto *tensor = get_tensor(binding.owner, binding.id);
  const auto &attr = tensor->attr;
  const auto type = expected.type == DataType::Float16   ? VSI_NN_TYPE_FLOAT16
                    : expected.type == DataType::Float32 ? VSI_NN_TYPE_FLOAT32
                    : expected.type == DataType::Int32   ? VSI_NN_TYPE_INT32
                                                         : VSI_NN_TYPE_BOOL8;
  if (attr.dim_num != expected.shape.size() ||
      !std::equal(expected.shape.begin(), expected.shape.end(), attr.size) ||
      attr.dtype.vx_type != type || attr.dtype.qnt_type != VSI_NN_QNT_TYPE_NONE ||
      expected.bytes() != tensor_bytes(tensor)) {
    throw std::invalid_argument("bound tensor does not match the requested shape/dtype");
  }
  return retain_tensor(binding.owner, binding.id, receiver);
}

} // namespace specferry::np101
