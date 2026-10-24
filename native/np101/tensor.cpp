#include "np101/diagnostics.hpp"
#include "np101/tensor.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
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

void transfer_mapped_tensor(vsi_nn_tensor_t *tensor, const std::vector<std::uint32_t> &region_start,
                            const std::vector<std::uint32_t> &region_end,
                            std::vector<std::uint8_t> *read_data,
                            const std::vector<std::uint8_t> *write_data) {
  const auto storage_bytes = tensor_bytes(tensor);
  const auto rank = tensor->attr.dim_num;
  if (!tensor->t || tensor->attr.vtl || tensor->attr.is_created_from_handle || rank > 3 ||
      tensor->attr.dtype.qnt_type != VSI_NN_QNT_TYPE_NONE ||
      (write_data && tensor->attr.is_const)) {
    throw std::invalid_argument("mapped transfer requires an ordinary unquantized rank-1/2/3 "
                                "tensor; writes require mutable storage");
  }
  if (region_start.size() != rank || region_end.size() != rank) {
    throw std::invalid_argument("mapped tensor region rank differs");
  }
  std::array<vx_size, 3> start{}, end{0, 1, 1}, length{1, 1, 1};
  std::size_t elements = 1;
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (region_start[axis] >= region_end[axis] || region_end[axis] > tensor->attr.size[axis]) {
      throw std::out_of_range("mapped tensor region exceeds storage");
    }
    start[axis] = region_start[axis];
    end[axis] = region_end[axis];
    length[axis] = end[axis] - start[axis];
    elements *= tensor->attr.size[axis];
  }
  const auto element_bytes = storage_bytes / elements;
  const auto row_bytes = length[0] * element_bytes;
  const auto bytes = row_bytes * length[1] * length[2];
  std::array<vx_size, 3> stride{};
  if (write_data) {
    if (write_data->size() != bytes) {
      throw std::invalid_argument("mapped write byte count differs from region");
    }
  } else {
    read_data->resize(bytes);
  }
  void *pointer = nullptr;
  vx_map_id map_id = 0;
  check(sdk_call("vxMapTensorPatch",
                 [&] {
                   return vxMapTensorPatch(
                       tensor->t, rank, start.data(), end.data(), &map_id, stride.data(), &pointer,
                       write_data ? VX_WRITE_ONLY : VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
                 }),
        "map tensor for host transfer");

  struct Mapping {
    vx_tensor tensor;
    vx_map_id id;
    bool active = true;

    ~Mapping() {
      if (active) {
        vxUnmapTensorPatch(tensor, id);
      }
    }

    void close() {
      active = false;
      check(sdk_call("vxUnmapTensorPatch", [&] { return vxUnmapTensorPatch(tensor, id); }),
            "unmap host tensor transfer");
    }
  } mapping{tensor->t, map_id};

  auto span = row_bytes;
  bool valid = pointer && stride[0] == element_bytes;
  for (unsigned axis = 1; valid && axis < rank; ++axis) {
    valid = stride[axis] >= span &&
            (length[axis] == 1 ||
             stride[axis] <= (std::numeric_limits<std::size_t>::max() - span) / (length[axis] - 1));
    if (valid) {
      span += (length[axis] - 1) * stride[axis];
    }
  }
  if (!valid) {
    throw std::runtime_error("mapped tensor pointer or strides violate the transfer contract");
  }
  // The pointer expires at unmap. Copy rows while honoring padding in every mapped axis.
  for (vx_size plane = 0; plane < length[2]; ++plane) {
    for (vx_size row = 0; row < length[1]; ++row) {
      auto *mapped = static_cast<std::uint8_t *>(pointer) + plane * stride[2] + row * stride[1];
      const auto packed = (plane * length[1] + row) * row_bytes;
      if (write_data) {
        std::memcpy(mapped, write_data->data() + packed, row_bytes);
      } else {
        std::memcpy(read_data->data() + packed, mapped, row_bytes);
      }
    }
  }
  mapping.close();
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

std::vector<std::uint8_t> read_tensor(Graph &graph, vsi_nn_tensor_id_t id, TensorReadback method) {
  auto *tensor = get_tensor(graph, id);
  auto bytes = tensor_bytes(tensor);
  if (method == TensorReadback::MappedHost) {
    std::vector<std::uint32_t> start(tensor->attr.dim_num, 0);
    const std::vector<std::uint32_t> end(tensor->attr.size,
                                         tensor->attr.size + tensor->attr.dim_num);
    return read_tensor_region(graph, id, start, end);
  }
  if (method != TensorReadback::SdkCopy) {
    throw std::invalid_argument("unknown tensor readback method");
  }
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

std::vector<std::uint8_t> read_tensor_region(Graph &graph, vsi_nn_tensor_id_t id,
                                             const std::vector<std::uint32_t> &start,
                                             const std::vector<std::uint32_t> &end) {
  std::vector<std::uint8_t> result;
  read_tensor_region_into(graph, id, start, end, result);
  return result;
}

void read_tensor_region_into(Graph &graph, vsi_nn_tensor_id_t id,
                             const std::vector<std::uint32_t> &start,
                             const std::vector<std::uint32_t> &end,
                             std::vector<std::uint8_t> &data) {
  transfer_mapped_tensor(get_tensor(graph, id), start, end, &data, nullptr);
  ++transfers.reads;
  transfers.read_bytes += data.size();
}

void upload_tensor_region(Graph &graph, vsi_nn_tensor_id_t id,
                          const std::vector<std::uint32_t> &start,
                          const std::vector<std::uint32_t> &end,
                          const std::vector<std::uint8_t> &data) {
  transfer_mapped_tensor(get_tensor(graph, id), start, end, nullptr, &data);
  ++transfers.uploads;
  transfers.upload_bytes += data.size();
}

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
