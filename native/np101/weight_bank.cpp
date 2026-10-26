#include "np101/diagnostics.hpp"
#include "np101/weight_bank.hpp"
#include "vsi_nn_dtype_util.h"

#include <cstring>
#include <exception>
#include <stdexcept>

namespace specferry::np101 {
std::vector<std::uint8_t> prepare_weight_chunk(const WeightStore &store, const WeightRecord &record,
                                               const TensorSpec &target, std::size_t offset) {
  const SdkTimer preparation("weight_prepare");
  if (target.type != record.spec.type &&
      !(record.spec.type == DataType::Float16 && target.type == DataType::Float32)) {
    throw std::invalid_argument("weight preparation only supports identity or FP16 to FP32");
  }
  if (offset % dtype_bytes(record.spec.type)) {
    throw std::invalid_argument("weight chunk offset is not aligned to its source dtype");
  }
  auto source = sdk_call("weight_read", [&] {
    return store.read(record, offset, TensorSpec{record.spec.type, target.shape}.bytes());
  });
  if (target.type == record.spec.type) {
    return source;
  }
  // Use the SDK's host dtype utility; this does not create tensors or run a graph.
  vsi_nn_dtype_t dtype{};
  dtype.vx_type = VSI_NN_TYPE_FLOAT16;
  dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
  std::vector<std::uint8_t> expanded(target.bytes());
  const auto count = target.elements();
  for (std::size_t index = 0; index < count; ++index) {
    float value;
    check(vsi_nn_DtypeToFloat32(source.data() + index * 2, &value, &dtype), "prepare FP32 weight");
    std::memcpy(expanded.data() + index * sizeof(value), &value, sizeof(value));
  }
  return expanded;
}

WeightBank::WeightBank(Context &context) : storage_(context, 1024, 0) {}

vsi_nn_tensor_id_t WeightBank::ensure_loaded(const WeightStore &store, const WeightRecord &record,
                                             const TensorSpec &spec, std::size_t offset) {
  try {
    if (!storage_.get()) {
      throw std::logic_error("shared weight bank is closed");
    }
    if (source_ && source_ != &store) {
      throw std::invalid_argument("shared weight bank source differs");
    }
    source_ = &store;
    std::string shape;
    for (auto dimension : spec.shape) {
      shape += std::to_string(dimension) + ",";
    }
    const auto key = std::make_tuple(record.name, offset, spec.type, shape);
    auto found = tensors_.find(key);
    if (found == tensors_.end()) {
      const auto data = prepare_weight_chunk(store, record, spec, offset);
      const auto id = add_tensor(storage_, spec, false, data);
      found = tensors_.emplace(key, id).first;
      bytes_ += data.size();
    }
    return found->second;
  } catch (const std::exception &error) {
    throw std::runtime_error("shared weight load failed: " + record.name +
                             " offset=" + std::to_string(offset) + ": " + error.what());
  }
}

void WeightBank::preload(const WeightStore &store, const WeightRecord &record,
                         const TensorSpec &spec, std::size_t offset) {
  ensure_loaded(store, record, spec, offset);
}

vsi_nn_tensor_id_t WeightBank::bind(Graph &receiver, const WeightStore &store,
                                    const WeightRecord &record, const TensorSpec &spec,
                                    std::size_t offset) {
  const auto id = ensure_loaded(store, record, spec, offset);
  try {
    return retain_tensor(storage_, id, receiver);
  } catch (const std::exception &error) {
    throw std::runtime_error("shared weight bind failed: " + record.name +
                             " offset=" + std::to_string(offset) + ": " + error.what());
  }
}

std::size_t WeightBank::payload_bytes() const { return bytes_; }

void WeightBank::close() { storage_.close(); }
} // namespace specferry::np101
