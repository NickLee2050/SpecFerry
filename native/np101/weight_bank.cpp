#include "np101/weight_bank.hpp"

#include <stdexcept>

namespace specferry::np101 {
WeightBank::WeightBank(Context &context) : storage_(context, 1024, 0) {}

vsi_nn_tensor_id_t WeightBank::bind(Graph &receiver, const WeightStore &store,
                                    const WeightRecord &record, const TensorSpec &spec,
                                    std::size_t offset) {
  if (!storage_.get()) {
    throw std::logic_error("shared weight bank is closed");
  }
  if ((source_ && source_ != &store) || spec.type != record.spec.type) {
    throw std::invalid_argument("shared weight bank source/dtype differs");
  }
  source_ = &store;
  std::string shape;
  for (auto dimension : spec.shape) {
    shape += std::to_string(dimension) + ",";
  }
  const auto key = std::make_tuple(record.name, offset, shape);
  auto found = tensors_.find(key);
  if (found == tensors_.end()) {
    const auto data = store.read(record, offset, spec.bytes());
    const auto id = add_tensor(storage_, spec, false, data);
    found = tensors_.emplace(key, Chunk{id, data.size()}).first;
    bytes_ += data.size();
  }
  return retain_tensor(storage_, found->second.id, receiver);
}

std::size_t WeightBank::payload_bytes() const { return bytes_; }

void WeightBank::verify() {
  for (const auto &[key, chunk] : tensors_) {
    const auto &[name, offset, shape] = key;
    const auto expected = source_->read(source_->find(name), offset, chunk.bytes);
    if (read_tensor(storage_, chunk.id) != expected) {
      throw std::runtime_error("shared weight readback mismatch: " + name + " at chunk offset " +
                               std::to_string(offset));
    }
  }
}

void WeightBank::close() { storage_.close(); }
} // namespace specferry::np101
