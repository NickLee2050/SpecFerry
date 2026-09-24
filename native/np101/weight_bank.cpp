#include "np101/weight_bank.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
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

void WeightBank::verify_if_requested(const char *phase) {
  const auto *path = std::getenv("SPECFERRY_WEIGHT_READBACK_REPORT");
  if (!path || !*path || tensors_.empty()) {
    return;
  }
  std::ofstream report(path, std::ios::app);
  std::cout << "Checking all shared weights " << phase << ": " << bytes_ << " bytes" << std::endl;
  std::size_t mismatches = 0;
  for (const auto &[key, chunk] : tensors_) {
    const auto &[name, offset, shape] = key;
    const auto expected = source_->read(source_->find(name), offset, chunk.bytes);
    const auto actual = read_tensor(storage_, chunk.id);
    if (actual.size() != expected.size()) {
      throw std::runtime_error("weight readback size differs: " + name);
    }
    std::size_t different = 0, first = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (actual[index] != expected[index]) {
        if (!different) {
          first = index;
        }
        ++different;
      }
    }
    // Offsets refer to checkpoint tensor bytes, never physical addresses.
    report << phase << '\t' << name << '\t' << offset << '\t' << chunk.bytes << '\t' << different
           << '\t';
    if (different) {
      report << offset + first;
    } else {
      report << '-';
    }
    report << std::endl;
    if (!report) {
      throw std::runtime_error("cannot save shared weight readback");
    }
    mismatches += different;
  }
  if (mismatches) {
    throw std::runtime_error(std::string("shared weight corruption ") + phase + ": " +
                             std::to_string(mismatches) + " bytes; see weight-readback.tsv");
  }
}

void WeightBank::close() { storage_.close(); }
} // namespace specferry::np101
