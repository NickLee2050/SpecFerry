#include "np101/weight_bank.hpp"
#include "vsi_nn_dtype_util.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <ostream>
#include <stdexcept>

namespace specferry::np101 {
std::vector<std::uint8_t> prepare_weight_chunk(const WeightStore &store, const WeightRecord &record,
                                               const TensorSpec &target, std::size_t offset) {
  if (target.type != record.spec.type &&
      !(record.spec.type == DataType::Float16 && target.type == DataType::Float32)) {
    throw std::invalid_argument("weight preparation only supports identity or FP16 to FP32");
  }
  if (offset % dtype_bytes(record.spec.type)) {
    throw std::invalid_argument("weight chunk offset is not aligned to its source dtype");
  }
  auto source = store.read(record, offset, TensorSpec{record.spec.type, target.shape}.bytes());
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
      found = tensors_.emplace(key, Chunk{id, data.size()}).first;
      bytes_ += data.size();
    }
    return found->second.id;
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

void WeightBank::verify_preloaded(const std::vector<std::string> &names,
                                  const std::vector<Graph *> &consumers, std::ostream &report) {
  if (!storage_.get() || !source_ || names.empty() || consumers.empty()) {
    throw std::logic_error("weight preload check requires live weights and consumers");
  }
  std::map<vx_tensor, std::size_t> bindings;
  for (const auto *consumer : consumers) {
    auto *graph = consumer->get();
    for (unsigned index = 0; index < graph->input.num; ++index) {
      const auto *tensor = vsi_nn_GetTensor(graph, graph->input.tensors[index]);
      if (!tensor || !tensor->t) {
        throw std::runtime_error("missing consumer input during weight preload check");
      }
      ++bindings[tensor->t];
    }
  }
  report << "event\tname\tchunk_offset\tregion_offset\tchecked_bytes\tbindings\tmatch\n";
  std::size_t checked = 0;
  std::map<std::string, unsigned> selected;
  for (const auto &[key, chunk] : tensors_) {
    const auto &[name, offset, type, shape] = key;
    const auto *tensor = vsi_nn_GetTensor(storage_.get(), chunk.id);
    if (!tensor || !tensor->t || !bindings[tensor->t]) {
      throw std::runtime_error("shared weight has no live consumer input: " + name);
    }
    report << "binding\t" << name << '\t' << offset << "\t0\t0\t" << bindings[tensor->t] << "\t1\n";
    if (std::find(names.begin(), names.end(), name) == names.end()) {
      continue;
    }
    if (tensor->attr.dim_num != 2 || tensor->attr.dtype.vx_type != VSI_NN_TYPE_FLOAT16 ||
        tensor->attr.size[0] < 16 || !tensor->attr.size[1]) {
      throw std::invalid_argument("weight preload regions require rank2 FP16 width >= 16");
    }
    const auto width = tensor->attr.size[0], rows = tensor->attr.size[1];
    if (std::size_t(width) * rows * 2 != chunk.bytes) {
      throw std::runtime_error("weight preload tensor shape changed: " + name);
    }
    const std::vector<std::vector<std::uint32_t>> starts{
        {0, 0}, {width / 2 - 8, rows / 2}, {width - 16, rows - 1}};
    for (const auto &start : starts) {
      const auto relative = (std::size_t(start[1]) * width + start[0]) * 2;
      const auto expected = source_->read(source_->find(name), offset + relative, 32);
      const auto actual =
          read_tensor_region(storage_, chunk.id, start, {start[0] + 16, start[1] + 1});
      const bool matches = actual == expected;
      report << "region\t" << name << '\t' << offset << '\t' << relative << "\t32\t"
             << bindings[tensor->t] << '\t' << matches << '\n'
             << std::flush;
      if (!matches) {
        throw std::runtime_error("preloaded weight region differs: " + name +
                                 " offset=" + std::to_string(offset + relative));
      }
      checked += actual.size();
    }
    ++selected[name];
  }
  for (const auto &name : names) {
    if (!selected[name]) {
      throw std::runtime_error("requested preload weight was not loaded: " + name);
    }
  }
  report << "summary\t\t0\t0\t" << checked << '\t' << tensors_.size() << "\t1\n";
  if (!report) {
    throw std::runtime_error("cannot save weight preload report");
  }
}

void WeightBank::close() { storage_.close(); }
} // namespace specferry::np101
