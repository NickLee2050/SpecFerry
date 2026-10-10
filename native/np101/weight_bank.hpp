#pragma once

#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace specferry::np101 {
// Read a bounded source chunk, optionally widening FP16 to FP32 on the host.
// Shape only changes the view; offset is always measured in source bytes.
std::vector<std::uint8_t> prepare_weight_chunk(const WeightStore &store, const WeightRecord &record,
                                               const TensorSpec &target, std::size_t offset = 0);

// One owner for immutable application weights shared by fixed execution graphs.
// SDK storage is mutable only to permit the validated retained-tensor API.
// All consumer graphs must be closed before this bank; no writes occur after loading.
class WeightBank {
public:
  explicit WeightBank(Context &context);
  // Load a chunk into the shared bank before constructing consumer graphs.
  // A later bind with the same record, offset, target dtype and shape reuses this tensor.
  void preload(const WeightStore &store, const WeightRecord &record, const TensorSpec &spec,
               std::size_t offset = 0);
  vsi_nn_tensor_id_t bind(Graph &receiver, const WeightStore &store, const WeightRecord &record,
                          const TensorSpec &spec, std::size_t offset = 0);
  std::size_t payload_bytes() const;
  // Opt-in before execution: check all consumer input aliases and selected FP16 regions.
  // Consumer graphs must remain alive. This does not establish physical memory residency.
  void verify_preloaded(const std::vector<std::string> &names,
                        const std::vector<Graph *> &consumers, std::ostream &report);
  void close();

private:
  struct Chunk {
    vsi_nn_tensor_id_t id;
    std::size_t bytes;
  };

  vsi_nn_tensor_id_t ensure_loaded(const WeightStore &store, const WeightRecord &record,
                                   const TensorSpec &spec, std::size_t offset);
  Graph storage_;
  const WeightStore *source_ = nullptr;
  std::map<std::tuple<std::string, std::size_t, DataType, std::string>, Chunk> tensors_;
  std::size_t bytes_ = 0;
};
} // namespace specferry::np101
