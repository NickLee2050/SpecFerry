#pragma once

#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"
#include "vsi_nn_pub.h"

#include <cstddef>
#include <map>
#include <string>
#include <tuple>

namespace specferry::np101 {
// One owner for immutable application weights shared by fixed execution graphs.
// SDK storage is mutable only to permit the validated retained-tensor API.
// All consumer graphs must be closed before this bank; no writes occur after loading.
class WeightBank {
public:
  explicit WeightBank(Context &context);
  vsi_nn_tensor_id_t bind(Graph &receiver, const WeightStore &store, const WeightRecord &record,
                          const TensorSpec &spec, std::size_t offset = 0);
  std::size_t payload_bytes() const;
  // Opt-in diagnostics only: all retained bytes, never a sampled readback.
  void verify_if_requested(const char *phase);
  void close();

private:
  struct Chunk {
    vsi_nn_tensor_id_t id;
    std::size_t bytes;
  };

  Graph storage_;
  const WeightStore *source_ = nullptr;
  std::map<std::tuple<std::string, std::size_t, std::string>, Chunk> tensors_;
  std::size_t bytes_ = 0;
};
} // namespace specferry::np101
