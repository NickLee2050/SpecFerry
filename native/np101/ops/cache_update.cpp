#include "np101/component_spec.hpp"
#include "np101/ops/cache_update.hpp"
#include "vsi_nn_pub.h"

#include <stdexcept>
#include <vector>

namespace specferry::np101::ops {
Tensor indexed_append(GraphBuilder &g, Tensor column, Tensor index, Tensor storage) {
  validate_cache_append(column.spec, index.spec, storage.spec);
  g.node(VSI_NN_OP_TENSORSTACKCONCAT, {column, index}, storage)->nn_param.tensorstackconcat.axis =
      1;
  return storage;
}

Tensor storage_reshape(GraphBuilder &g, Tensor source, Shape shape) {
  TensorSpec spec{source.spec.type, shape};
  auto *input = vsi_nn_GetTensor(g.graph.get(), source.id);
  if (spec.elements() != source.spec.elements() || !input || !input->t || input->attr.vtl ||
      input->attr.is_const || input->attr.is_created_from_handle) {
    throw std::invalid_argument("storage reshape requires an equal-size ordinary mutable tensor");
  }
  auto output = g.tensor(spec);
  auto *wrapper = vsi_nn_GetTensor(g.graph.get(), output.id);
  if (!wrapper) {
    throw std::runtime_error("missing cache alias wrapper");
  }
#if VX_VA40_EXT_SUPPORT
  std::vector<vx_size> dimensions(shape.begin(), shape.end());
#else
  std::vector<vx_int32> dimensions(shape.begin(), shape.end());
#endif
  auto alias = vxReshapeTensor(input->t, dimensions.data(), dimensions.size());
  if (!alias) {
    throw std::runtime_error("cache alias creation returned null");
  }
  const auto status = vxGetStatus(reinterpret_cast<vx_reference>(alias));
  if (status != VX_SUCCESS) {
    vxReleaseTensor(&alias);
    check(status, "reshape cache storage");
  }
  if (wrapper->t) {
    const auto released = vxReleaseTensor(&wrapper->t);
    if (released != VX_SUCCESS) {
      vxReleaseTensor(&alias);
      check(released, "release unused alias backing");
    }
  }
  wrapper->t = alias;
  g.graph.alias_storage(output.id, g.graph, source.id);
  return output;
}
} // namespace specferry::np101::ops
