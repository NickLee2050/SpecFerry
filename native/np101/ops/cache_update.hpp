#pragma once

#include "np101/ops/graph_builder.hpp"

namespace specferry::np101::ops {
// Experimental fixed-graph append. Restrict the writer to a single 2-D column:
// this SDK crashes while reshaping a rank-three multi-head TENSORSTACKCONCAT.
// Runtime bounds belong to the caller; no upload, execution or reverify occurs here.
Tensor indexed_append(GraphBuilder &graph, Tensor column, Tensor index, Tensor storage);

// Reshape an existing mutable allocation without a copy node. Used only for
// cache block geometry; the owning allocation must outlive the execution graph.
Tensor storage_reshape(GraphBuilder &graph, Tensor source, Shape shape);
} // namespace specferry::np101::ops
