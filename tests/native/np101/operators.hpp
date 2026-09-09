#pragma once

#include "case_file.hpp"
#include "np101/tensor.hpp"

#include <map>

namespace specferry::testing {
vsi_nn_node_t *add_operator(np101::Graph &graph, const NodeDefinition &definition,
                            const std::map<std::string, vsi_nn_tensor_id_t> &tensors);
} // namespace specferry::testing
