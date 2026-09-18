#pragma once

#include "case_file.hpp"
#include "np101/context.hpp"
#include "vsi_nn_pub.h"

#include <map>
#include <string>

namespace specferry::testing {
vsi_nn_node_t *add_operator(np101::Graph &graph, const NodeDefinition &definition,
                            const std::map<std::string, vsi_nn_tensor_id_t> &tensors);
} // namespace specferry::testing
