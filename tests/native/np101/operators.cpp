#include "operators.hpp"

#include <algorithm>
#include <map>

namespace specferry::testing {
namespace {
struct Operation {
  vsi_nn_op_t code;
  std::size_t inputs;
  std::size_t parameters;
};

const std::map<std::string, Operation> operations{
    {"ADD", {VSI_NN_OP_ADD, 2, 0}},
    {"SUBTRACT", {VSI_NN_OP_SUBTRACT, 2, 0}},
    {"MULTIPLY", {VSI_NN_OP_MULTIPLY, 2, 0}},
    {"MATRIXMUL", {VSI_NN_OP_MATRIXMUL, 2, 2}},
    {"FCL", {VSI_NN_OP_FCL, 2, 1}},
    {"GATHER", {VSI_NN_OP_GATHER, 2, 1}},
    {"REDUCE_SUM", {VSI_NN_OP_REDUCE, 1, 1}},
    {"REDUCE_MEAN", {VSI_NN_OP_REDUCE, 1, 1}},
    {"REDUCE_MAX", {VSI_NN_OP_REDUCE, 1, 1}},
    {"RSQRT", {VSI_NN_OP_RSQRT, 1, 0}},
    {"SIGMOID", {VSI_NN_OP_SIGMOID, 1, 0}},
    {"SWISH", {VSI_NN_OP_SWISH, 1, 0}},
    {"EXP", {VSI_NN_OP_EXP, 1, 0}},
    {"SOFTRELU", {VSI_NN_OP_SOFTRELU, 1, 0}},
    {"SOFTMAX", {VSI_NN_OP_SOFTMAX, 1, 1}},
    {"SCATTER_ND_UPDATE", {VSI_NN_OP_SCATTER_ND_UPDATE, 3, 0}},
    {"ARGMAX", {VSI_NN_OP_ARGMAX, 1, 1}},
    {"DATACONVERT", {VSI_NN_OP_DATACONVERT, 1, 0}},
    {"LESS", {VSI_NN_OP_RELATIONAL_OPS, 2, 0}},
    {"SELECT", {VSI_NN_OP_SELECT, 3, 0}},
    {"CONCAT", {VSI_NN_OP_CONCAT, 2, 1}},
    {"SLICE", {VSI_NN_OP_SLICE, 1, 0}},
    {"PERMUTE", {VSI_NN_OP_PERMUTE, 1, 0}},
    {"RESHAPE", {VSI_NN_OP_RESHAPE, 1, 0}},
};
} // namespace

vsi_nn_node_t *add_operator(np101::Graph &graph, const NodeDefinition &definition,
                            const std::map<std::string, vsi_nn_tensor_id_t> &tensors) {
  const auto &name = definition.operation;
  auto operation = operations.find(name);
  if (operation == operations.end()) {
    throw std::invalid_argument("unsupported fixture operation: " + name);
  }
  const auto &parameters = definition.parameters;
  bool variable_parameters = name == "SLICE" || name == "RESHAPE" || name == "PERMUTE";
  if (definition.inputs.size() != operation->second.inputs ||
      (!variable_parameters && parameters.size() != operation->second.parameters)) {
    throw std::invalid_argument("incorrect input or parameter count: " + name);
  }
  auto *node = vsi_nn_AddNode(graph.get(), operation->second.code, 0, 0, nullptr);
  if (!node || node->input.num < definition.inputs.size() || node->output.num < 1) {
    throw std::runtime_error("AddNode failed: " + name);
  }
  std::fill_n(node->input.tensors, node->input.num, VSI_NN_TENSOR_ID_NA);
  for (std::size_t index = 0; index < definition.inputs.size(); ++index) {
    node->input.tensors[index] = tensors.at(definition.inputs[index]);
  }
  node->output.tensors[0] = tensors.at(definition.output);

  // Modify only public parameters; AddNode initialized SDK-owned local fields.
  if (name == "MULTIPLY") {
    node->nn_param.multiply.scale = 1.0f;
  } else if (name == "MATRIXMUL") {
    node->nn_param.matrixmul.transpose[0] = parameters[0];
    node->nn_param.matrixmul.transpose[1] = parameters[1];
  } else if (name == "FCL") {
    node->nn_param.fcl.weights = parameters[0];
    node->nn_param.fcl.axis = 0;
  } else if (name == "GATHER") {
    node->nn_param.gather.axis = parameters[0];
  } else if (name.rfind("REDUCE_", 0) == 0) {
    node->nn_param.reduce.type = name == "REDUCE_SUM"   ? VSI_NN_REDUCE_SUM
                                 : name == "REDUCE_MAX" ? VSI_NN_REDUCE_MAX
                                                        : VSI_NN_REDUCE_MEAN;
    // The immutable case definition outlives graph setup, execution, and release.
    node->nn_param.reduce.axis = reinterpret_cast<const int32_t *>(parameters.data());
    node->nn_param.reduce.axis_num = 1;
    node->nn_param.reduce.keep_dim = TRUE;
  } else if (name == "SWISH") {
    node->nn_param.swish.beta = 1.0f;
    node->nn_param.swish.type = VSI_NN_SWISH;
  } else if (name == "SOFTMAX") {
    node->nn_param.softmax.axis = parameters[0];
    node->nn_param.softmax.beta = 1.0f;
  } else if (name == "ARGMAX") {
    node->nn_param.argmax.axis = parameters[0];
  } else if (name == "LESS") {
    node->nn_param.relational_ops.op = VSI_NN_RELATIONAL_OPS_LESS;
  } else if (name == "CONCAT") {
    node->nn_param.concat.axis = parameters[0];
  } else if (name == "SLICE") {
    if (parameters.empty() || parameters.size() % 2 != 0) {
      throw std::invalid_argument("SLICE requires starts followed by lengths");
    }
    node->nn_param.slice.dims = parameters.size() / 2;
    node->nn_param.slice.start = parameters.data();
    node->nn_param.slice.length = parameters.data() + parameters.size() / 2;
  } else if (name == "PERMUTE") {
    node->nn_param.permute.perm = parameters.data();
    node->nn_param.permute.dim_num = parameters.size();
  } else if (name == "RESHAPE") {
    node->nn_param.reshape.size = parameters.data();
    node->nn_param.reshape.dim_num = parameters.size();
  }
  return node;
}
} // namespace specferry::testing
