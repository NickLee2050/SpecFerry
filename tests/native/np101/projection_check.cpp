#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "test_support.hpp"
#include "vsi_nn_pub.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;

constexpr unsigned width = 8;
constexpr auto f16 = DataType::Float16;

Tensor build_projection(GraphBuilder &graph, Tensor input, bool fused) {
  std::vector<float> identity(width * width, 0);
  for (unsigned index = 0; index < width; ++index) {
    identity[index * width + index] = 1;
  }
  auto weight = graph.floats(identity, {width, width}, f16);
  auto bias = graph.floats(std::vector<float>(width, 0.25f), {width}, f16);
  auto output = graph.tensor(input.spec);
  if (fused) {
    auto *operation = graph.node(VSI_NN_OP_FCL, {input, weight, bias}, output);
    operation->nn_param.fcl.weights = width;
    operation->nn_param.fcl.axis = 0;
    return output;
  }

  auto *operation = graph.node(VSI_NN_OP_MATRIXMUL, {input, weight}, output);
  operation->nn_param.matrixmul.transpose[0] = FALSE;
  operation->nn_param.matrixmul.transpose[1] = TRUE;
  return graph.binary(VSI_NN_OP_ADD, output, graph.reshape(bias, {width, 1}));
}

void compile_projection(GraphBuilder &graph, Tensor input, Tensor output, bool fused) {
  const std::vector<vsi_nn_op_t> expected =
      fused ? std::vector<vsi_nn_op_t>{VSI_NN_OP_FCL}
            : std::vector<vsi_nn_op_t>{VSI_NN_OP_MATRIXMUL, VSI_NN_OP_RESHAPE, VSI_NN_OP_ADD};
  if (graph.graph.get()->cur_nid != expected.size()) {
    throw std::runtime_error("projection node count differs from the requested mode");
  }
  std::cout << "Graph: " << (fused ? "FCL(axis=0, weights=8)" : "MATRIXMUL -> RESHAPE(bias) -> ADD")
            << "; FP16 input [8,1], identity weight [8,8], bias [8]=0.25\n";
  for (unsigned index = 0; index < expected.size(); ++index) {
    auto *operation = vsi_nn_GetNode(graph.graph.get(), index);
    if (!operation || operation->op != expected[index]) {
      throw std::runtime_error("projection operator differs from the requested mode");
    }
    std::cout << "node " << index << " confirmed, op=" << operation->op << '\n';
  }

  auto input_id = input.id, output_id = output.id;
  if (!vsi_nn_SetGraphInputs(graph.graph.get(), &input_id, 1) ||
      !vsi_nn_SetGraphOutputs(graph.graph.get(), &output_id, 1)) {
    throw std::runtime_error("cannot declare projection inputs/outputs");
  }
  check(sdk_call("vsi_nn_SetupGraph", [&] { return vsi_nn_SetupGraph(graph.graph.get(), FALSE); }),
        "SetupGraph");
  for (unsigned index = 0; index < graph.graph.get()->cur_tid; ++index) {
    auto *tensor = vsi_nn_GetTensor(graph.graph.get(), index);
    if (!tensor || !tensor->t) {
      throw std::runtime_error("null projection tensor before VerifyGraph");
    }
    check(vxGetStatus(reinterpret_cast<vx_reference>(tensor->t)), "projection tensor status");
  }
  std::cout << "All tensor references valid; entering VerifyGraph" << std::endl;
  check(sdk_call("vsi_nn_VerifyGraph", [&] { return vsi_nn_VerifyGraph(graph.graph.get()); }),
        "VerifyGraph");
}

void check_projection(Context &context, bool fused) {
  GraphBuilder graph(context, 6, 3);
  auto input = graph.tensor({f16, {width, 1}});
  auto output = build_projection(graph, input, fused);
  // Both modes start with valid FP16 data before setup/verify, as in the vendor demo.
  upload_tensor(graph.graph, input.id, encode_floats(std::vector<float>(width, 1)));
  compile_projection(graph, input, output, fused);

  for (const float value : {1.0f, -0.5f}) {
    upload_tensor(graph.graph, input.id, encode_floats(std::vector<float>(width, value)));
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.graph.get()); }),
          "RunGraph");
    if (read_tensor(graph.graph, output.id) !=
        encode_floats(std::vector<float>(width, value + 0.25f))) {
      throw std::runtime_error("projection differs from the exact FP16 reference");
    }
    std::cout << "PASS: all 8 outputs = " << value + 0.25f << std::endl;
  }
  graph.graph.close();
}
} // namespace

int main(int argc, char **argv) {
  // Reject unknown modes before opening the device; never fall back to FCL.
  if (argc != 3 || (std::string(argv[2]) != "fcl" && std::string(argv[2]) != "matmul-add")) {
    std::cerr << "usage: np101_projection_check OUTPUT fcl|matmul-add\n";
    return 2;
  }
  return run_test(argv[1], [&] {
    std::cout << "Requested mode: " << argv[2] << '\n';
    Context context;
    check_projection(context, std::string(argv[2]) == "fcl");
    context.close();
  });
}
