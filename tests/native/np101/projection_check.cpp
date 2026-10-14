#include "binary_io.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "test_support.hpp"
#include "vsi_nn_pub.h"

#include <cstddef>
#include <cstdint>
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

Tensor build_projection(GraphBuilder &graph, Tensor input) {
  std::vector<float> identity(width * width, 0);
  for (unsigned index = 0; index < width; ++index) {
    identity[index * width + index] = 1;
  }
  auto weight = graph.floats(identity, {width, width}, f16);
  auto bias = graph.floats(std::vector<float>(width, 0.25f), {width}, f16);
  auto output = graph.tensor(input.spec);

  auto *operation = graph.node(VSI_NN_OP_MATRIXMUL, {input, weight}, output);
  operation->nn_param.matrixmul.transpose[0] = FALSE;
  operation->nn_param.matrixmul.transpose[1] = TRUE;
  return graph.binary(VSI_NN_OP_ADD, output, graph.reshape(bias, {width, 1}));
}

void check_projection(Context &context) {
  GraphBuilder graph(context, 6, 3);
  auto input = graph.tensor({f16, {width, 1}});
  auto output = build_projection(graph, input);
  // Initialize valid FP16 inputs before setup/verify.
  upload_tensor(graph.graph, input.id, encode_floats(std::vector<float>(width, 1)));
  graph.compile({input}, {output});

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

void check_mapped_readback(Context &context) {
  // Match KV backing and full/tail logit blocks; rank 1 checks the public helper contract.
  for (const Shape &shape : {Shape{1024, 16}, Shape{1, 4096}, Shape{1, 1120}, Shape{16}}) {
    const TensorSpec spec{f16, shape};
    GraphBuilder graph(context, 3, 1);
    const auto input = graph.tensor(spec);
    const auto bias = graph.floats(std::vector<float>(spec.elements(), 0.5f), shape, f16);
    const auto output = graph.binary(VSI_NN_OP_ADD, input, bias);
    upload_tensor(graph.graph, input.id, encode_floats(std::vector<float>(spec.elements(), 0)));
    graph.compile({input}, {output});
    for (unsigned iteration = 0; iteration < 4; ++iteration) {
      std::vector<float> values(spec.elements()), expected(values.size());
      for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = 0.125f * (int(index % 31) - 15) + (iteration % 2);
        expected[index] = values[index] + 0.5f;
      }
      upload_tensor(graph.graph, input.id, encode_floats(values));
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.graph.get()); }),
            "run mapped-readback calibration");
      if (read_tensor(graph.graph, output.id, TensorReadback::MappedHost) !=
          encode_floats(expected)) {
        throw std::runtime_error("mapped device output differs from exact FP16 reference");
      }
      std::cout << "PASS: mapped readback rank=" << shape.size() << " bytes=" << spec.bytes()
                << " iteration=" << iteration << std::endl;
    }
    graph.graph.close();
  }
  if (context.memory().live(true) || context.memory().live(false)) {
    throw std::runtime_error("mapped-readback calibration retained application tensor storage");
  }
}
} // namespace

int main(int argc, char **argv) {
  // Reject unknown modes before opening the device; never fall back to FCL.
  const std::string mode = argc > 2 ? argv[2] : "";
  if (argc != 3 || (mode != "matmul-add" && mode != "mapped-readback")) {
    std::cerr << "usage: np101_projection_check OUTPUT matmul-add|mapped-readback\n";
    return 2;
  }
  return run_test(argv[1], [&] {
    std::cout << "Requested mode: " << argv[2] << '\n';
    Context context;
    if (mode == "mapped-readback") {
      check_mapped_readback(context);
    } else {
      check_projection(context);
    }
    context.close();
  });
}
