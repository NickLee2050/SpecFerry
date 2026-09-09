#include "np101/tensor.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>

namespace {
using namespace specferry::np101;

std::vector<std::uint8_t> bytes(const std::array<float, 4> &values) {
  std::vector<std::uint8_t> result(sizeof(values));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

void binary_node(Graph &graph, vsi_nn_op_t operation, vsi_nn_tensor_id_t left,
                 vsi_nn_tensor_id_t right, vsi_nn_tensor_id_t output) {
  auto *node = vsi_nn_AddNode(graph.get(), operation, 2, 1, nullptr);
  if (!node) {
    throw std::runtime_error("AddNode failed");
  }
  node->input.tensors[0] = left;
  node->input.tensors[1] = right;
  node->output.tensors[0] = output;
  if (operation == VSI_NN_OP_MULTIPLY) {
    node->nn_param.multiply.scale = 1.0f;
  }
  if (!vsi_nn_SetGraphInputs(graph.get(), &left, 1) ||
      !vsi_nn_SetGraphOutputs(graph.get(), &output, 1)) {
    throw std::runtime_error("SetGraphInputs/Outputs failed");
  }
  check(vsi_nn_SetupGraph(graph.get(), FALSE), "SetupGraph");
  check(vsi_nn_VerifyGraph(graph.get()), "VerifyGraph");
}

void check_sharing() {
  Context context;
  Graph producer(context, 3, 1);
  Graph consumer(context, 3, 1);
  const TensorSpec spec{DataType::Float32, {4, 1}};
  auto input = add_tensor(producer, spec);
  auto bias = add_tensor(producer, spec, true, bytes({0.25f, 0.5f, -0.25f, -0.5f}));
  auto intermediate = add_tensor(producer, spec);
  binary_node(producer, VSI_NN_OP_ADD, input, bias, intermediate);

  // Destruction order: detach the borrowed wrapper, release consumer, producer,
  // then context. It is the same SDK tensor in both graph maps.
  TensorAttachment shared(producer, intermediate, consumer);
  auto scale = add_tensor(consumer, spec, true, bytes({2, 2, 2, 2}));
  auto output = add_tensor(consumer, spec);
  binary_node(consumer, VSI_NN_OP_MULTIPLY, shared.id(), scale, output);
  if (vsi_nn_GetTensor(producer.get(), intermediate) !=
      vsi_nn_GetTensor(consumer.get(), shared.id())) {
    throw std::runtime_error("graph attachment did not retain tensor identity");
  }

  for (unsigned step = 0; step < 2; ++step) {
    std::array<float, 4> values{float(step), 1 + float(step), 2 + float(step), 3 + float(step)};
    upload_tensor(producer, input, bytes(values));
    check(vsi_nn_RunGraph(producer.get()), "RunGraph(producer)");
    check(vsi_nn_RunGraph(consumer.get()), "RunGraph(consumer)");
    auto result = read_tensor(consumer, output);
    std::array<float, 4> actual{};
    std::memcpy(actual.data(), result.data(), sizeof(actual));
    const std::array<float, 4> bias_values{0.25f, 0.5f, -0.25f, -0.5f};
    for (std::size_t index = 0; index < actual.size(); ++index) {
      auto expected = (values[index] + bias_values[index]) * 2;
      if (!std::isfinite(actual[index]) || std::abs(actual[index] - expected) > 1e-4f) {
        throw std::runtime_error("cross-graph result mismatch");
      }
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: np101_graph_sharing_check REPORT_JSON\n";
    return 2;
  }
  try {
    if (!dlsym(RTLD_DEFAULT, "vsi_nn_AttachTensorToGraph")) {
      std::ofstream report(argv[1]);
      report << "{\"status\":\"unavailable\",\"symbol_exported\":false,"
                "\"reason\":\"SDK does not export vsi_nn_AttachTensorToGraph\","
                "\"hardware_execution_proven\":false}\n";
      if (!report) {
        throw std::runtime_error("failed to write sharing report");
      }
      return 3;
    }
    check_sharing();
    std::ofstream report(argv[1]);
    report << "{\"status\":\"numerical_pass\",\"steps\":2,\"tensor_identity_shared\":true,"
              "\"application_intermediate_readbacks\":0,\"hardware_execution_proven\":false,"
              "\"device_residency_verified\":false}\n";
    if (!report) {
      throw std::runtime_error("failed to write sharing report");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
