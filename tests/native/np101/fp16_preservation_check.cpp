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
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;

bool check_preservation(Context &context, const std::filesystem::path &directory, bool select) {
  // Construct exact FP16 bits, avoiding host or SDK float-conversion rounding.
  // Include both signs, the subnormal/normal boundary, zero and ordinary values.
  const std::vector<std::uint16_t> expected{0x0000, 0x0001, 0x0002, 0x0010, 0x0100, 0x0200,
                                            0x03ff, 0x0400, 0x8001, 0x8100, 0x8200, 0x83ff,
                                            0x8400, 0x3555, 0x3c00, 0xbc00};
  std::vector<std::uint16_t> left(expected.size()), right(expected.size());
  std::vector<std::uint8_t> condition(expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    condition[index] = index % 2 == 0;
    (condition[index] ? left : right)[index] = expected[index];
  }

  // Every element has one value and one zero. ADD and SELECT should both
  // preserve that value; alternating the condition exercises both SELECT inputs.
  GraphBuilder graph(context, 4, 1);
  const TensorSpec spec{DataType::Float16, {16, 1}};
  auto a = graph.tensor(spec), b = graph.tensor(spec), output = graph.tensor(spec);
  std::vector<Tensor> inputs{a, b};
  if (select) {
    auto mask = graph.tensor({DataType::Bool8, {16, 1}});
    upload_tensor(graph.graph, mask.id, condition);
    graph.node(VSI_NN_OP_SELECT, {mask, a, b}, output);
    inputs.push_back(mask);
  } else {
    graph.node(VSI_NN_OP_ADD, {a, b}, output);
  }
  const auto left_bytes = as_bytes(left), right_bytes = as_bytes(right);
  upload_tensor(graph.graph, a.id, left_bytes);
  upload_tensor(graph.graph, b.id, right_bytes);
  if (read_tensor(graph.graph, a.id) != left_bytes ||
      read_tensor(graph.graph, b.id) != right_bytes) {
    throw std::runtime_error("input bits changed before graph compilation");
  }
  write_bytes(directory / "expected.bin", as_bytes(expected));
  write_bytes(directory / "left.bin", left_bytes);
  write_bytes(directory / "right.bin", right_bytes);
  std::cout << "Graph: " << (select ? "SELECT" : "ADD")
            << ", one node; mutable FP16 [16,1] inputs and output\n";
  graph.compile(inputs, {output});
  check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.graph.get()); }),
        "run FP16 preservation check");
  const auto actual_bytes = read_tensor(graph.graph, output.id);
  if (actual_bytes.size() != expected.size() * sizeof(std::uint16_t)) {
    throw std::runtime_error("wrong output byte count");
  }
  std::vector<std::uint16_t> actual(expected.size());
  std::memcpy(actual.data(), actual_bytes.data(), actual_bytes.size());
  write_bytes(directory / "actual.bin", actual_bytes);
  const bool inputs_unchanged =
      read_tensor(graph.graph, a.id) == left_bytes && read_tensor(graph.graph, b.id) == right_bytes;

  unsigned mismatches = 0, flushed_subnormals = 0;
  std::cout << "index  expected_bits  actual_bits  result\n";
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const bool equal = actual[index] == expected[index];
    const bool subnormal = (expected[index] & 0x7c00) == 0 && (expected[index] & 0x03ff) != 0;
    mismatches += !equal;
    flushed_subnormals += subnormal && (actual[index] & 0x7fff) == 0;
    std::cout << std::dec << index << "  0x" << std::hex << std::setfill('0') << std::setw(4)
              << expected[index] << "  0x" << std::setw(4) << actual[index] << std::dec << "  "
              << (equal ? "PASS" : "FAIL") << '\n';
  }
  std::cout << "Mismatches: " << mismatches << '/' << expected.size()
            << "; subnormals changed to zero: " << flushed_subnormals
            << "; input bits unchanged: " << (inputs_unchanged ? "yes" : "no") << '\n';
  graph.graph.close();
  return mismatches == 0 && inputs_unchanged;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || (std::string(argv[2]) != "add" && std::string(argv[2]) != "select")) {
    std::cerr << "usage: np101_fp16_preservation_check OUTPUT add|select\n";
    return 2;
  }
  return run_test(argv[1], [&] {
    Context context;
    const bool passed = check_preservation(context, argv[1], std::string(argv[2]) == "select");
    context.close();
    // A numerical failure is reported only after explicit graph/context release.
    if (!passed) {
      throw std::runtime_error("FP16 values were not preserved; graph/context released");
    }
  });
}
