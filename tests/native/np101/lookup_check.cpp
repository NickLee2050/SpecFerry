#include "binary_io.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "test_support.hpp"
#include "vsi_nn_pub.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;

constexpr unsigned width = 8;
constexpr unsigned rows = 8;

std::vector<std::uint8_t> table_bytes() {
  std::array<std::uint16_t, width * rows> values{};
  for (unsigned index = 0; index < values.size(); ++index) {
    values[index] = 0x3000 + index;
  }
  // Include signed zero, subnormals and the smallest normal FP16 values.
  // Lookup must copy their bits without performing floating-point arithmetic.
  const std::array<std::uint16_t, width> edge_values{0x0000, 0x8000, 0x0001, 0x8001,
                                                     0x03ff, 0x83ff, 0x0400, 0x8400};
  std::memcpy(values.data(), edge_values.data(), sizeof(edge_values));
  std::vector<std::uint8_t> bytes(sizeof(values));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

void check_lookup(Context &context, bool embedding) {
  const TensorSpec table_spec{DataType::Float16, {width, rows}};
  const TensorSpec result_spec{DataType::Float16, {width, 1}};
  const auto table_data = table_bytes();

  // Match Vocabulary's ordinary mutable backing and retained graph references.
  Graph storage(context, 2, 0);
  auto table_id = add_tensor(storage, table_spec, false, table_data);
  auto result_id =
      add_tensor(storage, result_spec, false, std::vector<std::uint8_t>(result_spec.bytes()));
  GraphBuilder lookup(context, 3, 1);
  auto index = lookup.tensor({DataType::Int32, {1}});
  auto table = lookup.bind({storage, table_id}, table_spec);
  auto result = lookup.bind({storage, result_id}, result_spec);
  if (embedding) {
    lookup.node(VSI_NN_OP_EMBEDDING_LOOKUP, {index, table}, result);
  } else {
    lookup.node(VSI_NN_OP_GATHER, {table, index}, result)->nn_param.gather.axis = 1;
  }

  // Unlike the historical selection fixture, initialize a valid index before
  // Verify. Both alternatives use this same condition; no private SDK state is set.
  upload_tensor(lookup.graph, index.id, as_bytes<std::int32_t>({0}));
  std::cout << "Graph: " << (embedding ? "EMBEDDING_LOOKUP" : "GATHER(axis=1)")
            << ", 1 node, 3 tensors; FP16 [8,8] + INT32 [1] -> FP16 [8,1]\n"
            << "Setup and verify ..." << std::endl;
  lookup.compile({table, index}, {result});

  const std::array<std::int32_t, 10> indices{0, 7, 3, 1, 6, 2, 5, 4, 0, 7};
  for (const auto row : indices) {
    upload_tensor(lookup.graph, index.id, as_bytes<std::int32_t>({row}));
    std::cout << "Read row " << row << " ..." << std::endl;
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(lookup.graph.get()); }),
          "run lookup");
    const auto actual = read_tensor(lookup.graph, result.id);
    const auto offset = row * result_spec.bytes();
    for (std::size_t byte = 0; byte < actual.size(); ++byte) {
      if (actual[byte] != table_data[offset + byte]) {
        throw std::runtime_error("row " + std::to_string(row) + ", byte " + std::to_string(byte) +
                                 ": expected " + std::to_string(table_data[offset + byte]) +
                                 ", got " + std::to_string(actual[byte]));
      }
    }
  }
  if (read_tensor(storage, table_id) != table_data) {
    throw std::runtime_error("lookup changed the source table");
  }

  std::cout << "10 exact row checks passed; releasing ..." << std::endl;
  lookup.graph.close();
  storage.close();
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || (std::string(argv[2]) != "embedding" && std::string(argv[2]) != "gather")) {
    std::cerr << "usage: np101_lookup_check OUTPUT embedding|gather\n";
    return 2;
  }
  return run_test(argv[1], [&] {
    Context context;
    check_lookup(context, std::string(argv[2]) == "embedding");
    context.close();
  });
}
