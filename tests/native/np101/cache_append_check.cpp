#include "binary_io.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "test_support.hpp"
#include "vsi_nn_pub.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;

constexpr unsigned width = 8, capacity = 8;

void check_append(Context &context, const std::filesystem::path &output, bool three_dimensional) {
  const unsigned heads = three_dimensional ? 2 : 1;
  const Shape column_shape = three_dimensional ? Shape{width, 1, heads} : Shape{width, 1};
  const Shape cache_shape =
      three_dimensional ? Shape{width, capacity, heads} : Shape{width, capacity};
  GraphBuilder graph(context, 3, 1);
  auto column = graph.tensor({DataType::Float16, column_shape});
  auto index = graph.tensor({DataType::Int32, {1}});
  auto cache = graph.tensor({DataType::Float16, cache_shape});
  std::vector<float> expected(cache.spec.elements(), 0);
  std::vector<float> values(column.spec.elements(), 1);

  // Allocate the entire cache once. Only its valid prefix grows; no tensor is resized.
  upload_tensor(graph.graph, cache.id, encode_floats(expected));
  upload_tensor(graph.graph, column.id, encode_floats(values));
  upload_tensor(graph.graph, index.id, as_bytes<std::int32_t>({0}));
  // Call the SDK directly: the production helper deliberately accepts only 2-D writes.
  // Whether this same-rank 3-D configuration is supported requires SDK confirmation.
  graph.node(VSI_NN_OP_TENSORSTACKCONCAT, {column, index}, cache)->nn_param.tensorstackconcat.axis =
      1;
  std::cout << "Graph: one TENSORSTACKCONCAT(axis=1); no RESHAPE nodes\n"
            << "FP16 update: " << (three_dimensional ? "[8,1,2]" : "[8,1]")
            << "; preallocated cache: " << (three_dimensional ? "[8,8,2]" : "[8,8]")
            << "; INT32 index: [1]\n"
            << "Append " << values.size() << " elements per step; cache capacity stays "
            << expected.size() << " elements. Inputs and cache initialized before SetupGraph.\n";
  graph.compile({column, index}, {cache});

  // Two writes test both the new slot and preservation of the preceding slot/suffix.
  for (unsigned position = 0; position < 2; ++position) {
    for (unsigned head = 0; head < heads; ++head) {
      for (unsigned dim = 0; dim < width; ++dim) {
        const float value = 1 + position * 32 + head * width + dim;
        values[head * width + dim] = value;
        expected[(head * capacity + position) * width + dim] = value;
      }
    }
    upload_tensor(graph.graph, column.id, encode_floats(values));
    upload_tensor(graph.graph, index.id, as_bytes<std::int32_t>({std::int32_t(position)}));
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.graph.get()); }),
          "append cache column");
    const auto actual = read_tensor(graph.graph, cache.id);
    const auto reference = encode_floats(expected);
    write_bytes(output / ("expected." + std::to_string(position) + ".bin"), reference);
    write_bytes(output / ("actual." + std::to_string(position) + ".bin"), actual);
    if (actual != reference) {
      throw std::runtime_error("cache append changed the wrong slot or lost previous data");
    }
    std::cout << "PASS: slot " << position << ", previous data and untouched suffix\n";
  }
  graph.graph.close();
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || (std::string(argv[2]) != "2" && std::string(argv[2]) != "3")) {
    std::cerr << "usage: np101_cache_append_check OUTPUT RANK[2|3]\n";
    return 2;
  }
  return run_test(argv[1], [&] {
    Context context;
    check_append(context, argv[1], std::string(argv[2]) == "3");
    context.close();
  });
}
