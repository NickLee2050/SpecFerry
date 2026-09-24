#include "inference/generation.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/tensor.hpp"
#include "vsi_nn_pub.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
constexpr unsigned width = 8, heads = 2, capacity = 8;
constexpr auto f16 = DataType::Float16;

template <class T> std::vector<std::uint8_t> bytes(const std::vector<T> &values) {
  std::vector<std::uint8_t> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

std::vector<std::uint8_t> half_bytes(const std::vector<float> &values) {
  vsi_nn_dtype_t dtype{};
  dtype.vx_type = VSI_NN_TYPE_FLOAT16;
  std::vector<std::uint8_t> result(values.size() * 2);
  for (std::size_t i = 0; i < values.size(); ++i) {
    check(vsi_nn_Float32ToDtype(values[i], result.data() + i * 2, &dtype), "encode FP16");
  }
  return result;
}

class AppendGraph : public GraphBuilder {
public:
  Tensor input, index, limits;
  std::vector<Tensor> attended;
  const unsigned block;

  AppendGraph(Context &context, Graph &storage, const std::vector<Tensor> &parents, unsigned size)
      : GraphBuilder(context, 160, 100), block(size) {
    input = tensor({f16, {width, block}});
    index = tensor({DataType::Int32, {1}});
    limits = tensor({DataType::Int32, {block}});
    for (unsigned head = 0; head < heads; ++head) {
      std::vector<Tensor> caches;
      for (unsigned kind = 0; kind < 2; ++kind) {
        const auto parent = parents.at(head * 2 + kind);
        auto original = bind({storage, parent.id}, parent.spec);
        auto target = block == 1
                          ? original
                          : storage_reshape(*this, original, {width * block, capacity / block});
        auto produced = binary(VSI_NN_OP_ADD, input, scalar(0.25f * (head + kind + 1), f16));
        auto updated = indexed_append(*this, reshape(produced, {width * block, 1}), index, target);
        caches.push_back(storage_reshape(*this, updated, {width, capacity, 1}));
      }
      std::vector<float> query_values(width * block);
      for (unsigned column = 0; column < block; ++column) {
        query_values[column * width] = 0.5f;
      }
      auto query = floats(query_values, {width, block, 1}, f16);
      attended.push_back(attention_core(*this, query, caches[0], caches[1], limits, 1).attended);
    }
    compile({input, index, limits}, attended);
  }
};

bool append_experiment(Context &context, const std::filesystem::path &output, unsigned block) {
  Graph storage(context, heads * 2, 0);
  const TensorSpec spec{f16, {width, capacity}};
  std::vector<Tensor> parents;
  std::vector<std::vector<float>> expected(heads * 2, std::vector<float>(spec.elements()));
  for (unsigned index = 0; index < heads * 2; ++index) {
    parents.push_back({add_tensor(storage, spec, false, half_bytes(expected[index])), spec});
  }
  AppendGraph decode(context, storage, parents, 1);
  std::unique_ptr<AppendGraph> prefill;
  if (block != 1) {
    prefill = std::make_unique<AppendGraph>(context, storage, parents, block);
  }
  unsigned launches = 0, request = 0;
  // Full prefix, shorter reset with a one-token tail, then full prefix again.
  for (unsigned length : {capacity, 5u, capacity}) {
    for (auto chunk : specferry::inference::plan_prefill(length, block)) {
      auto &g = chunk.count == 1 ? decode : *prefill;
      std::vector<float> source(width * chunk.count);
      std::vector<std::int32_t> limits;
      for (unsigned column = 0; column < chunk.count; ++column) {
        const auto position = chunk.begin + column;
        const auto value = float(request * 16 + position) * 0.125f;
        limits.push_back(position + 1);
        for (unsigned dim = 0; dim < width; ++dim) {
          source[column * width + dim] = value;
          for (unsigned head = 0; head < heads; ++head) {
            for (unsigned kind = 0; kind < 2; ++kind) {
              expected[head * 2 + kind][position * width + dim] = value + 0.25f * (head + kind + 1);
            }
          }
        }
      }
      upload_tensor(g.graph, g.input.id, half_bytes(source));
      upload_tensor(g.graph, g.index.id,
                    bytes<std::int32_t>({std::int32_t(chunk.begin / g.block)}));
      upload_tensor(g.graph, g.limits.id, bytes(limits));
      if (!vxIsGraphVerified(g.graph.get()->g)) {
        throw std::runtime_error("indexed append invalidated its fixed graph");
      }
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(g.graph.get()); }),
            "run indexed append");
      ++launches;
      for (unsigned head = 0; head < heads; ++head) {
        const auto actual = read_tensor(g.graph, g.attended[head].id);
        if (actual.size() != source.size() * 2) {
          throw std::runtime_error("unexpected per-head attention size");
        }
        for (unsigned column = 0; column < chunk.count; ++column) {
          float weighted = 0, normalization = 0;
          for (int token = 0; token < limits[column]; ++token) {
            const auto score = std::exp(0.5f * expected[head * 2][token * width]);
            weighted += score * expected[head * 2 + 1][token * width];
            normalization += score;
          }
          const auto mean = weighted / normalization;
          for (unsigned dim = 0; dim < width; ++dim) {
            std::uint16_t word;
            std::memcpy(&word, actual.data() + (column * width + dim) * 2, 2);
            const auto value = vsi_nn_Fp16ToFp32(word);
            if (!std::isfinite(value) || std::abs(value - mean) > 0.004f) {
              throw std::runtime_error("indexed append attention/causal mask mismatch");
            }
          }
        }
      }
    }
    for (unsigned index = 0; index < parents.size(); ++index) {
      if (read_tensor(storage, parents[index].id) != half_bytes(expected[index])) {
        throw std::runtime_error("indexed cache lost history or changed an untouched suffix");
      }
    }
    ++request;
  }
  if (prefill) {
    prefill->graph.close();
  }
  decode.graph.close();
  prefill.reset();
  storage.close();
  std::ofstream report(output / "cache.txt");
  report << "status: numerical_pass\nmode: " << (block == 1 ? "stack" : "stack-block")
         << "\nblock: " << block << "\nlaunches: " << launches
         << "\nconsumed_tokens: 21\nrevalidations: 0\ngraph_released: true\n";
  if (!report) {
    throw std::runtime_error("cannot save indexed cache result");
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || (std::string(argv[2]) != "stack" && std::string(argv[2]) != "stack-block")) {
    std::cerr << "usage: np101_graph_cache_check OUTPUT stack|stack-block\n";
    return 2;
  }
  try {
    std::filesystem::create_directories(argv[1]);
    Context context;
    const bool passed =
        append_experiment(context, argv[1], std::string(argv[2]) == "stack" ? 1 : 4);
    context.close();
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
