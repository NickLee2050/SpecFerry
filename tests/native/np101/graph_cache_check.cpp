#include "inference/generation.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/tensor.hpp"
#include "vsi_nn_pub.h"

#include <array>
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

Tensor destination(GraphBuilder &g, Tensor parent, const std::string &mode) {
  if (mode == "separate") {
    return g.tensor(parent.spec);
  }
  if (mode == "shared") {
    return {retain_tensor(g.graph, parent.id, g.graph), parent.spec};
  }
  auto output = g.tensor(parent.spec);
  auto *wrapper = vsi_nn_GetTensor(g.graph.get(), output.id);
  auto *source = vsi_nn_GetTensor(g.graph.get(), parent.id);
  if (!wrapper || !source || !source->t) {
    throw std::runtime_error("cache tensor is not materialized");
  }
  std::array<vsi_size_t, 3> begin{0, 0, 0}, end{width, capacity, heads};
  auto view = vsi_nn_CreateViewTensor(g.graph.get(), begin.data(), end.data(), source);
  if (!view) {
    throw std::runtime_error("cache view creation returned null");
  }
  const auto status = vxGetStatus(reinterpret_cast<vx_reference>(view));
  if (status != VX_SUCCESS) {
    vxReleaseTensor(&view);
    check(status, "create cache view");
  }
  if (wrapper->t) {
    const auto release = vxReleaseTensor(&wrapper->t);
    if (release != VX_SUCCESS) {
      vxReleaseTensor(&view);
      check(release, "release unused cache output");
    }
  }
  wrapper->t = view; // The graph wrapper owns this reference; the parent outlives it.
  return output;
}

bool experiment(Context &context, const std::string &mode, const std::filesystem::path &output) {
  GraphBuilder g(context, 64, 32);
  std::string phase = "construct";
  bool passed = false;
  unsigned completed = 0;
  std::ofstream report(output / "cache.txt");
  try {
    const TensorSpec cache_spec{f16, {width, capacity, heads}};
    auto keys = Tensor{
        add_tensor(g.graph, cache_spec, false, std::vector<std::uint8_t>(cache_spec.bytes())),
        cache_spec};
    auto values = Tensor{
        add_tensor(g.graph, cache_spec, false, std::vector<std::uint8_t>(cache_spec.bytes())),
        cache_spec};
    auto input = g.tensor({f16, {width, 1, heads}});
    auto indices = g.tensor({DataType::Int32, {2, heads}});
    auto length = g.tensor({DataType::Int32, {1}});
    // Producing K/V in the same graph tests ordering, not just cache upload/readback.
    auto next_key = g.binary(VSI_NN_OP_ADD, input, g.scalar(0.25f, f16));
    auto next_value = g.binary(VSI_NN_OP_ADD, input, g.scalar(0.5f, f16));
    auto updated_keys = destination(g, keys, mode);
    auto updated_values = destination(g, values, mode);
    g.node(VSI_NN_OP_SCATTER_ND_UPDATE, {keys, indices, g.reshape(next_key, {width, heads})},
           updated_keys);
    g.node(VSI_NN_OP_SCATTER_ND_UPDATE, {values, indices, g.reshape(next_value, {width, heads})},
           updated_values);
    auto query = g.floats(std::vector<float>(width * heads), {width, 1, heads}, f16);
    auto attention = attention_core(g, query, updated_keys, updated_values, length, 1.0f);
    phase = "compile";
    std::cout << "Compile producer -> cache update -> attention (" << mode << ")" << std::endl;
    g.compile({input, indices, length, keys, values},
              {attention.attended, updated_keys, updated_values});
    phase = "execute";
    std::vector<float> expected_keys(cache_spec.elements()), expected_values(cache_spec.elements());
    // A full prefix followed by a shorter reset must mask the old suffix.
    for (unsigned step = 0; step < capacity + 2; ++step) {
      const auto position = step % capacity;
      if (mode == "separate") {
        upload_tensor(g.graph, keys.id, half_bytes(expected_keys));
        upload_tensor(g.graph, values.id, half_bytes(expected_values));
      }
      std::vector<float> next(width * heads);
      std::vector<std::int32_t> coordinates;
      for (unsigned head = 0; head < heads; ++head) {
        // Scatter coordinates follow logical outer axes: [head, token], while
        // SDK shape arrays list the contiguous dimension first.
        coordinates.insert(coordinates.end(), {std::int32_t(head), std::int32_t(position)});
        for (unsigned dim = 0; dim < width; ++dim) {
          const auto index = head * width + dim;
          next[index] = 0.125f * step + 0.25f * head;
          const auto offset = (head * capacity + position) * width + dim;
          expected_keys[offset] = next[index] + 0.25f;
          expected_values[offset] = next[index] + 0.5f;
        }
      }
      upload_tensor(g.graph, input.id, half_bytes(next));
      upload_tensor(g.graph, indices.id, bytes(coordinates));
      upload_tensor(g.graph, length.id, bytes<std::int32_t>({std::int32_t(position + 1)}));
      if (!vxIsGraphVerified(g.graph.get()->g)) {
        throw std::runtime_error("data upload invalidated the compiled graph");
      }
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(g.graph.get()); }),
            "run cache graph");
      // Read attention before touching either parent on the host.
      auto actual = read_tensor(g.graph, attention.attended.id);
      if (actual.size() != width * heads * 2) {
        throw std::runtime_error("attention output has an unexpected byte count");
      }
      for (unsigned head = 0; head < heads; ++head) {
        float mean = 0;
        for (unsigned token = 0; token <= position; ++token) {
          mean += expected_values[(head * capacity + token) * width];
        }
        mean /= position + 1;
        for (unsigned dim = 0; dim < width; ++dim) {
          std::uint16_t word;
          std::memcpy(&word, actual.data() + (head * width + dim) * 2, 2);
          const auto value = vsi_nn_Fp16ToFp32(word);
          if (!std::isfinite(value) || std::abs(value - mean) > 0.004f) {
            throw std::runtime_error("attention differs after append/reset at step " +
                                     std::to_string(step) + ", head " + std::to_string(head) +
                                     ": expected " + std::to_string(mean) + ", got " +
                                     std::to_string(value));
          }
        }
      }
      ++completed;
    }
    phase = "final_readback";
    auto key_source = mode == "separate" ? updated_keys : keys;
    auto value_source = mode == "separate" ? updated_values : values;
    if (read_tensor(g.graph, key_source.id) != half_bytes(expected_keys) ||
        read_tensor(g.graph, value_source.id) != half_bytes(expected_values)) {
      throw std::runtime_error("parent cache bytes differ after final execution");
    }
    passed = true;
  } catch (const std::exception &error) {
    report << "failure_phase: " << phase << "\nerror: " << error.what() << '\n';
    std::cerr << phase << ": " << error.what() << '\n';
  }
  g.graph.close();
  report << "mode: " << mode << "\ncompleted_steps: " << completed
         << "\nstatus: " << (passed ? "numerical_pass" : "failed") << "\ngraph_released: true\n";
  if (!report) {
    throw std::runtime_error("cannot write cache report");
  }
  return passed;
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
  if (argc != 3 || (std::string(argv[2]) != "separate" && std::string(argv[2]) != "view" &&
                    std::string(argv[2]) != "shared" && std::string(argv[2]) != "stack" &&
                    std::string(argv[2]) != "stack-block")) {
    std::cerr << "usage: np101_graph_cache_check OUTPUT_DIRECTORY "
                 "separate|view|shared|stack|stack-block\n";
    return 2;
  }
  try {
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    SdkTimings timings(directory);
    Context context;
    const std::string mode(argv[2]);
    const bool passed = mode == "stack" || mode == "stack-block"
                            ? append_experiment(context, directory, mode == "stack" ? 1 : 4)
                            : experiment(context, mode, directory);
    context.close();
    timings.save();
    std::cout << "Context released; " << (passed ? "PASS" : "FAIL") << std::endl;
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
