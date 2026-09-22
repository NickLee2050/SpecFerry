#include "np101/context.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;

void save(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) {
    throw std::runtime_error("cannot save sampled tokens");
  }
}

class Sampler : public GraphBuilder {
public:
  Tensor logits, seed, samples;

  Sampler(Context &context, unsigned classes, unsigned count, DataType type, bool promote)
      : GraphBuilder(context, 4, 2), logits(tensor({type, {classes, 1}})),
        seed(tensor({DataType::Int32, {4}})), samples(tensor({DataType::Int32, {count, 1}})) {
    auto input = promote ? convert(logits, DataType::Float32) : logits;
    auto *operation = node(VSI_NN_OP_RANDOM_MULTINOMIAL, {input, seed}, samples);
    operation->nn_param.random_multinomial.sample_num = count;
    std::cout << "compile classes=" << classes << " samples=" << count << std::endl;
    compile({logits, seed}, {samples});
  }

  std::vector<std::uint8_t> run(const std::vector<float> &values, std::int32_t key,
                                std::int32_t counter = 17) {
    vsi_nn_dtype_t dtype{};
    dtype.vx_type =
        logits.spec.type == DataType::Float32 ? VSI_NN_TYPE_FLOAT32 : VSI_NN_TYPE_FLOAT16;
    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    std::vector<std::uint8_t> bytes(logits.spec.bytes());
    for (unsigned index = 0; index < values.size(); ++index) {
      check(vsi_nn_Float32ToDtype(values[index],
                                  bytes.data() + index * dtype_bytes(logits.spec.type), &dtype),
            "encode sampling logits");
    }
    upload_tensor(graph, logits.id, bytes);
    const std::array<std::int32_t, 4> words{key, counter, 0, 0};
    bytes.resize(sizeof(words));
    std::memcpy(bytes.data(), words.data(), bytes.size());
    upload_tensor(graph, seed.id, bytes);
    check(vsi_nn_RunGraph(graph.get()), "random multinomial");
    return read_tensor(graph, samples.id);
  }
};
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: np101_sampling_check OUTPUT F16|F32|F16_TO_F32 CLASSES SAMPLES\n";
    return 2;
  }
  try {
    const bool promote = std::string(argv[2]) == "F16_TO_F32";
    const auto type = parse_dtype(promote ? "F16" : argv[2]);
    const auto shape = parse_shape(std::string(argv[3]) + "," + argv[4]);
    const auto classes = shape[0], count = shape[1];
    if ((type != DataType::Float16 && type != DataType::Float32) || classes < 8 ||
        classes > 50272 || count > 8192) {
      throw std::invalid_argument("unsupported sampling diagnostic dimensions");
    }
    const std::filesystem::path output(argv[1]);
    std::filesystem::create_directories(output);
    Context context;
    Sampler sampler(context, classes, count, type, promote);
    std::vector<float> logits(classes, 0);
    for (unsigned trial = 0; trial < 3; ++trial) {
      save(output / ("uniform." + std::to_string(trial) + ".bin"),
           sampler.run(logits, trial == 1 ? 2718 : 3141));
    }
    save(output / "counter.bin", sampler.run(logits, 3141, 18));
    // Softmax probabilities proportional to 1..8, with all other classes suppressed.
    std::fill(logits.begin(), logits.end(), -1000.0f);
    for (unsigned index = 0; index < 8; ++index) {
      logits[index] = std::log(float(index + 1));
    }
    save(output / "weighted.bin", sampler.run(logits, 3141));
    std::fill(logits.begin(), logits.end(), -1000.0f);
    logits.back() = 0;
    save(output / "last.bin", sampler.run(logits, 3141));
    sampler.graph.close();
    Sampler fresh(context, classes, count, type, promote);
    save(output / "fresh.bin", fresh.run(std::vector<float>(classes, 0), 3141));
    fresh.graph.close();
    context.close();
    std::ofstream(output / "execution.json") << "{\"status\":\"executed\",\"released\":true}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
