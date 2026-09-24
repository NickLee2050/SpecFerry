#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using specferry::np101::check;

struct ConvolutionData {
  std::vector<float> input, weights, bias;
  std::vector<std::uint8_t> input_fp16, weights_fp16, bias_fp16;
};

struct GraphIO {
  vsi_nn_tensor_id_t input, output;
};

static vsi_nn_dtype_t dtype() {
  vsi_nn_dtype_t d{};
  d.vx_type = VSI_NN_TYPE_FLOAT16;
  d.qnt_type = VSI_NN_QNT_TYPE_NONE;
  d.fmt = VSI_NN_DIM_FMT_NCHW;
  return d;
}

static std::vector<uint8_t> f16(const std::vector<float> &data) {
  std::vector<uint8_t> result(data.size() * 2);
  auto d = dtype();
  for (size_t i = 0; i < data.size(); ++i) {
    if (!std::isfinite(data[i])) {
      throw std::runtime_error("nonfinite input");
    }
    check(vsi_nn_Float32ToDtype(data[i], result.data() + 2 * i, &d), "Float32ToDtype");
  }
  return result;
}

static vsi_nn_tensor_id_t tensor(specferry::np101::Graph &graph,
                                 std::initializer_list<std::uint32_t> shape, bool constant = false,
                                 const std::vector<uint8_t> &data = {}) {
  if (shape.size()) {
    return specferry::np101::add_tensor(graph, {specferry::np101::DataType::Float16, shape},
                                        constant, data);
  }
  // The demo's virtual intermediate storage is chosen internally by the SDK.
  vsi_nn_tensor_attr_t a{};
  a.dtype = dtype();
  a.vtl = true;
  a.dim_num = VSI_NN_DIM_AUTO;
  auto id = vsi_nn_AddTensor(graph.get(), VSI_NN_TENSOR_ID_AUTO, &a, nullptr);
  if (id == VSI_NN_TENSOR_ID_NA || !vsi_nn_GetTensor(graph.get(), id)) {
    throw std::runtime_error("vsi_nn_AddTensor failed (FP16)");
  }
  return id;
}

static std::vector<float> golden(const std::vector<float> &x, const std::vector<float> &w,
                                 const std::vector<float> &bias) {
  // FP32 convolution with bias, followed by ReLU.
  std::array<float, 256> conv{};
  for (int oc = 0; oc < 4; ++oc) {
    for (int y = 0; y < 8; ++y) {
      for (int z = 0; z < 8; ++z) {
        float v = bias[oc];
        for (int ic = 0; ic < 3; ++ic) {
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              int iy = y + ky - 1, ix = z + kx - 1;
              if (iy >= 0 && iy < 8 && ix >= 0 && ix < 8) {
                v += x[(ic * 8 + iy) * 8 + ix] * w[((oc * 3 + ic) * 3 + ky) * 3 + kx];
              }
            }
          }
        }
        conv[(oc * 8 + y) * 8 + z] = std::max(0.f, v);
      }
    }
  }

  // 2x2 max pooling produces the expected NCHW output.
  std::vector<float> out(64);
  for (int c = 0; c < 4; ++c) {
    for (int y = 0; y < 4; ++y) {
      for (int z = 0; z < 4; ++z) {
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            out[(c * 4 + y) * 4 + z] =
                std::max(out[(c * 4 + y) * 4 + z], conv[(c * 8 + y * 2 + dy) * 8 + z * 2 + dx]);
          }
        }
      }
    }
  }
  return out;
}

static ConvolutionData make_test_data() {
  std::srand(42);
  auto random_data = [](size_t n) {
    std::vector<float> values(n);
    for (auto &value : values) {
      value = (static_cast<float>(std::rand() % 2000) - 1000.f) / 1000.f;
    }
    return values;
  };

  ConvolutionData data;
  data.input = random_data(192);
  data.weights = random_data(108);
  data.bias = random_data(4);
  data.input_fp16 = f16(data.input);
  data.weights_fp16 = f16(data.weights);
  data.bias_fp16 = f16(data.bias);
  return data;
}

static GraphIO build_convolution_graph(specferry::np101::Graph &owner, ConvolutionData &data) {
  // Allocate input, constant weights, intermediate tensors, and output.
  auto input = tensor(owner, {8, 8, 3, 1});
  auto weight = tensor(owner, {3, 3, 3, 4}, true, data.weights_fp16);
  auto bias = tensor(owner, {4}, true, data.bias_fp16);
  auto conv_output = tensor(owner, {});
  auto relu_output = tensor(owner, {});
  auto output = tensor(owner, {4, 4, 4, 1});
  auto *graph = owner.get();

  // Configure the demo's CONV2D -> RELU -> POOL operators.
  auto *conv = vsi_nn_AddNode(graph, VSI_NN_OP_CONV2D, 3, 1, nullptr);
  auto *relu = vsi_nn_AddNode(graph, VSI_NN_OP_RELU, 1, 1, nullptr);
  auto *pool = vsi_nn_AddNode(graph, VSI_NN_OP_POOL, 1, 1, nullptr);
  if (!conv || !relu || !pool) {
    throw std::runtime_error("vsi_nn_AddNode returned null");
  }

  auto &conv_params = conv->nn_param.conv2d;
  conv_params.ksize[0] = conv_params.ksize[1] = 3;
  conv_params.stride[0] = conv_params.stride[1] = 1;
  conv_params.dilation[0] = conv_params.dilation[1] = 1;
  for (auto &pad : conv_params.pad) {
    pad = 1;
  }
  conv_params.pad_type = VSI_NN_PAD_AUTO;
  conv_params.weights = 4;
  conv_params.group = 1;
  conv_params.multiplier = 0;

  auto &pool_params = pool->nn_param.pool;
  pool_params.type = VX_NN_POOLING_MAX;
  pool_params.round_type = VSI_NN_ROUND_FLOOR;
  pool_params.ksize[0] = pool_params.ksize[1] = 2;
  pool_params.stride[0] = pool_params.stride[1] = 2;
  for (auto &pad : pool_params.pad) {
    pad = 0;
  }
  pool_params.pad_type = VSI_NN_PAD_VALID; // Keep SDK-owned pool_params.local intact.

  // Connect the operator chain and expose graph IO.
  conv->input.tensors[0] = input;
  conv->input.tensors[1] = weight;
  conv->input.tensors[2] = bias;
  conv->output.tensors[0] = conv_output;
  relu->input.tensors[0] = conv_output;
  relu->output.tensors[0] = relu_output;
  pool->input.tensors[0] = relu_output;
  pool->output.tensors[0] = output;
  if (!vsi_nn_SetGraphInputs(graph, &input, 1) || !vsi_nn_SetGraphOutputs(graph, &output, 1)) {
    throw std::runtime_error("setting graph IO failed");
  }

  check(vsi_nn_CopyDataToTensor(graph, vsi_nn_GetTensor(graph, input), data.input_fp16.data()),
        "CopyDataToTensor");
  return {input, output};
}

int main() {
  try {
    specferry::np101::SdkTimings timings(".");
    auto data = make_test_data();
    specferry::np101::Context context;
    specferry::np101::Graph graph(context, 6, 3);
    const auto io = build_convolution_graph(graph, data);
    check(specferry::np101::sdk_call("vsi_nn_SetupGraph",
                                     [&] { return vsi_nn_SetupGraph(graph.get(), FALSE); }),
          "SetupGraph");
    check(specferry::np101::sdk_call("vsi_nn_VerifyGraph",
                                     [&] { return vsi_nn_VerifyGraph(graph.get()); }),
          "VerifyGraph");
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
      auto input = data.input;
      for (auto &value : input) {
        value += iteration * .03125f;
      }
      const auto expected = golden(input, data.weights, data.bias);
      specferry::np101::upload_tensor(graph, io.input, f16(input));
      check(specferry::np101::sdk_call("vsi_nn_RunGraph",
                                       [&] { return vsi_nn_RunGraph(graph.get()); }),
            "RunGraph");
      std::unique_ptr<float, decltype(&std::free)> actual(
          vsi_nn_ConvertTensorToFloat32Data(graph.get(), vsi_nn_GetTensor(graph.get(), io.output)),
          &std::free);
      if (!actual) {
        throw std::runtime_error("null convolution readback");
      }
      float maximum = 0;
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto error = std::abs(actual.get()[i] - expected[i]);
        if (!std::isfinite(actual.get()[i]) || error > .1f) {
          throw std::runtime_error("convolution mismatch at output " + std::to_string(i));
        }
        maximum = std::max(maximum, error);
      }
      std::cout << "PASS iteration=" << iteration << " max_abs_error=" << maximum << std::endl;
    }
    graph.close();
    context.close();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
