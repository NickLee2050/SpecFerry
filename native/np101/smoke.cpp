#include "np101/context.hpp"

#include "gc_hal.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using specferry::np101::check;
using Clock = std::chrono::steady_clock;

struct Options {
  std::string output;
  unsigned repeats = 10;
};

struct DeviceInfo {
  std::string target;
  std::array<char, VX_MAX_IMPLEMENTATION_NAME> implementation{};
  vx_status implementation_status{};
  gceSTATUS memory_status{};
  gctSIZE_T internal_bytes = 0;
  gctSIZE_T external_bytes = 0;
  gctSIZE_T contiguous_bytes = 0;
};

struct SmokeData {
  std::vector<float> input, weights, bias;
  // Keep tensor initialization buffers alive until the graph is released.
  std::vector<uint8_t> input_fp16, weights_fp16, bias_fp16;
};

struct GraphIO {
  vsi_nn_tensor_id_t input;
  vsi_nn_tensor_id_t output;
};

struct PhaseTimings {
  double initialize_ms = 0;
  double setup_ms = 0;
  double verify_ms = 0;
  double release_ms = 0;
};

static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      out << '\\' << c;
    } else if (c < 32) {
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
    } else {
      out << c;
    }
  }
  out << '"';
  return out.str();
}

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

static vsi_nn_tensor_id_t tensor(vsi_nn_graph_t *g, std::initializer_list<size_t> shape,
                                 bool constant = false, uint8_t *data = nullptr) {
  vsi_nn_tensor_attr_t a{};
  a.dtype = dtype();
  a.is_const = constant;
  a.vtl = shape.size() == 0;
  a.dim_num = a.vtl ? VSI_NN_DIM_AUTO : shape.size();
  size_t i = 0;
  for (auto dim : shape) {
    a.size[i++] = dim;
  }
  auto id = vsi_nn_AddTensor(g, VSI_NN_TENSOR_ID_AUTO, &a, data);
  if (id == VSI_NN_TENSOR_ID_NA || !vsi_nn_GetTensor(g, id)) {
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

static void parse_options(int argc, char **argv, Options &options) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc) {
      options.output = argv[++i];
    } else if (arg == "--repeats" && i + 1 < argc) {
      std::string value = argv[++i];
      size_t used = 0;
      unsigned long n = std::stoul(value, &used);
      if (used != value.size() || n < 1 || n > 1000) {
        throw std::runtime_error("invalid repeats");
      }
      options.repeats = n;
    } else {
      throw std::runtime_error("usage: np101_smoke --output report.json [--repeats 10]");
    }
  }
  if (options.output.empty()) {
    throw std::runtime_error("--output is required");
  }
}

static void write_running_report(const std::string &output) {
  std::ofstream report(output);
  if (!report) {
    throw std::runtime_error("cannot write report: " + output);
  }
  report << "{\"status\":\"running\"}\n";
}

static DeviceInfo query_device_info(const specferry::np101::Context &context) {
  DeviceInfo info;
  info.target.assign(context.get()->config.target_name,
                     strnlen(context.get()->config.target_name, VSI_NN_MAX_TARGET_NAME));
  info.implementation_status =
      vxQueryContext(context.get()->c, VX_CONTEXT_IMPLEMENTATION, info.implementation.data(),
                     info.implementation.size());

  gctUINT32 internal_name = 0, external_name = 0, contiguous_name = 0;
  info.memory_status =
      gcoHAL_QueryVideoMemory(nullptr, &internal_name, &info.internal_bytes, &external_name,
                              &info.external_bytes, &contiguous_name, &info.contiguous_bytes);
  return info;
}

static SmokeData make_test_data() {
  std::srand(42);
  auto random_data = [](size_t n) {
    std::vector<float> values(n);
    for (auto &value : values) {
      value = (static_cast<float>(std::rand() % 2000) - 1000.f) / 1000.f;
    }
    return values;
  };

  SmokeData data;
  data.input = random_data(192);
  data.weights = random_data(108);
  data.bias = random_data(4);
  data.input_fp16 = f16(data.input);
  data.weights_fp16 = f16(data.weights);
  data.bias_fp16 = f16(data.bias);
  return data;
}

static GraphIO build_smoke_graph(vsi_nn_graph_t *graph, SmokeData &data) {
  // Allocate input, constant weights, intermediate tensors, and output.
  auto input = tensor(graph, {8, 8, 3, 1});
  auto weight = tensor(graph, {3, 3, 3, 4}, true, data.weights_fp16.data());
  auto bias = tensor(graph, {4}, true, data.bias_fp16.data());
  auto conv_output = tensor(graph, {});
  auto relu_output = tensor(graph, {});
  auto output = tensor(graph, {4, 4, 4, 1});

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

static void run_iterations(vsi_nn_graph_t *graph, const GraphIO &io, const SmokeData &data,
                           unsigned repeats, std::string &phase, std::ostringstream &records) {
  records << std::setprecision(10);
  for (unsigned iteration = 0; iteration < repeats; ++iteration) {
    // Vary input while reusing the graph and weights to detect stale output.
    auto current = data.input;
    for (auto &value : current) {
      value += static_cast<float>(iteration) * .03125f;
    }
    auto expected = golden(current, data.weights, data.bias);
    auto input_bytes = f16(current);

    // Measure input upload, SDK execution, and output readback separately.
    auto start = Clock::now();
    check(vsi_nn_CopyDataToTensor(graph, vsi_nn_GetTensor(graph, io.input), input_bytes.data()),
          "CopyDataToTensor(iteration)");
    double upload_ms = elapsed(start);

    phase = "run[" + std::to_string(iteration) + "]";
    start = Clock::now();
    check(vsi_nn_RunGraph(graph), "RunGraph");
    double run_ms = elapsed(start);

    start = Clock::now();
    std::unique_ptr<float, decltype(&std::free)> values(
        vsi_nn_ConvertTensorToFloat32Data(graph, vsi_nn_GetTensor(graph, io.output)), &std::free);
    if (!values) {
      throw std::runtime_error("readback returned null");
    }
    double readback_ms = elapsed(start);

    // Compare every output with the FP32 CPU reference before recording success.
    double max_error = 0, output_sum = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (!std::isfinite(values.get()[i]) || !std::isfinite(expected[i])) {
        throw std::runtime_error("nonfinite output at " + std::to_string(i));
      }
      double error = std::abs(values.get()[i] - expected[i]);
      output_sum += values.get()[i];
      max_error = std::max(max_error, error);
      if (error > .1) {
        throw std::runtime_error("FP16 comparison failed at " + std::to_string(i));
      }
    }

    if (iteration) {
      records << ',';
    }
    records << "{\"iteration\":" << iteration << ",\"run_ms\":" << run_ms
            << ",\"input_upload_ms\":" << upload_ms << ",\"readback_ms\":" << readback_ms
            << ",\"output_sum\":" << output_sum << ",\"max_abs_error\":" << max_error << '}';
    std::cout << "iteration=" << iteration << " run_ms=" << run_ms << " max_abs_error=" << max_error
              << '\n';
  }
}

static void write_success_report(const std::string &output, const DeviceInfo &device,
                                 const PhaseTimings &timings, const std::string &records) {
  std::ofstream report(output);
  report << "{\"status\":\"numerical_pass\",\"hardware_execution_proven\":false,"
         << "\"evidence_note\":\"Correlate this run with driver trace; SDK target alone is "
            "insufficient.\","
         << "\"graph\":\"CONV2D-RELU-POOL\",\"dtype\":\"FP16\",\"output_shape\":[1,4,4,4],"
         << "\"atol\":0.1,\"context_target\":" << quote(device.target)
         << ",\"implementation_query_status\":" << device.implementation_status
         << ",\"implementation\":" << quote(device.implementation.data())
         << ",\"memory_query\":{\"status\":" << device.memory_status
         << ",\"internal_bytes\":" << device.internal_bytes
         << ",\"external_bytes\":" << device.external_bytes
         << ",\"contiguous_bytes\":" << device.contiguous_bytes
         << ",\"meaning\":\"HAL pool sizes; not measured free model memory\"},"
         << "\"initialize_ms\":" << timings.initialize_ms << ",\"setup_ms\":" << timings.setup_ms
         << ",\"verify_ms\":" << timings.verify_ms << ",\"release_ms\":" << timings.release_ms
         << ",\"handles_cleared\":true,\"iterations\":[" << records << "]}\n";
  if (!report) {
    throw std::runtime_error("report write failed");
  }
}

int main(int argc, char **argv) {
  std::cout << std::unitbuf;
  Options options;
  PhaseTimings timings;
  std::string phase = "arguments";
  std::ostringstream records;

  try {
    // Validate arguments and create the report before opening the device.
    parse_options(argc, argv, options);
    write_running_report(options.output);

    // Initialize the SDK, capture device information, and construct the graph.
    phase = "initialize";
    auto start = Clock::now();
    specferry::np101::Context context;
    auto device = query_device_info(context);
    auto data = make_test_data();
    specferry::np101::Graph graph(context, 6, 3);
    auto io = build_smoke_graph(graph.get(), data);
    timings.initialize_ms = elapsed(start);

    // Prepare and verify once; every iteration reuses this graph.
    phase = "setup";
    start = Clock::now();
    check(vsi_nn_SetupGraph(graph.get(), FALSE), "SetupGraph");
    timings.setup_ms = elapsed(start);

    phase = "verify";
    start = Clock::now();
    check(vsi_nn_VerifyGraph(graph.get()), "VerifyGraph");
    timings.verify_ms = elapsed(start);

    // Run varying inputs, compare SDK outputs with CPU references, and collect timings.
    run_iterations(graph.get(), io, data, options.repeats, phase, records);

    // Release the graph before its context, then publish the completed report.
    phase = "release";
    start = Clock::now();
    graph.close();
    context.close();
    timings.release_ms = elapsed(start);
    write_success_report(options.output, device, timings, records.str());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL phase=" << phase << ": " << error.what() << '\n';
    if (!options.output.empty()) {
      std::ofstream failed(options.output);
      failed << "{\"status\":\"failed\",\"phase\":" << quote(phase)
             << ",\"error\":" << quote(error.what()) << ",\"iterations\":[" << records.str()
             << "]}\n";
    }
    return 1;
  }
}
