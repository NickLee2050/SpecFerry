#include "binary_io.hpp"
#include "models/opt/config.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weight_bank.hpp"
#include "np101/weights.hpp"
#include "sys/resource.h"
#include "sys/time.h"
#include "test_support.hpp"
#include "vsi_nn_pub.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;

constexpr unsigned width = 8;
constexpr auto f16 = DataType::Float16;
namespace fs = std::filesystem;
namespace opt = specferry::models::opt;
using Clock = std::chrono::steady_clock;

struct CpuSnapshot {
  rusage process{};
  rusage main_thread{};
};

CpuSnapshot read_cpu_snapshot() {
  CpuSnapshot snapshot;
  if (getrusage(RUSAGE_SELF, &snapshot.process) ||
      getrusage(RUSAGE_THREAD, &snapshot.main_thread)) {
    throw std::runtime_error("cannot read FFN collection CPU usage");
  }
  return snapshot;
}

double elapsed_cpu_seconds(const timeval &before, const timeval &after) {
  return static_cast<double>(after.tv_sec - before.tv_sec) +
         static_cast<double>(after.tv_usec - before.tv_usec) * 1e-6;
}

struct FfnFixture {
  opt::Config config;
  WeightStore weights;
  std::vector<std::vector<std::uint8_t>> inputs;
  unsigned input_rows;

  FfnFixture(const fs::path &deployment, const fs::path &fixture, bool full, unsigned rows)
      : config(opt::read_config(fixture / "components.txt")), weights(deployment),
        input_rows(rows) {
    if (config.hidden != 1024 || config.intermediate != 4096 || config.heads != 16 ||
        config.layers != 24 || config.epsilon != 1e-5f) {
      throw std::invalid_argument("FFN collection requires OPT-350m 1024/4096/16/24/epsilon=1e-5");
    }
    for (unsigned layer = 0; layer < config.layers; ++layer) {
      const auto require_weight = [&](const char *name, const Shape &shape) {
        const auto &record = weights.find(config.prefix(layer) + name);
        if (record.spec.type != f16 || record.spec.shape != shape) {
          throw std::invalid_argument("FFN weight contract mismatch: " + record.name);
        }
      };
      require_weight("fc1.weight", {config.hidden, config.intermediate});
      require_weight("fc1.bias", {config.intermediate});
      require_weight("fc2.weight", {config.intermediate, config.hidden});
      require_weight("fc2.bias", {config.hidden});
      if (full) {
        require_weight("final_layer_norm.weight", {config.hidden});
        require_weight("final_layer_norm.bias", {config.hidden});
      }
      auto input =
          read_bytes(fixture, "teacher.0.layer." + std::to_string(layer) + ".attention_norm.bin",
                     std::size_t(config.hidden) * input_rows * 2);
      for (std::size_t offset = 0; offset < input.size(); offset += 2) {
        const auto bits = unsigned(input[offset]) | (unsigned(input[offset + 1]) << 8);
        if ((bits & 0x7c00) == 0x7c00) {
          throw std::invalid_argument("FFN fixture contains nonfinite FP16 input");
        }
      }
      inputs.push_back(std::move(input));
    }
  }
};

struct FfnGraphs {
  std::vector<std::unique_ptr<GraphBuilder>> graphs;

  ~FfnGraphs() {
    while (!graphs.empty()) {
      graphs.pop_back();
    }
  }

  void close() {
    while (!graphs.empty()) {
      graphs.back()->graph.close();
      graphs.pop_back();
    }
  }
};

void check_ffn_collection(const fs::path &output, const FfnFixture &fixture,
                          const std::string &mode) {
  const bool full = mode == "ffn-full";
  const auto fc1_parts = mode == "ffn-linear-fc1-split2" ? 2u : 1u;
  const auto started = Clock::now();
  Context context;
  WeightBank bank(context);
  FfnGraphs owners;
  Tensor last_output{};
  double initialization_seconds = 0, release_seconds = 0;
  const auto report = [&](const std::string &phase, bool released = false) {
    std::ostringstream record;
    record << std::setprecision(17) << "{\"phase\":\"" << phase << "\",\"status\":\""
           << (released ? "executed" : "running")
           << "\",\"released\":" << (released ? "true" : "false") << ",\"mode\":\"" << mode
           << "\",\"graph_count\":24,\"weight_policy\":\"shared-mutable\""
           << ",\"weight_sets\":" << fixture.config.layers
           << ",\"input_rows\":" << fixture.input_rows
           << ",\"initialization_seconds\":" << initialization_seconds
           << ",\"release_seconds\":" << release_seconds
           << ",\"weight_payload_bytes\":" << bank.payload_bytes() << ",\"matrix_payload_bytes\":"
           << std::size_t(fixture.config.hidden) * fixture.config.intermediate * 4 *
                  fixture.config.layers
           << ",\"application_segment_limit_bytes\":" << segment_payload_limit;
    for (const bool constant : {true, false}) {
      const auto name = constant ? "const" : "nonconst";
      record << ",\"" << name << "_live_bytes\":" << context.memory().live(constant) << ",\""
             << name << "_peak_bytes\":" << context.memory().peak(constant);
    }
    record << "}\n";
    write_text(output / "execution.json", record.str());
  };
  for (unsigned layer = 0; layer < fixture.config.layers; ++layer) {
    report("initialize.layer." + std::to_string(layer));
    auto owner = std::make_unique<GraphBuilder>(context, 128, 96, &bank);
    auto &graph = *owner;
    const auto input = graph.tensor({f16, {fixture.config.hidden, fixture.input_rows}});
    upload_tensor(graph.graph, input.id, fixture.inputs[layer]);
    const auto prefix = fixture.config.prefix(layer);
    const auto linear = [&](Tensor value, const char *name) {
      const auto &matrix = fixture.weights.find(prefix + name + ".weight");
      const auto &bias = fixture.weights.find(prefix + name + ".bias");
      return graph.linear(value, fixture.weights, matrix, bias,
                          std::string(name) == "fc1" ? fc1_parts : 1u);
    };
    auto value = linear(input, "fc1");
    value = graph.unary(VSI_NN_OP_RELU, value, f16);
    value = linear(value, "fc2");
    if (full) {
      value = graph.binary(VSI_NN_OP_ADD, input, value);
      const auto affine = [&](const char *name) {
        return graph.weight(fixture.weights, fixture.weights.find(prefix + name),
                            {f16, {fixture.config.hidden}});
      };
      const auto scale = affine("final_layer_norm.weight");
      const auto bias = affine("final_layer_norm.bias");
      value = graph.layer_norm(value, scale, bias, fixture.config.epsilon);
    }
    graph.compile({input}, {value});
    last_output = value;
    owners.graphs.push_back(std::move(owner));
  }
  initialization_seconds = std::chrono::duration<double>(Clock::now() - started).count();
  std::ofstream profile(output / "profile.tsv");
  profile << "phase\tbatch\tsweeps\tgraph_calls\tseconds\tprocess_user_seconds"
             "\tprocess_system_seconds\tmain_thread_user_seconds\tmain_thread_system_seconds\n"
          << std::setprecision(17);
  const auto run_batch = [&](const char *phase, unsigned batch, unsigned sweeps) {
    report(std::string(phase) + "." + std::to_string(batch));
    // Timing covers the same graph sequence without per-call instrumentation or explicit IO.
    const auto cpu_before = read_cpu_snapshot();
    const auto begin = Clock::now();
    for (unsigned sweep = 0; sweep < sweeps; ++sweep) {
      for (const auto &owner : owners.graphs) {
        check(vsi_nn_RunGraph(owner->graph.get()), "FFN collection RunGraph");
      }
    }
    const auto elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
    const auto cpu_after = read_cpu_snapshot();
    profile << phase << '\t' << batch << '\t' << sweeps << '\t' << sweeps * 24 << '\t' << elapsed
            << '\t' << elapsed_cpu_seconds(cpu_before.process.ru_utime, cpu_after.process.ru_utime)
            << '\t' << elapsed_cpu_seconds(cpu_before.process.ru_stime, cpu_after.process.ru_stime)
            << '\t'
            << elapsed_cpu_seconds(cpu_before.main_thread.ru_utime, cpu_after.main_thread.ru_utime)
            << '\t'
            << elapsed_cpu_seconds(cpu_before.main_thread.ru_stime, cpu_after.main_thread.ru_stime)
            << '\n';
    profile.flush();
    if (!profile) {
      throw std::runtime_error("cannot write FFN batch timing");
    }
  };
  run_batch("warmup", 0, 4);
  for (unsigned batch = 0; batch < 3; ++batch) {
    run_batch("measured", batch, 8);
  }
  report("readback");
  write_bytes(output / "last-layer.bin",
              read_tensor(owners.graphs.back()->graph, last_output.id, TensorReadback::MappedHost));
  report("release");
  const auto release_started = Clock::now();
  owners.close();
  bank.close();
  context.close();
  release_seconds = std::chrono::duration<double>(Clock::now() - release_started).count();
  if (context.memory().live(true) || context.memory().live(false)) {
    throw std::runtime_error("FFN collection retained application tensor payload after release");
  }
  report("complete", true);
}

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

void compile_projection(GraphBuilder &graph, Tensor input, Tensor output) {
  const std::vector<vsi_nn_op_t> expected =
      std::vector<vsi_nn_op_t>{VSI_NN_OP_MATRIXMUL, VSI_NN_OP_RESHAPE, VSI_NN_OP_ADD};
  if (graph.graph.get()->cur_nid != expected.size()) {
    throw std::runtime_error("projection node count differs from the requested mode");
  }
  std::cout << "Graph: " << "MATRIXMUL -> RESHAPE(bias) -> ADD"
            << "; FP16 input [8,1], identity weight [8,8], bias [8]=0.25\n";
  for (unsigned index = 0; index < expected.size(); ++index) {
    auto *operation = vsi_nn_GetNode(graph.graph.get(), index);
    if (!operation || operation->op != expected[index]) {
      throw std::runtime_error("projection operator differs from the requested mode");
    }
    std::cout << "node " << index << " confirmed, op=" << operation->op << '\n';
  }

  auto input_id = input.id, output_id = output.id;
  if (!vsi_nn_SetGraphInputs(graph.graph.get(), &input_id, 1) ||
      !vsi_nn_SetGraphOutputs(graph.graph.get(), &output_id, 1)) {
    throw std::runtime_error("cannot declare projection inputs/outputs");
  }
  check(sdk_call("vsi_nn_SetupGraph", [&] { return vsi_nn_SetupGraph(graph.graph.get(), FALSE); }),
        "SetupGraph");
  for (unsigned index = 0; index < graph.graph.get()->cur_tid; ++index) {
    auto *tensor = vsi_nn_GetTensor(graph.graph.get(), index);
    if (!tensor || !tensor->t) {
      throw std::runtime_error("null projection tensor before VerifyGraph");
    }
    check(vxGetStatus(reinterpret_cast<vx_reference>(tensor->t)), "projection tensor status");
  }
  std::cout << "All tensor references valid; entering VerifyGraph" << std::endl;
  check(sdk_call("vsi_nn_VerifyGraph", [&] { return vsi_nn_VerifyGraph(graph.graph.get()); }),
        "VerifyGraph");
}

void check_projection(Context &context) {
  GraphBuilder graph(context, 6, 3);
  auto input = graph.tensor({f16, {width, 1}});
  auto output = build_projection(graph, input);
  // Initialize valid FP16 inputs before setup/verify.
  upload_tensor(graph.graph, input.id, encode_floats(std::vector<float>(width, 1)));
  compile_projection(graph, input, output);

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
  const bool batched = mode == "ffn-linear-batched";
  const bool collection =
      mode == "ffn-full" || mode == "ffn-linear" || mode == "ffn-linear-fc1-split2" || batched;
  if ((collection && argc != (batched ? 6 : 5)) ||
      (!collection && (argc != 3 || (mode != "matmul-add" && mode != "mapped-readback")))) {
    std::cerr << "usage: np101_projection_check OUTPUT matmul-add|mapped-readback\n"
                 "   or: np101_projection_check OUTPUT "
                 "ffn-full|ffn-linear|ffn-linear-fc1-split2 DEPLOYMENT FIXTURE\n"
                 "   or: np101_projection_check OUTPUT ffn-linear-batched DEPLOYMENT FIXTURE "
                 "4|8|16\n";
    return 2;
  }
  if (collection) {
    try {
      unsigned rows = 1;
      if (batched) {
        const std::string value = argv[5];
        if (value != "4" && value != "8" && value != "16") {
          throw std::invalid_argument("batched FFN input rows must be 4, 8 or 16");
        }
        rows = static_cast<unsigned>(std::stoul(value));
      }
      const FfnFixture fixture(argv[3], argv[4], mode == "ffn-full", rows);
      return run_test(argv[1], [&] { check_ffn_collection(argv[1], fixture, mode); });
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 2;
    }
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
