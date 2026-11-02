#include "binary_io.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "vsi_nn_dtype_util.h"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Port {
  std::string name;
  TensorSpec spec;
  double atol = 0, rtol = 0;
};

struct Case {
  std::vector<std::vector<std::uint8_t>> inputs, expected;
};

struct Fixture {
  CompiledGraphSpec graph;
  std::vector<Port> inputs, outputs;
  std::vector<Case> cases;
};

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

Fixture load_fixture(const fs::path &directory) {
  Fixture fixture;
  auto &spec = fixture.graph;
  std::ifstream contract(directory / "contract.txt");
  unsigned cases = 0, inputs = 0, outputs = 0;
  contract >> spec.graph_version[0] >> spec.graph_version[1] >> spec.graph_version[2] >>
      spec.constant_bytes >> cases >> inputs >> outputs;
  if (!contract || !spec.graph_version[0] || cases < 1 || cases > 3 || inputs < 1 || inputs > 5 ||
      outputs < 1 || outputs > 3 || !spec.constant_bytes ||
      spec.constant_bytes > 32 * 1024 * 1024) {
    throw std::invalid_argument("invalid bounded compiled graph contract");
  }
  for (const bool output : {false, true}) {
    std::set<std::string> names;
    for (unsigned index = 0; index < (output ? outputs : inputs); ++index) {
      Port port;
      std::string dtype;
      unsigned rank = 0;
      contract >> port.name >> dtype >> rank;
      if (!contract || rank < 1 || rank > 3 || port.name.empty() ||
          port.name.find_first_not_of(
              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.") !=
              std::string::npos ||
          !names.insert(port.name).second) {
        throw std::invalid_argument("invalid or duplicate compiled graph port");
      }
      port.spec.type = parse_dtype(dtype);
      port.spec.shape.resize(rank);
      for (auto &axis : port.spec.shape) {
        contract >> axis;
      }
      if (!contract || port.spec.bytes() > 1024 * 1024 ||
          (port.spec.type != DataType::Float16 && port.spec.type != DataType::Float32 &&
           (output || port.spec.type != DataType::Int32))) {
        throw std::invalid_argument("invalid bounded compiled graph shape or dtype");
      }
      if (output) {
        contract >> port.atol >> port.rtol;
        if (!contract || !std::isfinite(port.atol) || !std::isfinite(port.rtol) || port.atol < 0 ||
            port.rtol < 0) {
          throw std::invalid_argument("invalid output tolerance");
        }
        fixture.outputs.push_back(port);
        spec.outputs.push_back(port.spec);
      } else {
        fixture.inputs.push_back(port);
        spec.inputs.push_back(port.spec);
      }
    }
  }
  std::string extra;
  if (!contract || contract >> extra) {
    throw std::invalid_argument("invalid compiled graph contract suffix");
  }
  spec.path = (directory / "model.nb").string();
  if (!fs::is_regular_file(spec.path) || !fs::file_size(spec.path) ||
      fs::file_size(spec.path) > 32 * 1024 * 1024) {
    throw std::invalid_argument("invalid bounded NBG file");
  }
  fixture.cases.resize(cases);
  for (unsigned index = 0; index < cases; ++index) {
    const auto path = directory / ("case" + std::to_string(index));
    for (const auto &port : fixture.inputs) {
      fixture.cases[index].inputs.push_back(
          read_bytes(path, "input." + port.name + ".bin", port.spec.bytes()));
    }
    for (const auto &port : fixture.outputs) {
      fixture.cases[index].expected.push_back(
          read_bytes(path, "expected." + port.name + ".bin", port.spec.bytes()));
    }
  }
  return fixture;
}

float floating(const std::uint8_t *bytes, DataType type) {
  float value;
  if (type == DataType::Float32) {
    std::memcpy(&value, bytes, sizeof(value));
  } else {
    vsi_nn_dtype_t dtype{};
    dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    check(vsi_nn_DtypeToFloat32(const_cast<std::uint8_t *>(bytes), &value, &dtype), "decode FP16");
  }
  return value;
}
} // namespace

int main(int argc, char **argv) {
  if ((argc != 3 && argc != 4) || (argc == 4 && std::string(argv[3]) != "12")) {
    std::cerr << "usage: np101_compiled_graph_check FIXTURE OUTPUT [12]\n";
    return 2;
  }
  try {
    // Resolve every bounded input/output contract before opening a device Context.
    const auto fixture = load_fixture(argv[1]);
    const auto count = static_cast<unsigned>(fixture.cases.size());
    const unsigned runs = argc == 4 ? 12 : count;
    if (argc == 4 && count != 3) {
      throw std::invalid_argument("alternating control requires three existing input cases");
    }
    const fs::path output(argv[2]);
    fs::create_directories(output);
    std::cout << std::unitbuf;
    SdkTimings timings(output);
    std::ofstream trace(output / "rungraph-calls.tsv"), numerical(output / "numerical.tsv");
    trace << "event\tsequence\tcase\tmonotonic_seconds\tstatus\n" << std::setprecision(17);
    numerical << "run\tcase\toutput\telements\tmax_abs_error\tfailed\n";
    const auto phase = [](const std::string &name) { std::cout << "phase=" << name << '\n'; };
    phase("CreateContext");
    const auto preparation_start = Clock::now();
    Context context;
    GraphBuilder graph(context, fixture.inputs.size() + fixture.outputs.size(), 1);
    std::vector<Tensor> inputs;
    for (const auto &port : fixture.inputs) {
      inputs.push_back(graph.tensor(port.spec));
    }
    const auto outputs = graph.compiled(fixture.graph, inputs);
    double upload_seconds = 0, readback_seconds = 0, graph_seconds = 0;
    const auto upload = [&](unsigned index) {
      const auto start = Clock::now();
      for (unsigned port = 0; port < inputs.size(); ++port) {
        upload_tensor(graph.graph, inputs[port].id, fixture.cases[index].inputs[port]);
      }
      upload_seconds += elapsed(start);
    };
    upload(0);
    phase("SetupVerify");
    graph.compile(inputs, outputs);
    const auto preparation_seconds = elapsed(preparation_start);
    bool passed = true;
    unsigned completed = 0;
    for (unsigned run = 0; run < runs && passed; ++run) {
      const auto index = run % count;
      const SdkScope request("request", "run." + std::to_string(run));
      phase("run." + std::to_string(run) + ".upload");
      upload(index);
      const auto record = [&](const char *event, const std::string &status) {
        trace << event << '\t' << run + 1 << '\t' << index << '\t'
              << std::chrono::duration<double>(Clock::now().time_since_epoch()).count() << '\t'
              << status << std::endl;
        if (!trace) {
          throw std::runtime_error("cannot save RunGraph interval");
        }
      };
      phase("run." + std::to_string(run) + ".RunGraph");
      record("begin", "pending");
      const auto start = Clock::now();
      const auto status = vsi_nn_RunGraph(graph.graph.get());
      graph_seconds += elapsed(start);
      record("end", std::to_string(status));
      check(status, "RunGraph");
      phase("run." + std::to_string(run) + ".readback");
      // Retain the successful fixed-binding upload/Run/map-unmap protocol. Never append
      // an unchanged-input/no-readback replay collection to this compatibility control.
      for (unsigned port_index = 0; port_index < outputs.size(); ++port_index) {
        const auto read_start = Clock::now();
        const auto actual =
            read_tensor(graph.graph, outputs[port_index].id, TensorReadback::MappedHost);
        readback_seconds += elapsed(read_start);
        const auto &port = fixture.outputs[port_index];
        const auto &expected = fixture.cases[index].expected[port_index];
        if (actual.size() != expected.size()) {
          throw std::runtime_error("compiled graph readback byte count differs");
        }
        const auto width = dtype_bytes(port.spec.type);
        double max_error = 0;
        std::size_t failed = 0;
        for (std::size_t offset = 0; offset < actual.size(); offset += width) {
          const auto value = floating(actual.data() + offset, port.spec.type);
          const auto reference = floating(expected.data() + offset, port.spec.type);
          const auto error = std::abs(double(value) - reference);
          max_error = std::max(max_error, error);
          failed += !std::isfinite(value) || !std::isfinite(reference) ||
                    error > port.atol + port.rtol * std::abs(reference);
        }
        numerical << run << '\t' << index << '\t' << port.name << '\t' << port.spec.elements()
                  << '\t' << std::setprecision(12) << max_error << '\t' << failed << std::endl;
        if (!numerical) {
          throw std::runtime_error("cannot save numerical result");
        }
        // Keep each case's final outputs and any failure, without writing every replay.
        if (failed || run + count >= runs) {
          write_bytes(output / ("case" + std::to_string(index) + "." + port.name + ".bin"), actual);
        }
        passed = passed && failed == 0;
      }
      ++completed;
    }
    const auto const_peak = context.memory().peak(true),
               mutable_peak = context.memory().peak(false);
    phase("ReleaseGraph_ReleaseContext");
    const auto release_start = Clock::now();
    graph.graph.close();
    context.close();
    const auto release_seconds = elapsed(release_start);
    timings.flush();
    if (context.memory().live(true) || context.memory().live(false)) {
      throw std::runtime_error("application payload remained after release");
    }
    std::ofstream report(output / "execution.json");
    report << std::setprecision(12) << "{\"numerical_passed\":" << (passed ? "true" : "false")
           << ",\"released\":true,\"requested_runs\":" << runs
           << ",\"completed_runs\":" << completed << ",\"const_peak_bytes\":" << const_peak
           << ",\"nonconst_peak_bytes\":" << mutable_peak
           << ",\"const_live_bytes\":0,\"nonconst_live_bytes\":0,\"preparation_seconds\":"
           << preparation_seconds << ",\"upload_seconds\":" << upload_seconds
           << ",\"graph_seconds\":" << graph_seconds << ",\"readback_seconds\":" << readback_seconds
           << ",\"release_seconds\":" << release_seconds << "}\n";
    report.close();
    if (!report) {
      throw std::runtime_error("cannot save execution result");
    }
    phase(passed ? "complete" : "numerical_failure");
    return passed ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
