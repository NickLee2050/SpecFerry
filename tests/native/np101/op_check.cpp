#include "case_file.hpp"
#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "operators.hpp"
#include "unistd.h"
#include "vsi_nn_pub.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <map>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry;
using Clock = std::chrono::steady_clock;

std::string json_string(const std::string &text) {
  std::ostringstream output;
  output << '"';
  for (unsigned char character : text) {
    if (character == '"' || character == '\\') {
      output << '\\' << character;
    } else if (character < 32) {
      output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(character);
    } else {
      output << character;
    }
  }
  output << '"';
  return output.str();
}

struct Progress {
  struct PhaseTiming {
    std::size_t cycle;
    std::string phase;
    double milliseconds;
  };

  struct ResourceSample {
    std::size_t completed_cycles;
    std::size_t rss_bytes;
    std::size_t open_descriptors;
  };

  std::filesystem::path directory;
  std::string phase = "load_case";
  bool setup = false;
  bool verified = false;
  bool argmax_software = false;
  std::size_t completed_steps = 0;
  std::vector<double> run_times;
  std::vector<PhaseTiming> phase_times;
  std::vector<ResourceSample> resources;
  std::size_t cycles = 1;
  std::size_t cycle = 0;
  std::size_t completed_cycles = 0;
  std::size_t readback_count = 0;
  bool final_readback = false;
  std::size_t phase_cycle = 0;
  Clock::time_point phase_started = Clock::now();

  std::string output_name(const std::string &name, std::size_t step) const {
    auto prefix = cycles == 1 ? "" : "cycle." + std::to_string(cycle) + ".";
    return prefix + name + "." + std::to_string(step) + ".bin";
  }

  void observe_resources() {
    std::ifstream memory("/proc/self/statm");
    std::size_t virtual_pages = 0, resident_pages = 0;
    if (!(memory >> virtual_pages >> resident_pages)) {
      throw std::runtime_error("cannot read process memory counters");
    }
    memory.close();
    auto page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      throw std::runtime_error("cannot read system page size");
    }
    std::size_t descriptors = 0;
    for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fd")) {
      (void)entry;
      ++descriptors;
    }
    resources.push_back(
        {completed_cycles, resident_pages * static_cast<std::size_t>(page_size), descriptors});
  }

  void save(const std::string &status, const std::string &error = "") const {
    std::ofstream output(directory / "execution.json");
    output << std::boolalpha << "{\n  \"status\": " << json_string(status)
           << ",\n  \"phase\": " << json_string(phase) << ",\n  \"error\": " << json_string(error)
           << ",\n  \"setup_passed\": " << setup << ",\n  \"verify_passed\": " << verified
           << ",\n  \"completed_steps\": " << completed_steps
           << ",\n  \"cycles_requested\": " << cycles
           << ",\n  \"completed_cycles\": " << completed_cycles
           << ",\n  \"readback_mode\": " << json_string(final_readback ? "final" : "each-step")
           << ",\n  \"application_readback_count\": " << readback_count
           << ",\n  \"hardware_execution_proven\": false"
           << ",\n  \"argmax_execute_on_sw\": " << argmax_software << ",\n  \"run_ms\": [";
    for (std::size_t index = 0; index < run_times.size(); ++index) {
      if (index) {
        output << ", ";
      }
      output << run_times[index];
    }
    output << "],\n  \"phase_ms\": [";
    for (std::size_t index = 0; index < phase_times.size(); ++index) {
      const auto &timing = phase_times[index];
      if (index) {
        output << ',';
      }
      output << "{\"cycle\":" << timing.cycle << ",\"phase\":" << json_string(timing.phase)
             << ",\"milliseconds\":" << timing.milliseconds << '}';
    }
    output << "],\n  \"resource_samples\": [";
    for (std::size_t index = 0; index < resources.size(); ++index) {
      const auto &sample = resources[index];
      if (index) {
        output << ',';
      }
      output << "{\"completed_cycles\":" << sample.completed_cycles
             << ",\"host_rss_bytes\":" << sample.rss_bytes
             << ",\"open_descriptors\":" << sample.open_descriptors << '}';
    }
    output << "],\n  \"device_memory_measurement_available\": false\n}\n";
    if (!output) {
      throw std::runtime_error("failed to write execution report");
    }
  }

  void enter(const std::string &next) {
    auto now = Clock::now();
    phase_times.push_back({phase_cycle, phase,
                           std::chrono::duration<double, std::milli>(now - phase_started).count()});
    phase = next;
    phase_cycle = cycle;
    phase_started = now;
    save("running");
  }
};

const testing::TensorDefinition &tensor_definition(const testing::CaseDefinition &test,
                                                   const std::string &name) {
  for (const auto &tensor : test.tensors) {
    if (tensor.name == name) {
      return tensor;
    }
  }
  throw std::invalid_argument("undefined tensor: " + name);
}

void run_steps(np101::Graph &graph, const testing::CaseDefinition &test,
               const std::map<std::string, vsi_nn_tensor_id_t> &ids, Progress &progress) {
  for (std::size_t step = 0; step < test.steps; ++step) {
    progress.enter("upload_step_" + std::to_string(step));
    for (const auto &input : test.inputs) {
      const auto &tensor = tensor_definition(test, input.tensor);
      auto data = testing::read_bytes(test.root, input.files[step], tensor.spec.bytes());
      np101::upload_tensor(graph, ids.at(input.tensor), data);
    }

    progress.enter("run_step_" + std::to_string(step));
    auto start = Clock::now();
    np101::check(vsi_nn_RunGraph(graph.get()), "RunGraph");
    progress.run_times.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    ++progress.completed_steps;

    if (progress.final_readback && step + 1 != test.steps) {
      continue;
    }
    progress.enter("read_step_" + std::to_string(step));
    for (const auto &name : test.outputs) {
      auto bytes = np101::read_tensor(graph, ids.at(name));
      auto path = progress.directory / progress.output_name(name, step);
      std::ofstream output(path, std::ios::binary);
      output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      if (!output) {
        throw std::runtime_error("failed to write output: " + path.string());
      }
      ++progress.readback_count;
    }
  }
}

void execute(const testing::CaseDefinition &test, Progress &progress) {
  progress.setup = false;
  progress.verified = false;
  progress.completed_steps = 0;
  progress.enter("initialize");
  np101::Context context;
  np101::Graph graph(context, test.tensors.size(), test.nodes.size());
  std::map<std::string, vsi_nn_tensor_id_t> ids;
  for (const auto &tensor : test.tensors) {
    progress.enter("tensor_" + tensor.name);
    std::vector<std::uint8_t> initial;
    if (tensor.initial_file != "-") {
      initial = testing::read_bytes(test.root, tensor.initial_file, tensor.spec.bytes());
    }
    ids[tensor.name] = np101::add_tensor(graph, tensor.spec, tensor.storage == "constant", initial);
  }
  std::vector<vsi_nn_node_t *> nodes;
  for (const auto &node : test.nodes) {
    progress.enter("node_" + node.operation);
    nodes.push_back(testing::add_operator(graph, node, ids));
  }
  std::vector<vsi_nn_tensor_id_t> inputs, outputs;
  for (const auto &input : test.inputs) {
    inputs.push_back(ids.at(input.tensor));
  }
  for (const auto &output : test.outputs) {
    outputs.push_back(ids.at(output));
  }
  if (!vsi_nn_SetGraphInputs(graph.get(), inputs.data(), inputs.size()) ||
      !vsi_nn_SetGraphOutputs(graph.get(), outputs.data(), outputs.size())) {
    throw std::runtime_error("SetGraphInputs/Outputs failed");
  }

  progress.enter("setup");
  np101::check(vsi_nn_SetupGraph(graph.get(), FALSE), "SetupGraph");
  progress.setup = true;
  progress.enter("verify");
  np101::check(vsi_nn_VerifyGraph(graph.get()), "VerifyGraph");
  progress.verified = true;
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (test.nodes[index].operation == "ARGMAX") {
      progress.argmax_software |= nodes[index]->nn_param.argmax.local.execute_on_sw;
    }
  }
  run_steps(graph, test, ids, progress);

  progress.enter("release");
  graph.close();
  context.close();
  ++progress.completed_cycles;
  progress.enter("cycle_complete");
  progress.observe_resources();
  progress.save("running");
}
} // namespace

int main(int argc, char **argv) {
  std::cout << std::unitbuf;
  if (argc == 3 && std::string(argv[1]) == "--validate-only") {
    try {
      const auto test = specferry::testing::load_case(argv[2]);
      specferry::testing::validate_case_data(test);
      std::cout << "Fixture structure and input byte counts passed.\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
  if (argc < 3) {
    std::cerr << "usage: np101_op_check CASE_FILE OUTPUT_DIRECTORY [--readback final|each-step] "
                 "[--cycles N]\n"
                 "       np101_op_check --validate-only CASE_FILE\n";
    return 2;
  }
  Progress progress;
  progress.directory = argv[2];
  try {
    for (int index = 3; index < argc; index += 2) {
      if (index + 1 >= argc) {
        throw std::invalid_argument("missing diagnostic option value");
      }
      const std::string option = argv[index], value = argv[index + 1];
      if (option == "--readback" && (value == "final" || value == "each-step")) {
        progress.final_readback = value == "final";
      } else if (option == "--cycles" && !value.empty() &&
                 value.find_first_not_of("0123456789") == std::string::npos) {
        progress.cycles = std::stoul(value);
        if (progress.cycles == 0 || progress.cycles > 20) {
          throw std::invalid_argument("cycles must be between 1 and 20");
        }
      } else {
        throw std::invalid_argument("invalid diagnostic option");
      }
    }
    std::filesystem::create_directories(progress.directory);
    progress.save("running");
    const auto test = specferry::testing::load_case(argv[1]);
    specferry::testing::validate_case_data(test);
    progress.observe_resources();
    for (; progress.cycle < progress.cycles; ++progress.cycle) {
      execute(test, progress);
    }
    progress.enter("complete");
    progress.save("executed");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << progress.phase << ": " << error.what() << '\n';
    try {
      progress.save("failed", error.what());
    } catch (const std::exception &report_error) {
      std::cerr << report_error.what() << '\n';
    }
    return 1;
  }
}
