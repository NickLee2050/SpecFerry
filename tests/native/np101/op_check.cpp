#include "case_file.hpp"
#include "operators.hpp"

#include "vsi_nn_rnn_prv.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

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
  std::filesystem::path directory;
  std::string phase = "load_case";
  bool setup = false;
  bool verified = false;
  bool rnn = false;
  bool state_swappable = false;
  std::size_t host_state_bytes = 0;
  bool argmax_software = false;
  std::size_t completed_steps = 0;
  std::vector<double> run_times;

  void save(const std::string &status, const std::string &error = "") const {
    std::ofstream output(directory / "execution.json");
    output << std::boolalpha << "{\n  \"status\": " << json_string(status)
           << ",\n  \"phase\": " << json_string(phase) << ",\n  \"error\": " << json_string(error)
           << ",\n  \"setup_passed\": " << setup << ",\n  \"verify_passed\": " << verified
           << ",\n  \"completed_steps\": " << completed_steps
           << ",\n  \"hardware_execution_proven\": false,\n  \"rnn_connection\": " << rnn
           << ",\n  \"rnn_tensor_swappable\": " << state_swappable
           << ",\n  \"rnn_host_buffer_bytes\": " << host_state_bytes
           << ",\n  \"argmax_execute_on_sw\": " << argmax_software << ",\n  \"run_ms\": [";
    for (std::size_t index = 0; index < run_times.size(); ++index) {
      if (index) {
        output << ", ";
      }
      output << run_times[index];
    }
    output << "]\n}\n";
    if (!output) {
      throw std::runtime_error("failed to write execution report");
    }
  }

  void enter(const std::string &next) {
    phase = next;
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

void configure_feedback(np101::Graph &graph, const testing::CaseDefinition &test,
                        const std::map<std::string, vsi_nn_tensor_id_t> &ids, Progress &progress) {
  if (test.feedback_input.empty()) {
    return;
  }
  vsi_nn_rnn_external_connection_t connection{};
  std::fill(std::begin(connection.inputs), std::end(connection.inputs), VSI_NN_TENSOR_ID_NA);
  connection.output = ids.at(test.feedback_output);
  connection.inputs[0] = ids.at(test.feedback_input);
  np101::check(vsi_nn_SetupRNNConnections(graph.get(), &connection, 1), "SetupRNNConnections");
  progress.rnn = true;

  // Diagnostic access to this installed SDK's workspace. A copy-backed connection
  // may pass numerical tests but cannot satisfy the device-resident state contract.
  auto *workspace = static_cast<vsi_nn_rnn_wksp_t *>(graph.get()->rnn_wksp);
  if (!workspace || !workspace->external_connection_list) {
    throw std::runtime_error("RNN connection workspace missing");
  }
  auto *state = workspace->external_connection_list;
  progress.state_swappable = state->tensor_swappable;
  progress.host_state_bytes = state->buffer.data_size;
}

void run_steps(np101::Graph &graph, const testing::CaseDefinition &test,
               const std::map<std::string, vsi_nn_tensor_id_t> &ids, Progress &progress) {
  for (std::size_t step = 0; step < test.steps; ++step) {
    progress.enter("upload_step_" + std::to_string(step));
    if (step == test.reset_after && step != 0) {
      np101::check(vsi_nn_ResetRNNBuffers(graph.get()), "ResetRNNBuffers");
      // The fixture explicitly restores the initial input state after reset.
      const auto &state = tensor_definition(test, test.feedback_input);
      auto initial = testing::read_bytes(test.root, state.initial_file, state.spec.bytes());
      np101::upload_tensor(graph, ids.at(state.name), initial);
    }
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

    progress.enter("read_step_" + std::to_string(step));
    for (const auto &name : test.outputs) {
      auto bytes = np101::read_tensor(graph, ids.at(name));
      auto path = progress.directory / (name + "." + std::to_string(step) + ".bin");
      std::ofstream output(path, std::ios::binary);
      output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      if (!output) {
        throw std::runtime_error("failed to write output: " + path.string());
      }
    }
  }
}

void execute(const testing::CaseDefinition &test, Progress &progress) {
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
    ids[tensor.name] = np101::add_tensor(graph, tensor.spec, tensor.storage == "constant", initial,
                                         tensor.storage == "handle");
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
  if (!test.feedback_input.empty()) {
    inputs.push_back(ids.at(test.feedback_input));
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
  configure_feedback(graph, test, ids, progress);
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    if (test.nodes[index].operation == "ARGMAX") {
      progress.argmax_software |= nodes[index]->nn_param.argmax.local.execute_on_sw;
    }
  }
  run_steps(graph, test, ids, progress);

  progress.enter("release");
  graph.close();
  context.close();
  progress.save("executed");
}
} // namespace

int main(int argc, char **argv) {
  std::cout << std::unitbuf;
  if (argc == 3 && std::string(argv[1]) == "--validate-only") {
    try {
      const auto test = specferry::testing::load_case(argv[2]);
      for (const auto &tensor : test.tensors) {
        if (tensor.initial_file != "-") {
          specferry::testing::read_bytes(test.root, tensor.initial_file, tensor.spec.bytes());
        }
      }
      for (const auto &input : test.inputs) {
        const auto &tensor = tensor_definition(test, input.tensor);
        for (const auto &file : input.files) {
          specferry::testing::read_bytes(test.root, file, tensor.spec.bytes());
        }
      }
      std::cout << "Fixture structure and input byte counts passed.\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
  if (argc != 3) {
    std::cerr << "usage: np101_op_check CASE_FILE OUTPUT_DIRECTORY\n"
                 "       np101_op_check --validate-only CASE_FILE\n";
    return 2;
  }
  Progress progress;
  progress.directory = argv[2];
  try {
    std::filesystem::create_directories(progress.directory);
    progress.save("running");
    const auto test = specferry::testing::load_case(argv[1]);
    execute(test, progress);
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
