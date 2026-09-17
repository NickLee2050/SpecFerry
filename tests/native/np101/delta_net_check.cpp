#include "np101/delta_net.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace specferry::np101;
const std::array<std::string, 8> outputs{"output",    "recurrent", "convolution", "qkv",
                                         "convolved", "decay",     "beta",        "core"};

std::vector<std::uint8_t> read_bytes(const fs::path &path, std::size_t size) {
  if (fs::file_size(path) != size) {
    throw std::invalid_argument("unexpected fixture size: " + path.string());
  }
  std::ifstream file(path, std::ios::binary);
  std::vector<std::uint8_t> result(size);
  if (!file.read(reinterpret_cast<char *>(result.data()), result.size())) {
    throw std::runtime_error("cannot read fixture: " + path.string());
  }
  return result;
}

void write_bytes(const fs::path &path, const std::vector<std::uint8_t> &bytes) {
  std::ofstream file(path, std::ios::binary);
  if (!file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) {
    throw std::runtime_error("cannot write result: " + path.string());
  }
}

struct Progress {
  fs::path directory;
  std::string phase = "validate";
  std::string sequence = "none";
  unsigned completed_steps = 0;
  unsigned completed_sequences = 0;
  unsigned readbacks = 0;

  void save(const std::string &status = "running") const {
    std::ofstream file(directory / "execution.json");
    file << "{\n  \"status\": \"" << status << "\",\n  \"phase\": \"" << phase
         << "\",\n  \"sequence\": \"" << sequence
         << "\",\n  \"completed_steps\": " << completed_steps
         << ",\n  \"completed_sequences\": " << completed_sequences
         << ",\n  \"application_readbacks\": " << readbacks
         << ",\n  \"per_step_host_state_uploads\": 0,"
            "\n  \"hardware_execution_proven\": false,"
            "\n  \"device_residency_verified\": false\n}\n";
    if (!file) {
      throw std::runtime_error("cannot write DeltaNet execution report");
    }
  }

  void enter(const std::string &next) {
    phase = next;
    std::cout << sequence << ": " << phase << " step=" << completed_steps << '\n';
    save();
  }
};

void run_sequence(DeltaNet &model, const std::vector<std::vector<std::uint8_t>> &inputs,
                  const std::string &name, bool final_only, Progress &progress) {
  progress.sequence = name;
  progress.completed_steps = 0;
  for (unsigned step = 0; step < inputs.size(); ++step) {
    progress.enter("execute");
    model.step(inputs[step]);
    ++progress.completed_steps;
    if (!final_only || step + 1 == inputs.size()) {
      progress.enter("readback");
      for (const auto &output : outputs) {
        write_bytes(progress.directory /
                        (name + "." + output + "." + std::to_string(step) + ".bin"),
                    model.read(output));
        ++progress.readbacks;
      }
    }
  }
  ++progress.completed_sequences;
  progress.enter("sequence_complete");
}
} // namespace

int main(int argc, char **argv) {
  std::cout << std::unitbuf;
  if (argc != 5) {
    std::cerr << "usage: np101_delta_net_check DEPLOYMENT FIXTURE OUTPUT STEPS\n";
    return 1;
  }
  Progress progress{argv[3]};
  try {
    fs::create_directories(progress.directory);
    const std::string count_text = argv[4];
    if (count_text.empty() || count_text.find_first_not_of("0123456789") != std::string::npos) {
      throw std::invalid_argument("steps must be an integer in [1, 32]");
    }
    const auto count = std::stoul(count_text);
    if (count < 1 || count > 32) {
      throw std::invalid_argument("steps must be in [1, 32]");
    }
    progress.enter("validate");
    WeightStore weights(argv[1]);
    weights.verify();
    const fs::path fixture = argv[2];
    std::vector<std::vector<std::uint8_t>> inputs;
    for (unsigned step = 0; step < count; ++step) {
      inputs.push_back(read_bytes(fixture / ("input." + std::to_string(step) + ".bin"),
                                  DeltaNet::hidden_size * 2));
    }
    const auto initial_state =
        read_bytes(fixture / "initial.recurrent.bin", DeltaNet::recurrent_spec().bytes());
    const auto initial_conv =
        read_bytes(fixture / "initial.convolution.bin", DeltaNet::convolution_spec().bytes());

    progress.enter("initialize");
    Context context;
    DeltaNet model(context, weights);
    run_sequence(model, inputs, "zero", false, progress);

    progress.enter("reset_nonzero");
    model.reset(initial_state, initial_conv);
    run_sequence(model, inputs, "nonzero", false, progress);

    progress.enter("reset_zero");
    model.reset();
    run_sequence(model, inputs, "reset", false, progress);

    progress.enter("reset_final_only");
    model.reset(initial_state, initial_conv);
    run_sequence(model, inputs, "final", true, progress);
    progress.enter("release");
    model.close();

    progress.enter("recreate");
    DeltaNet fresh(context, weights);
    run_sequence(fresh, inputs, "fresh", false, progress);
    progress.enter("release");
    fresh.close();
    context.close();
    progress.enter("complete");
    progress.save("executed");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << progress.phase << ": " << error.what() << '\n';
    try {
      progress.save("failed");
    } catch (...) {
      // Preserve the original SDK/fixture error if reporting also fails.
    }
    return 1;
  }
}
