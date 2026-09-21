#include "case_file.hpp"
#include "models/qwen3_5/attention.hpp"
#include "np101/context.hpp"
#include "np101/kv_cache.hpp"
#include "np101/weights.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::models::qwen3_5;
namespace fs = std::filesystem;
const std::array<std::string, 7> outputs{"output", "query",  "key",          "value",
                                         "keys",   "values", "probabilities"};

bool checkpoint(unsigned step, unsigned count) {
  return step == 0 || step == 1 || step == 3 || step == 7 || step == 31 || step == 255 ||
         step == 511 || step + 1 == count;
}

struct Progress {
  fs::path directory;
  KvSpec kv{};
  std::string phase = "validate";
  std::string sequence = "none";
  unsigned completed_steps = 0;
  unsigned completed_sequences = 0;
  std::size_t cache_writes = 0;
  std::size_t revalidations = 0;
  bool capacity_rejected = false;
  bool invalid_truncate_rejected = false;

  void save(const std::string &status = "running") const {
    std::ofstream file(directory / "execution.json");
    file << "{\"status\":\"" << status << "\",\"phase\":\"" << phase << "\",\"sequence\":\""
         << sequence << "\",\"completed_steps\":" << completed_steps
         << ",\"completed_sequences\":" << completed_sequences
         << ",\"cache_writes\":" << cache_writes
         << ",\"copy_graph_revalidations\":" << revalidations
         << ",\"capacity_rejected\":" << (capacity_rejected ? "true" : "false")
         << ",\"invalid_truncate_rejected\":" << (invalid_truncate_rejected ? "true" : "false")
         << ",\"cache_payload_bytes\":" << kv.heads * kv.head_dim * 4ULL * kv.capacity
         << ",\"slot_write_payload_bytes\":" << kv.heads * kv.head_dim * 4ULL
         << ","
            "\"cache_copies\":1,\"per_step_host_kv_transfers\":false,"
            "\"hardware_execution_proven\":false,\"device_residency_verified\":false}\n";
    if (!file) {
      throw std::runtime_error("cannot write attention report");
    }
  }

  void enter(const std::string &next) {
    phase = next;
    std::cout << sequence << ": " << phase << " steps=" << completed_steps << std::endl;
    save();
  }
};

void capture(Attention &model, const std::string &name, unsigned index, Progress &progress) {
  for (const auto &output : outputs) {
    auto bytes = model.read(output);
    std::ofstream file(progress.directory /
                           (name + "." + output + "." + std::to_string(index) + ".bin"),
                       std::ios::binary);
    if (!file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) {
      throw std::runtime_error("cannot write attention output");
    }
  }
}

void run_sequence(Attention &model, const std::vector<std::vector<std::uint8_t>> &inputs,
                  const std::string &name, bool final_only, Progress &progress) {
  progress.sequence = name;
  for (unsigned index = 0; index < inputs.size(); ++index) {
    progress.enter("execute");
    model.step(inputs[index]);
    ++progress.completed_steps;
    if ((final_only && index + 1 == inputs.size()) ||
        (!final_only && checkpoint(index, inputs.size()))) {
      progress.enter("readback");
      capture(model, name, index, progress);
    }
  }
  ++progress.completed_sequences;
  progress.enter("sequence_complete");
}

void check_rejections(Attention &model, const std::vector<std::uint8_t> &input,
                      Progress &progress) {
  const auto length = model.length();
  const auto writes = model.cache_writes();
  try {
    model.truncate(length + 1);
  } catch (const std::out_of_range &) {
    progress.invalid_truncate_rejected = true;
  }
  if (length == progress.kv.capacity) {
    try {
      model.step(input);
    } catch (const std::out_of_range &) {
      progress.capacity_rejected = true;
    }
    if (!progress.capacity_rejected) {
      throw std::runtime_error("attention accepted an out-of-capacity token");
    }
  }
  if (!progress.invalid_truncate_rejected || model.length() != length ||
      model.cache_writes() != writes) {
    throw std::runtime_error(
        "invalid cache operation changed the valid prefix or submitted a write");
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: np101_attention_check DEPLOYMENT FIXTURE OUTPUT STEPS\n";
    return 1;
  }
  Progress progress{argv[3]};
  try {
    const std::string text = argv[4];
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
      throw std::invalid_argument("steps must be an integer in [2, 512]");
    }
    auto count = std::stoul(text);
    if (count < 2 || count > 512) {
      throw std::invalid_argument("steps must be in [2, 512]");
    }
    fs::create_directories(progress.directory);
    progress.enter("validate");
    const auto config = read_config(fs::path(argv[2]) / "components.txt");
    progress.kv = config.kv;
    WeightStore weights(argv[1]);
    weights.verify();
    std::vector<std::vector<std::uint8_t>> inputs;
    for (unsigned index = 0; index < count; ++index) {
      inputs.push_back(specferry::testing::read_bytes(
          argv[2], "input." + std::to_string(index) + ".bin", config.hidden_spec().bytes()));
    }
    const std::vector<std::vector<std::uint8_t>> short_inputs(
        inputs.begin(), inputs.begin() + std::min<std::size_t>(8, count));
    const std::vector<std::vector<std::uint8_t>> branch_inputs{inputs.front(), inputs.back()};

    progress.enter("initialize");
    Context context;
    Attention model(context, weights, config, 3);
    run_sequence(model, inputs, "zero", false, progress);
    check_rejections(model, inputs.front(), progress);

    progress.enter("truncate");
    model.truncate(std::min<unsigned>(3, count - 1));
    run_sequence(model, branch_inputs, "branch", false, progress);

    progress.enter("reset");
    model.reset();
    run_sequence(model, short_inputs, "reset", false, progress);

    progress.enter("reset_final_only");
    model.reset();
    run_sequence(model, inputs, "final", true, progress);
    progress.cache_writes = model.cache_writes();
    progress.revalidations = model.cache_revalidations();
    progress.enter("release");
    model.close();

    progress.enter("recreate");
    Attention fresh(context, weights, config, 3);
    run_sequence(fresh, short_inputs, "fresh", false, progress);
    progress.cache_writes += fresh.cache_writes();
    progress.revalidations += fresh.cache_revalidations();
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
      // Do not replace the original device/fixture failure with a report error.
    }
    return 1;
  }
}
