#include "case_file.hpp"
#include "models/opt/config.hpp"
#include "models/opt/decoder.hpp"
#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using specferry::models::opt::DecoderSlice;
namespace fs = std::filesystem;

struct Report {
  fs::path directory;
  std::string phase = "validate";
  unsigned steps = 0, sequences = 0;
  std::size_t uploads = 0, upload_bytes = 0, reads = 0, cache_writes = 0, revalidations = 0;
  bool invalid_input_rejected = false, empty_output_rejected = false;
  bool capacity_rejected = false, released = false;
  double execute_seconds = 0;

  void save(const std::string &status = "running") const {
    std::ofstream out(directory / "execution.json");
    out << std::boolalpha << "{\"status\":\"" << status << "\",\"phase\":\"" << phase
        << "\",\"steps\":" << steps << ",\"sequences\":" << sequences << ",\"uploads\":" << uploads
        << ",\"upload_bytes\":" << upload_bytes << ",\"reads\":" << reads
        << ",\"cache_writes\":" << cache_writes << ",\"cache_revalidations\":" << revalidations
        << ",\"invalid_input_rejected\":" << invalid_input_rejected
        << ",\"empty_output_rejected\":" << empty_output_rejected
        << ",\"capacity_rejected\":" << capacity_rejected << ",\"released\":" << released
        << ",\"execute_seconds\":" << execute_seconds << "}\n";
    if (!out) {
      throw std::runtime_error("cannot write OPT execution report");
    }
  }
};

bool checkpoint(unsigned index, unsigned count) {
  return index == 0 || index == 1 || index == 3 || index == 7 || index == 31 || index == 255 ||
         index == 511 || index + 1 == count;
}

void capture(DecoderSlice &model, const std::vector<unsigned> &layers, const std::string &sequence,
             unsigned index, const fs::path &output) {
  for (auto layer : layers) {
    for (const auto *name :
         {"query", "key", "value", "probabilities", "mixer", "attention_residual", "attention_norm",
          "fc1", "activation", "mlp", "ffn_residual", "output", "keys", "values"}) {
      auto data = model.read(layer, name);
      const auto filename = sequence + ".layer." + std::to_string(layer) + "." + name + "." +
                            std::to_string(index) + ".bin";
      std::ofstream stream(output / filename, std::ios::binary);
      if (!stream.write(reinterpret_cast<const char *>(data.data()), data.size())) {
        throw std::runtime_error("cannot save OPT output: " + filename);
      }
    }
  }
}

void run(DecoderSlice &model, const std::vector<std::vector<std::uint8_t>> &inputs,
         const std::vector<unsigned> &layers, const std::string &sequence, unsigned count,
         bool final_only, Report &report) {
  for (unsigned index = 0; index < count; ++index) {
    report.phase = sequence;
    report.save();
    const auto before = specferry::np101::tensor_transfers();
    const auto start = std::chrono::steady_clock::now();
    model.step(inputs[index]);
    report.execute_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto after = specferry::np101::tensor_transfers();
    report.uploads += after.uploads - before.uploads;
    report.upload_bytes += after.upload_bytes - before.upload_bytes;
    report.reads += after.reads - before.reads;
    if (after.uploads - before.uploads != 1 + layers.size() ||
        after.upload_bytes - before.upload_bytes != inputs[index].size() + 4 * layers.size() ||
        after.reads != before.reads || model.length() != index + 1) {
      throw std::runtime_error("unexpected OPT step position or explicit tensor transfers");
    }
    ++report.steps;
    if ((final_only && index + 1 == count) || (!final_only && checkpoint(index, count))) {
      capture(model, layers, sequence, index, report.directory);
    }
  }
  ++report.sequences;
}

void reset(DecoderSlice &model, unsigned first, Report &report) {
  model.reset();
  bool rejected = false;
  try {
    model.read(first, "output");
  } catch (const std::logic_error &) {
    rejected = true;
  }
  if (!rejected || model.length()) {
    throw std::runtime_error("OPT reset exposed stale output");
  }
  report.empty_output_rejected = true;
}

void release(DecoderSlice &model, Report &report) {
  report.cache_writes += model.cache_writes();
  report.revalidations += model.cache_revalidations();
  model.close();
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: np101_opt_decoder_check DEPLOYMENT FIXTURE OUTPUT STEPS\n";
    return 2;
  }
  Report report{argv[3]};
  try {
    fs::create_directories(report.directory);
    report.save();
    const auto config = specferry::models::opt::read_config(fs::path(argv[2]) / "components.txt");
    const auto parsed = specferry::np101::parse_shape(argv[4]);
    if (parsed.size() != 1 || parsed[0] < 2 || parsed[0] > config.capacity) {
      throw std::invalid_argument("OPT step count exceeds capacity");
    }
    const auto count = parsed[0];
    std::vector<unsigned> layers;
    std::ifstream selection(fs::path(argv[2]) / "layers.txt");
    unsigned index;
    while (selection >> index) {
      layers.push_back(index);
    }
    if (!selection.eof() || layers.empty() || layers.size() > 4) {
      throw std::invalid_argument("expected an OPT slice of one to four layers");
    }
    specferry::np101::WeightStore weights(argv[1]);
    weights.verify();
    specferry::models::opt::validate_weights(weights, config, layers);
    std::vector<std::vector<std::uint8_t>> inputs;
    for (unsigned step = 0; step < count; ++step) {
      inputs.push_back(specferry::testing::read_bytes(
          argv[2], "input." + std::to_string(step) + ".bin", config.hidden_spec().bytes()));
    }
    report.phase = "initialize";
    report.save();
    specferry::np101::Context context;
    DecoderSlice model(context, weights, config, layers);
    run(model, inputs, layers, "zero", count, false, report);
    const auto prior_length = model.length();
    const auto prior_writes = model.cache_writes();
    try {
      model.step({});
    } catch (const std::invalid_argument &) {
      report.invalid_input_rejected = true;
    }
    if (count == config.capacity) {
      try {
        model.step(inputs.front());
      } catch (const std::out_of_range &) {
        report.capacity_rejected = true;
      }
      if (!report.capacity_rejected) {
        throw std::runtime_error("OPT accepted an exhausted cache");
      }
    }
    if (!report.invalid_input_rejected || model.length() != prior_length ||
        model.cache_writes() != prior_writes) {
      throw std::runtime_error("OPT rejected input changed state");
    }
    reset(model, layers.front(), report);
    run(model, inputs, layers, "reset", std::min(8U, count), false, report);
    reset(model, layers.front(), report);
    run(model, inputs, layers, "final", std::min(32U, count), true, report);
    release(model, report);
    DecoderSlice fresh(context, weights, config, layers);
    run(fresh, inputs, layers, "fresh", std::min(8U, count), false, report);
    release(fresh, report);
    context.close();
    report.released = true;
    report.phase = "complete";
    report.save("executed");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << report.phase << ": " << error.what() << '\n';
    try {
      report.save("failed");
    } catch (...) {
    }
    return 1;
  }
}
