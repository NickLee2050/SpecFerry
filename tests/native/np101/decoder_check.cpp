#include "case_file.hpp"
#include "models/qwen3_5/decoder.hpp"
#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <algorithm>
#include <chrono>
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
#include <utility>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::models::qwen3_5;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

bool checkpoint(unsigned step, unsigned count) {
  return step == 0 || step == 1 || step == 3 || step == 7 || step == 31 || step == 255 ||
         step == 511 || step + 1 == count;
}

struct Progress {
  fs::path directory;
  unsigned first_layer;
  Config config{};
  std::vector<unsigned> selected;
  unsigned attention_layers = 0;
  std::string phase = "validate";
  std::string sequence = "none";
  unsigned completed_steps = 0;
  unsigned completed_sequences = 0;
  std::size_t readback_bytes = 0;
  std::size_t step_upload_bytes = 0;
  std::size_t step_uploads = 0;
  std::size_t step_reads = 0;
  std::size_t cache_writes = 0;
  std::size_t revalidations = 0;
  bool capacity_rejected = false;
  bool invalid_input_rejected = false;
  bool empty_output_rejected = false;
  bool released = false;
  double execution_seconds = 0;
  double initialization_seconds = 0;
  double reset_seconds = 0;
  double release_seconds = 0;
  std::vector<DecoderMetrics> layers;

  explicit Progress(fs::path output) : directory(std::move(output)), first_layer(0) {}

  void save(const std::string &status = "running") const {
    std::ofstream file(directory / "execution.json");
    file << std::boolalpha << "{\"status\":\"" << status << "\",\"phase\":\"" << phase
         << "\",\"sequence\":\"" << sequence << "\",\"completed_steps\":" << completed_steps
         << ",\"completed_sequences\":" << completed_sequences << ",\"first_layer\":" << first_layer
         << ",\"cache_writes\":" << cache_writes
         << ",\"copy_graph_revalidations\":" << revalidations
         << ",\"capacity_rejected\":" << capacity_rejected
         << ",\"invalid_input_rejected\":" << invalid_input_rejected
         << ",\"empty_output_rejected\":" << empty_output_rejected << ",\"released\":" << released
         << ",\"application_readback_bytes\":" << readback_bytes
         << ",\"step_upload_bytes\":" << step_upload_bytes << ",\"step_uploads\":" << step_uploads
         << ",\"step_reads\":" << step_reads
         << ",\"boundary_upload_bytes\":" << completed_steps * config.hidden * 2ULL
         << ",\"attention_control_upload_bytes\":" << completed_steps * attention_layers * 8ULL
         << ",\"explicit_kv_copy_bytes\":"
         << cache_writes * config.kv.heads * config.kv.head_dim * 4ULL
         << ",\"execution_seconds\":" << execution_seconds
         << ",\"initialization_seconds\":" << initialization_seconds
         << ",\"reset_seconds\":" << reset_seconds << ",\"release_seconds\":" << release_seconds
         << ",\"layers\":[";
    for (std::size_t index = 0; index < layers.size(); ++index) {
      const auto &layer = layers[index];
      if (index) {
        file << ',';
      }
      file << "{\"layer\":" << selected[index] << ",\"steps\":" << layer.steps
           << ",\"normalization_seconds\":" << layer.normalization_seconds
           << ",\"mixer_seconds\":" << layer.mixer_seconds
           << ",\"feed_forward_seconds\":" << layer.feed_forward_seconds
           << ",\"cache_write_seconds\":" << layer.cache_write_seconds
           << ",\"cache_revalidation_seconds\":" << layer.cache_revalidation_seconds << '}';
    }
    file << "]}\n";
    if (!file) {
      throw std::runtime_error("cannot write decoder report");
    }
  }

  void enter(const std::string &next) {
    phase = next;
    std::cout << sequence << ": " << phase << " steps=" << completed_steps << std::endl;
    save();
  }

  void collect(const DecoderGroup &model) {
    const auto current = model.metrics();
    if (layers.empty()) {
      layers.resize(current.size());
    }
    for (std::size_t index = 0; index < current.size(); ++index) {
      auto &total = layers[index];
      const auto &value = current[index];
      total.steps += value.steps;
      total.normalization_seconds += value.normalization_seconds;
      total.mixer_seconds += value.mixer_seconds;
      total.feed_forward_seconds += value.feed_forward_seconds;
      total.cache_write_seconds += value.cache_write_seconds;
      total.cache_revalidation_seconds += value.cache_revalidation_seconds;
      cache_writes += value.cache_writes;
      revalidations += value.cache_revalidations;
    }
  }
};

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void capture(DecoderGroup &model, unsigned index, Progress &progress) {
  for (unsigned layer : progress.selected) {
    std::vector<std::string> outputs{"normalized", "mixer", "residual",
                                     "post_norm",  "mlp",   "output"};
    if (progress.config.mixer(layer) == MixerKind::Attention) {
      outputs.insert(outputs.end(), {"keys", "values"});
    } else {
      outputs.insert(outputs.end(), {"recurrent", "convolution"});
    }
    for (const auto &name : outputs) {
      const auto bytes = model.read(layer, name);
      progress.readback_bytes += bytes.size();
      const auto filename = progress.sequence + ".layer." + std::to_string(layer) + "." + name +
                            "." + std::to_string(index) + ".bin";
      std::ofstream file(progress.directory / filename, std::ios::binary);
      if (!file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) {
        throw std::runtime_error("cannot write decoder output");
      }
    }
  }
}

void run_sequence(DecoderGroup &model, const std::vector<std::vector<std::uint8_t>> &inputs,
                  unsigned count, const std::string &name, bool final_only, Progress &progress) {
  progress.sequence = name;
  for (unsigned index = 0; index < count; ++index) {
    progress.enter("execute");
    const auto start = Clock::now();
    const auto transfers_before = tensor_transfers();
    model.step(inputs[index]);
    const auto transfers_after = tensor_transfers();
    const auto uploads = transfers_after.uploads - transfers_before.uploads;
    const auto upload_bytes = transfers_after.upload_bytes - transfers_before.upload_bytes;
    const auto reads = transfers_after.reads - transfers_before.reads;
    progress.step_uploads += uploads;
    progress.step_upload_bytes += upload_bytes;
    progress.step_reads += reads;
    if (uploads != 1 + 2 * progress.attention_layers ||
        upload_bytes != progress.config.hidden_spec().bytes() + 8 * progress.attention_layers ||
        reads != 0) {
      throw std::runtime_error("decoder step performed unexpected explicit host tensor transfers");
    }
    progress.execution_seconds += elapsed(start);
    ++progress.completed_steps;
    if (model.length() != index + 1) {
      throw std::runtime_error("decoder group committed the wrong position");
    }
    if ((final_only && index + 1 == count) || (!final_only && checkpoint(index, count))) {
      progress.enter("readback");
      capture(model, index, progress);
    }
  }
  ++progress.completed_sequences;
}

void check_rejections(DecoderGroup &model, const std::vector<std::uint8_t> &input,
                      Progress &progress) {
  const auto length = model.length();
  const auto before = model.metrics();
  try {
    model.step({});
  } catch (const std::invalid_argument &) {
    progress.invalid_input_rejected = true;
  }
  if (length == progress.config.kv.capacity) {
    try {
      model.step(input);
    } catch (const std::out_of_range &) {
      progress.capacity_rejected = true;
    }
    if (!progress.capacity_rejected) {
      throw std::runtime_error("decoder accepted an out-of-capacity token");
    }
  }
  const auto after = model.metrics();
  if (!progress.invalid_input_rejected || model.length() != length) {
    throw std::runtime_error("invalid input changed group state");
  }
  for (std::size_t index = 0; index < before.size(); ++index) {
    if (before[index].steps != after[index].steps ||
        before[index].cache_writes != after[index].cache_writes) {
      throw std::runtime_error("rejected input executed a decoder layer");
    }
  }
}

void reset(DecoderGroup &model, Progress &progress) {
  progress.enter("reset");
  const auto start = Clock::now();
  model.reset();
  progress.reset_seconds += elapsed(start);
  bool rejected = false;
  try {
    model.read(progress.first_layer, "output");
  } catch (const std::logic_error &) {
    rejected = true;
  }
  if (!rejected || model.length() != 0) {
    throw std::runtime_error("reset exposed stale decoder output");
  }
  progress.empty_output_rejected = true;
}

void release(DecoderGroup &model, Progress &progress) {
  progress.collect(model);
  progress.enter("release");
  const auto start = Clock::now();
  model.close();
  progress.release_seconds += elapsed(start);
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: np101_decoder_check DEPLOYMENT FIXTURE OUTPUT STEPS FIRST_LAYER\n";
    return 1;
  }
  Progress progress{argv[3]};
  try {
    const std::string count_text = argv[4], layer_text = argv[5];
    if (count_text.empty() || count_text.find_first_not_of("0123456789") != std::string::npos ||
        (layer_text.empty() || layer_text.find_first_not_of("0123456789") != std::string::npos)) {
      throw std::invalid_argument("steps and first layer must be unsigned integers");
    }
    const auto count = std::stoul(count_text);
    if (count < 2 || count > 512) {
      throw std::invalid_argument("steps must be 2-512");
    }
    progress.first_layer = std::stoul(layer_text);
    fs::create_directories(progress.directory);
    progress.enter("validate");
    const auto config = read_config(fs::path(argv[2]) / "components.txt");
    if (count > config.kv.capacity) {
      throw std::invalid_argument("steps exceed configured capacity");
    }
    progress.config = config;
    std::ifstream layer_file(fs::path(argv[2]) / "layers.txt");
    unsigned index;
    while (layer_file >> index) {
      progress.selected.push_back(index);
    }
    if (!layer_file.eof() || progress.selected.empty() ||
        progress.selected.front() != progress.first_layer) {
      throw std::invalid_argument("invalid decoder layer list");
    }
    for (auto layer : progress.selected) {
      progress.attention_layers += config.mixer(layer) == MixerKind::Attention;
    }
    WeightStore weights(argv[1]);
    weights.verify();
    std::vector<std::vector<std::uint8_t>> inputs;
    for (unsigned index = 0; index < count; ++index) {
      inputs.push_back(specferry::testing::read_bytes(
          argv[2], "input." + std::to_string(index) + ".bin", config.hidden_spec().bytes()));
    }

    progress.enter("initialize");
    Context context;
    auto start = Clock::now();
    DecoderGroup model(context, weights, config, progress.selected);
    progress.initialization_seconds += elapsed(start);
    run_sequence(model, inputs, count, "zero", false, progress);
    check_rejections(model, inputs.front(), progress);

    reset(model, progress);
    run_sequence(model, inputs, std::min<unsigned>(8, count), "reset", false, progress);
    reset(model, progress);
    run_sequence(model, inputs, std::min<unsigned>(32, count), "final", true, progress);
    release(model, progress);

    progress.enter("recreate");
    start = Clock::now();
    DecoderGroup fresh(context, weights, config, progress.selected);
    progress.initialization_seconds += elapsed(start);
    run_sequence(fresh, inputs, std::min<unsigned>(8, count), "fresh", false, progress);
    release(fresh, progress);
    context.close();
    progress.released = true;
    progress.enter("complete");
    progress.save("executed");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << progress.phase << ": " << error.what() << '\n';
    try {
      progress.save("failed");
    } catch (...) {
      // Preserve the original fixture/SDK failure if report writing also fails.
    }
    return 1;
  }
}
