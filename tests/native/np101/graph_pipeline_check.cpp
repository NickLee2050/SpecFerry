#include "binary_io.hpp"
#include "inference/generation_io.hpp"
#include "models/opt/config.hpp"
#include "models/opt/partitioned_graph_model.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/memory_budget.hpp"
#include "np101/tensor.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::testing;
namespace opt = specferry::models::opt;
namespace fs = std::filesystem;
constexpr auto f16 = DataType::Float16;
using SplitVariant = opt::PartitionedGraphModel::Variant;

const char *variant_name(SplitVariant variant) {
  switch (variant) {
  case SplitVariant::MergedKv:
    return "merged-kv";
  case SplitVariant::MergedKvLocalConst:
    return "merged-kv-local-const";
  case SplitVariant::MergedKvLocalMutable:
    return "merged-kv-local-mutable";
  }
  throw std::invalid_argument("unknown partitioned-model diagnostic variant");
}

SplitVariant parse_variant(const std::string &name) {
  for (const auto variant : {SplitVariant::MergedKv, SplitVariant::MergedKvLocalConst,
                             SplitVariant::MergedKvLocalMutable}) {
    if (name == variant_name(variant)) {
      return variant;
    }
  }
  throw std::invalid_argument("unknown partitioned-model diagnostic variant: " + name);
}

struct SplitReadback {
  std::string mode = "outputs";
  std::vector<unsigned> boundary_layers;

  unsigned planned_checks(unsigned layers) const {
    if (mode == "outputs-kv") {
      return 10 * layers + 4;
    }
    return 2 * (2 + boundary_layers.size());
  }
};

unsigned parse_positive_integer(const std::string &text) {
  unsigned value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !value) {
    throw std::invalid_argument("expected a positive integer: " + text);
  }
  return value;
}

std::vector<unsigned> parse_boundary_layers(const std::string &text) {
  std::vector<unsigned> layers;
  std::size_t start = 0;
  while (true) {
    const auto end = text.find(',', start);
    const auto layer = parse_positive_integer(text.substr(start, end - start));
    if (!layers.empty() && layer <= layers.back()) {
      throw std::invalid_argument("boundary layers must be strictly increasing and unique");
    }
    layers.push_back(layer);
    if (end == std::string::npos) {
      return layers;
    }
    start = end + 1;
  }
}

struct PrefixOptions {
  unsigned layers = 1;
  unsigned decoder_group_size = 1;
  SplitVariant variant = SplitVariant::MergedKv;
  bool check_fixture = false, repeat = false;
  SplitReadback readback;
};

PrefixOptions parse_prefix_options(const fs::path &directory) {
  specferry::inference::RunSettings settings(directory);
  PrefixOptions options;
  options.layers =
      settings.number("layers", opt::read_model_config(directory).decoder.layers, 1, 32);
  options.decoder_group_size = settings.number("decoder_group_size", 1, 1, 2);
  options.variant = parse_variant(settings.text("variant", "merged-kv"));
  options.check_fixture = settings.number("check_fixture", 0, 0, 1);
  options.repeat = settings.number("repeat", 0, 0, 1);
  options.readback.mode = settings.text("readback", "outputs");
  const auto boundaries = settings.text("boundary_layers", "");
  if (!boundaries.empty()) {
    options.readback.boundary_layers = parse_boundary_layers(boundaries);
  }
  settings.finish();
  if ((options.readback.mode != "outputs" && options.readback.mode != "outputs-kv" &&
       options.readback.mode != "boundaries") ||
      ((options.readback.mode == "boundaries") != !boundaries.empty()) ||
      (!boundaries.empty() && options.readback.boundary_layers.back() > options.layers)) {
    throw std::invalid_argument("invalid prefix readback scope");
  }
  return options;
}

class Report {
public:
  const fs::path directory;
  unsigned checks = 0, failures = 0;

  explicit Report(const fs::path &output) : directory(output), rows_(output / "checks.tsv") {
    rows_ << "check\tpassed\telements\tnonzero\tmax_abs_error\n";
    phase("initialize");
  }

  void phase(const std::string &name, bool released = false) {
    std::ofstream stream(directory / "execution.json");
    stream << "{\"phase\":\"" << name << "\",\"released\":" << (released ? "true" : "false")
           << ",\"checks\":" << checks << ",\"failures\":" << failures << "}\n";
    if (!stream) {
      throw std::runtime_error("cannot save pipeline progress");
    }
  }

  void memory(const std::string &name, Context &context, std::size_t loaded_weight_payload_bytes) {
    const auto &budget = context.memory();
    std::ofstream stream(directory / ("memory." + name + ".json"));
    stream << "{\"application_segment_limit_bytes\":" << segment_payload_limit
           << ",\"loaded_weight_payload_bytes\":" << loaded_weight_payload_bytes
           << ",\"const_live_bytes\":" << budget.live(true)
           << ",\"const_peak_bytes\":" << budget.peak(true)
           << ",\"nonconst_live_bytes\":" << budget.live(false)
           << ",\"nonconst_peak_bytes\":" << budget.peak(false) << "}\n";
    if (!stream) {
      throw std::runtime_error("cannot save application memory accounting");
    }
  }

  bool compare(const std::string &name, const std::vector<std::uint8_t> &actual,
               const std::vector<std::uint8_t> &expected, DataType type, float atol = 0,
               float rtol = 0, bool exact_bits = false) {
    if (actual.size() != expected.size() || actual.size() % dtype_bytes(type)) {
      throw std::runtime_error("comparison shape mismatch: " + name);
    }
    bool passed = true;
    unsigned nonzero = 0;
    float maximum = 0;
    for (std::size_t offset = 0; offset < actual.size(); offset += dtype_bytes(type)) {
      if (type == f16) {
        std::uint16_t a, b;
        std::memcpy(&a, actual.data() + offset, 2);
        std::memcpy(&b, expected.data() + offset, 2);
        const auto value = vsi_nn_Fp16ToFp32(a), reference = vsi_nn_Fp16ToFp32(b);
        const auto error = std::abs(value - reference);
        maximum = std::max(maximum, error);
        nonzero += value != 0;
        passed &= std::isfinite(value) && error <= atol + rtol * std::abs(reference);
      } else if (type == DataType::Int32) {
        std::int32_t a, b;
        std::memcpy(&a, actual.data() + offset, 4);
        std::memcpy(&b, expected.data() + offset, 4);
        nonzero += a != 0;
        passed &= a == b;
        maximum = std::max(maximum, std::abs(float(a) - float(b)));
      } else {
        throw std::invalid_argument("diagnostic comparison requires FP16 or INT32");
      }
    }
    if (exact_bits) {
      passed &= actual == expected;
    }
    ++checks;
    failures += !passed;
    rows_ << name << '\t' << passed << '\t' << actual.size() / dtype_bytes(type) << '\t' << nonzero
          << '\t' << maximum << std::endl;
    write_bytes(directory / (name + ".bin"), actual);
    if (!rows_) {
      throw std::runtime_error("cannot save pipeline comparison");
    }
    if (!passed) {
      std::cout << "FAIL " << name << ": max_abs_error=" << maximum;
      if (exact_bits && actual != expected) {
        std::cout << "; bytes differ";
      }
      std::cout << std::endl;
    }
    return passed;
  }

private:
  std::ofstream rows_;
};

class SplitValidation {
public:
  std::size_t loaded_weight_payload_bytes = 0;

  SplitValidation(Report &report, const SplitReadback &readback, unsigned layers,
                  unsigned decoder_group_size, SplitVariant variant)
      : report_(report), readback_(readback), layers_(layers),
        decoder_group_size_(decoder_group_size), variant_(variant), before_(tensor_transfers()) {
    save();
  }

  ~SplitValidation() {
    try {
      save();
    } catch (...) {
      // Explicit saves report IO errors; unwinding must preserve the original error.
    }
  }

  void begin_step(unsigned index) {
    steps_.push_back({index, false, report_.checks, report_.failures, tensor_transfers()});
    save();
  }

  void executed_step() {
    steps_.back().executed = true;
    save();
  }

  void save() {
    if (!steps_.empty()) {
      auto &step = steps_.back();
      step.checks = report_.checks - step.checks_before;
      step.failures = report_.failures - step.failures_before;
      step.after = tensor_transfers();
    }
    std::ofstream stream(report_.directory / "validation.json");
    stream << "{\"mode\":\"" << readback_.mode << "\",\"boundary_layers\":[";
    for (std::size_t index = 0; index < readback_.boundary_layers.size(); ++index) {
      stream << (index ? "," : "") << readback_.boundary_layers[index];
    }
    const bool cache_readback = readback_.mode == "outputs-kv";
    const auto completed_steps =
        std::count_if(steps_.begin(), steps_.end(), [](const Step &step) { return step.executed; });
    stream << "],\"layer_count\":" << layers_ << ",\"decoder_group_size\":" << decoder_group_size_
           << ",\"variant\":\"" << variant_name(variant_) << '"' << ",\"execution_graphs\":"
           << 2 + (layers_ + decoder_group_size_ - 1) / decoder_group_size_
           << ",\"planned_steps\":2,\"planned_checks\":" << readback_.planned_checks(layers_)
           << ",\"completed_steps\":" << completed_steps << ",\"checks\":" << report_.checks
           << ",\"failures\":" << report_.failures
           << ",\"kv_checks_requested\":" << (cache_readback ? "true" : "false")
           << ",\"loaded_weight_payload_bytes\":" << loaded_weight_payload_bytes
           << ",\"transfer_scope\":\"explicit_tensor_helpers_excluding_sdk_internal_and_"
              "addtensor_initialization\",\"transfers\":{";
    write_transfers(stream, before_, tensor_transfers());
    stream << "},\"steps\":[";
    for (std::size_t index = 0; index < steps_.size(); ++index) {
      const auto &step = steps_[index];
      stream << (index ? "," : "") << "{\"index\":" << step.index
             << ",\"executed\":" << (step.executed ? "true" : "false")
             << ",\"checks\":" << step.checks << ",\"failures\":" << step.failures << ',';
      write_transfers(stream, step.before, step.after);
      stream << '}';
    }
    stream << "]}\n";
    if (!stream) {
      throw std::runtime_error("cannot save split-prefix validation scope");
    }
  }

private:
  struct Step {
    unsigned index;
    bool executed;
    unsigned checks_before, failures_before;
    TensorTransfers before;
    unsigned checks = 0, failures = 0;
    TensorTransfers after{};
  };

  static void write_transfers(std::ostream &stream, TensorTransfers before, TensorTransfers after) {
    stream << "\"reads\":" << after.reads - before.reads
           << ",\"read_bytes\":" << after.read_bytes - before.read_bytes
           << ",\"uploads\":" << after.uploads - before.uploads
           << ",\"upload_bytes\":" << after.upload_bytes - before.upload_bytes;
  }

  Report &report_;
  const SplitReadback &readback_;
  unsigned layers_;
  unsigned decoder_group_size_;
  SplitVariant variant_;
  TensorTransfers before_;
  std::vector<Step> steps_;
};

struct PrefixReference {
  opt::ModelConfig config;
  std::vector<std::int32_t> tokens = std::vector<std::int32_t>(2);
  std::vector<std::int32_t> predictions = std::vector<std::int32_t>(2);
  std::vector<std::map<std::string, std::vector<std::uint8_t>>> steps{2};

  PrefixReference(const fs::path &fixture, unsigned layers)
      : config(opt::read_model_config(fixture)) {
    if (!layers || layers != config.decoder.layers || config.decoder.capacity < tokens.size()) {
      throw std::invalid_argument("prepare a prefix fixture with exactly the requested layers");
    }
    std::ifstream source(fixture / "tokens.txt");
    std::string extra;
    if (!(source >> tokens[0] >> tokens[1]) || source >> extra) {
      throw std::invalid_argument("prefix fixture needs exactly two teacher tokens");
    }
    std::ifstream predicted(fixture / "predictions.txt");
    if (!(predicted >> predictions[0] >> predictions[1]) || predicted >> extra) {
      throw std::invalid_argument("prefix fixture needs exactly two predicted tokens");
    }
    // Validate all reference sizes before Context opens the board, including
    // intermediate tensors reserved for fine-grained readback.
    for (unsigned step = 0; step < tokens.size(); ++step) {
      config.validate_token(tokens[step]);
      config.validate_token(predictions[step]);
      auto read = [&](const std::string &name, std::size_t elements) {
        steps[step][name] = read_bytes(
            fixture, "teacher." + std::to_string(step) + "." + name + ".bin", elements * 2);
      };
      read("logits", config.vocabulary);
      for (unsigned layer = 0; layer < layers; ++layer) {
        const auto prefix = "layer." + std::to_string(layer);
        read(prefix, config.decoder.hidden);
        for (const auto *name : {"keys", "values"}) {
          read(prefix + "." + name, config.decoder.hidden * (step + 1));
        }
      }
    }
  }
};

using SplitPrefixModel = opt::PartitionedGraphModel;

std::vector<std::uint8_t> cache_rows(const std::vector<std::uint8_t> &data, unsigned heads,
                                     unsigned capacity, std::size_t row_bytes, unsigned first,
                                     unsigned end,
                                     const char *error = "invalid head-major cache slice") {
  if (first > end || end > capacity || data.size() != std::size_t(heads) * capacity * row_bytes) {
    throw std::runtime_error(error);
  }
  std::vector<std::uint8_t> result;
  result.reserve(std::size_t(heads) * (end - first) * row_bytes);
  for (unsigned head = 0; head < heads; ++head) {
    const auto begin = data.begin() + (std::size_t(head) * capacity + first) * row_bytes;
    result.insert(result.end(), begin, begin + std::size_t(end - first) * row_bytes);
  }
  return result;
}

std::vector<std::uint8_t> valid_prefix(const std::vector<std::uint8_t> &data, unsigned heads,
                                       unsigned capacity, std::size_t row_bytes, unsigned length) {
  // Native reads contain full capacity per head; CPU references contain valid rows only.
  return cache_rows(data, heads, capacity, row_bytes, 0, length,
                    "invalid head-major diagnostic tensor size");
}

bool compare_cache_lifecycle(SplitPrefixModel &model, const PrefixReference &reference,
                             unsigned step,
                             std::map<std::string, std::vector<std::uint8_t>> &previous_cache,
                             Report &report) {
  const auto &config = reference.config.decoder;
  const auto row_bytes = config.kv_spec().head_dim * 2;
  for (unsigned layer = 0; layer < config.layers; ++layer) {
    for (bool values : {false, true}) {
      const auto boundary = "layer." + std::to_string(layer) + (values ? ".values" : ".keys");
      const auto prefix = "teacher." + std::to_string(step) + "." + boundary;
      // Merged KV variants perform exactly one SDK read here. All subsequent
      // comparisons and saved views use these same normalized head-major bytes.
      const auto actual = model.read_cache(layer, values);
      write_bytes(report.directory / (prefix + ".backing.bin"), actual);
      const auto current =
          cache_rows(actual, config.heads, config.capacity, row_bytes, step, step + 1);
      const auto expected = cache_rows(reference.steps[step].at(boundary), config.heads, step + 1,
                                       row_bytes, step, step + 1);
      if (!report.compare(prefix + ".current", current, expected, f16, 0.08f, 0.02f)) {
        return false;
      }
      if (step) {
        const auto history = valid_prefix(actual, config.heads, config.capacity, row_bytes, step);
        if (!report.compare(prefix + ".history", history, previous_cache.at(boundary), f16, 0, 0,
                            true)) {
          return false;
        }
      }
      const auto suffix =
          cache_rows(actual, config.heads, config.capacity, row_bytes, step + 1, config.capacity);
      if (!report.compare(prefix + ".suffix", suffix, std::vector<std::uint8_t>(suffix.size()), f16,
                          0, 0, true)) {
        return false;
      }
      previous_cache[boundary] =
          valid_prefix(actual, config.heads, config.capacity, row_bytes, step + 1);
    }
  }
  return true;
}

void split_prefix_check(Context &context, const WeightStore &weights,
                        const PrefixReference &reference, Report &report,
                        const SplitReadback &readback, unsigned decoder_group_size,
                        SplitVariant variant) {
  const auto &config = reference.config.decoder;
  SplitValidation validation(report, readback, config.layers, decoder_group_size, variant);
  report.phase("setup");
  SplitPrefixModel model(context, weights, reference.config, {}, decoder_group_size, variant);
  validation.loaded_weight_payload_bytes = model.weight_bytes();
  report.memory("initialized", context, validation.loaded_weight_payload_bytes);
  validation.save();
  std::map<std::string, std::vector<std::uint8_t>> previous_cache;
  for (unsigned step = 0; step < reference.tokens.size(); ++step) {
    if (report.failures) {
      break;
    }
    report.phase("teacher." + std::to_string(step));
    validation.begin_step(step);
    model.step(reference.tokens[step]);
    validation.executed_step();
    const auto prefix = "teacher." + std::to_string(step) + ".";
    report.compare(prefix + "logits", model.read_output("logits"),
                   reference.steps[step].at("logits"), f16, 0.08f, 0.02f);
    report.compare(prefix + "selected", model.read_output("selected"),
                   as_bytes<std::int32_t>({reference.predictions[step]}), DataType::Int32);
    if (readback.mode == "outputs-kv" && !report.failures) {
      compare_cache_lifecycle(model, reference, step, previous_cache, report);
    }
    for (const auto boundary : readback.boundary_layers) {
      const auto name = "layer." + std::to_string(boundary - 1);
      report.compare(prefix + name, model.read_layer(boundary - 1, "output"),
                     reference.steps[step].at(name), f16, 0.08f, 0.02f);
    }
    validation.save();
  }
  report.phase("release");
  model.close();
  report.memory("released", context, validation.loaded_weight_payload_bytes);
  validation.save();
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: np101_graph_pipeline_check DEPLOYMENT FIXTURE OUTPUT\n";
    return 2;
  }
  try {
    const auto prefix_options = parse_prefix_options(argv[2]);
    const PrefixReference reference(argv[2], prefix_options.layers);
    if (prefix_options.check_fixture) {
      std::cout << "Prefix reference sizes and configuration passed; device not opened.\n"
                << "Variant: " << variant_name(prefix_options.variant)
                << "; decoder group size: " << prefix_options.decoder_group_size
                << "; execution graphs: "
                << 2 + (prefix_options.layers + prefix_options.decoder_group_size - 1) /
                           prefix_options.decoder_group_size
                << '\n';
      return 0;
    }
    fs::create_directories(argv[3]);
    Report report(argv[3]);
    SdkTimings timings(argv[3]);
    WeightStore weights(argv[1]);
    weights.verify();
    if (prefix_options.repeat) {
      for (unsigned cycle = 0; cycle < 2; ++cycle) {
        const auto name = "cycle." + std::to_string(cycle);
        report.phase(name);
        const auto directory = fs::path(argv[3]) / name;
        fs::create_directories(directory);
        Report cycle_report(directory);
        Context cycle_context;
        split_prefix_check(cycle_context, weights, reference, cycle_report, prefix_options.readback,
                           prefix_options.decoder_group_size, prefix_options.variant);
        cycle_context.close();
        cycle_report.phase(cycle_report.failures ? "numerical_failure" : "complete", true);
        report.checks += cycle_report.checks;
        report.failures += cycle_report.failures;
        if (report.failures) {
          break;
        }
      }
      report.phase(report.failures ? "numerical_failure" : "complete", true);
      std::cout << report.checks << " checks; " << report.failures << " failures" << std::endl;
      return report.failures ? 1 : 0;
    }
    Context context;
    split_prefix_check(context, weights, reference, report, prefix_options.readback,
                       prefix_options.decoder_group_size, prefix_options.variant);
    context.close();
    report.phase(report.failures ? "numerical_failure" : "complete", true);
    std::cout << report.checks << " checks; " << report.failures << " failures" << std::endl;
    return report.failures ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
