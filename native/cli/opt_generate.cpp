#include "models/opt/config.hpp"
#include "models/opt/model.hpp"
#include "np101/context.hpp"
#include "np101/ops/vocabulary.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"
#include "sys/resource.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using specferry::models::opt::Model;
using Clock = std::chrono::steady_clock;

struct Report {
  fs::path directory;
  std::string phase = "validate";
  unsigned layers = 0, steps = 0;
  std::size_t uploads = 0, upload_bytes = 0, reads = 0, read_bytes = 0, cache_writes = 0;
  bool released = false, bounds_rejected = false, reset_rejected_stale = false;
  double initialize_seconds = 0, execute_seconds = 0;

  void sample_host(const std::string &stage) const {
    std::ifstream status("/proc/self/status");
    std::string line;
    std::size_t rss_kib = 0;
    bool found = false;
    while (std::getline(status, line)) {
      if (line.rfind("VmRSS:", 0) == 0) {
        std::istringstream value(line.substr(6));
        found = bool(value >> rss_kib);
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("cannot query current host RSS");
    }
    std::ofstream out(directory / "host-resources.jsonl", std::ios::app);
    out << "{\"phase\":\"" << stage << "\",\"rss_bytes\":" << rss_kib * 1024 << "}\n";
    if (!out) {
      throw std::runtime_error("cannot write host resource observation");
    }
  }

  void save(const std::string &status = "running") const {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
      throw std::runtime_error("cannot query host process memory");
    }
    std::ofstream out(directory / "execution.json");
    out << std::boolalpha << "{\"status\":\"" << status << "\",\"phase\":\"" << phase
        << "\",\"layers\":" << layers << ",\"steps\":" << steps << ",\"uploads\":" << uploads
        << ",\"upload_bytes\":" << upload_bytes << ",\"reads\":" << reads
        << ",\"read_bytes\":" << read_bytes << ",\"cache_writes\":" << cache_writes
        << ",\"released\":" << released << ",\"bounds_rejected\":" << bounds_rejected
        << ",\"reset_rejected_stale\":" << reset_rejected_stale
        << ",\"initialize_seconds\":" << initialize_seconds
        << ",\"execute_seconds\":" << execute_seconds
        << ",\"host_peak_rss_kib\":" << usage.ru_maxrss << "}\n";
    if (!out) {
      throw std::runtime_error("cannot write execution report");
    }
  }

  template <class Action> void observe(Action action) {
    const auto before = specferry::np101::tensor_transfers();
    const auto start = Clock::now();
    action();
    execute_seconds += std::chrono::duration<double>(Clock::now() - start).count();
    const auto after = specferry::np101::tensor_transfers();
    uploads += after.uploads - before.uploads;
    upload_bytes += after.upload_bytes - before.upload_bytes;
    reads += after.reads - before.reads;
    read_bytes += after.read_bytes - before.read_bytes;
  }
};

unsigned number(const char *text) {
  const std::string value(text);
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("expected an unsigned integer");
  }
  const auto parsed = std::stoull(value);
  if (parsed > std::numeric_limits<unsigned>::max()) {
    throw std::out_of_range("integer is too large");
  }
  return static_cast<unsigned>(parsed);
}

std::vector<std::int32_t> read_tokens(const fs::path &file) {
  std::ifstream in(file);
  if (!in) {
    throw std::runtime_error("cannot open token input");
  }
  std::vector<std::int32_t> result;
  std::int64_t value;
  while (in >> value) {
    if (value < 0 || value > std::numeric_limits<std::int32_t>::max()) {
      throw std::invalid_argument("invalid token ID");
    }
    result.push_back(static_cast<std::int32_t>(value));
  }
  if (!in.eof()) {
    throw std::invalid_argument("malformed token file");
  }
  return result;
}

void capture(Model &model, const std::string &prefix, unsigned layers, const fs::path &directory) {
  auto save = [&](const std::string &name, const std::vector<std::uint8_t> &data) {
    std::ofstream out(directory / (prefix + name + ".bin"), std::ios::binary);
    if (!out.write(reinterpret_cast<const char *>(data.data()), data.size())) {
      throw std::runtime_error("cannot save model diagnostic");
    }
  };
  for (const auto *name : {"embedding", "projected", "logits"}) {
    save(name, model.read(name));
  }
  for (unsigned layer = 0; layer < layers; ++layer) {
    save("layer." + std::to_string(layer), model.read("output", layer));
  }
}

void teacher(Model &model, const std::vector<std::int32_t> &tokens, const std::string &sequence,
             bool final_only, Report &report) {
  report.phase = sequence;
  std::ofstream predictions(report.directory / (sequence + ".tokens.txt"));
  for (unsigned index = 0; index < tokens.size(); ++index) {
    report.save();
    const bool select = !final_only || index + 1 == tokens.size();
    std::int32_t prediction = -1;
    report.observe([&] {
      model.consume(tokens[index]);
      if (select) {
        prediction = model.predict();
      }
    });
    ++report.steps;
    if (select) {
      predictions << prediction << '\n';
      capture(model, sequence + "." + std::to_string(index) + ".", report.layers, report.directory);
    }
  }
  if (!predictions) {
    throw std::runtime_error("cannot save predictions");
  }
}

void reset(Model &model, Report &report) {
  model.reset();
  unsigned rejected = 0;
  try {
    model.predict();
  } catch (const std::logic_error &) {
    ++rejected;
  }
  try {
    model.read("embedding");
  } catch (const std::logic_error &) {
    ++rejected;
  }
  if (rejected != 2 || model.length()) {
    throw std::runtime_error("reset exposed stale model outputs");
  }
  report.reset_rejected_stale = true;
}

void save_generation(const specferry::models::opt::Generation &result, const fs::path &directory,
                     const std::string &filename = "generation.json") {
  std::ofstream out(directory / filename);
  out << std::setprecision(17);
  out << "{\"tokens\":[";
  for (unsigned index = 0; index < result.tokens.size(); ++index) {
    out << (index ? "," : "") << result.tokens[index];
  }
  out << "],\"consumed\":" << result.consumed << ",\"stop_reason\":\"" << result.stop_reason
      << "\",\"prefill_seconds\":" << result.prefill_seconds
      << ",\"first_token_seconds\":" << result.first_token_seconds << ",\"token_seconds\":[";
  for (unsigned index = 0; index < result.token_seconds.size(); ++index) {
    out << (index ? "," : "") << result.token_seconds[index];
  }
  out << "],\"total_seconds\":" << result.total_seconds << "}\n";
  if (!out) {
    throw std::runtime_error("cannot save generation result");
  }
}

void benchmark(Model &model, const std::vector<std::int32_t> &tokens, unsigned maximum,
               unsigned warmups, unsigned repeats, Report &report) {
  for (unsigned index = 0; index < warmups + repeats; ++index) {
    const bool warmup = index < warmups;
    const auto name = std::string(warmup ? "warmup." : "measured.") +
                      std::to_string(warmup ? index : index - warmups);
    report.phase = name;
    report.save();
    specferry::models::opt::Generation result;
    // Each request resets the valid KV prefix while retaining the same model.
    // No streaming callback, diagnostics or file IO runs inside the timed call.
    report.observe([&] { result = model.generate(tokens, maximum); });
    report.steps += result.consumed;
    save_generation(result, report.directory, name + ".json");
    report.sample_host(name);
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 8 && argc != 9 && argc != 10) {
    std::cerr << "usage: specferry_opt_generate DEPLOYMENT REQUEST OUTPUT MODE LAYERS MAX_NEW "
                 "TOKENS [SAMPLE_SEED]\n"
              << "MODE is load, teacher, generate, or benchmark; TOKENS is an ID file.\n"
              << "For benchmark, replace [SAMPLE_SEED] with WARMUPS REPEATS.\n";
    return 2;
  }
  Report report{argv[3]};
  try {
    fs::create_directories(report.directory);
    report.save();
    const std::string mode(argv[4]);
    if (mode != "load" && mode != "teacher" && mode != "generate" && mode != "benchmark") {
      throw std::invalid_argument("unknown execution mode");
    }
    if ((mode == "benchmark") != (argc == 10)) {
      throw std::invalid_argument("benchmark requires WARMUPS REPEATS instead of SAMPLE_SEED");
    }
    const auto warmups = mode == "benchmark" ? number(argv[8]) : 0;
    const auto repeats = mode == "benchmark" ? number(argv[9]) : 0;
    if (mode == "benchmark" && (warmups > 5 || repeats < 2 || repeats > 20)) {
      throw std::invalid_argument("benchmark requires 0..5 warmups and 2..20 measured requests");
    }
    const auto config = specferry::models::opt::read_model_config(argv[2]);
    report.layers = number(argv[5]);
    const auto maximum = number(argv[6]);
    const auto tokens = read_tokens(argv[7]);
    const specferry::np101::ops::SamplingOptions sampling{argc == 9,
                                                          argc == 9 ? number(argv[8]) : 0};
    if (sampling.enabled && mode != "generate") {
      throw std::invalid_argument("sampling is only supported in generation mode");
    }
    if (tokens.size() > config.decoder.capacity || (mode == "teacher" && tokens.empty()) ||
        ((mode == "generate" || mode == "benchmark") && report.layers != config.decoder.layers) ||
        (mode == "benchmark" && !maximum)) {
      throw std::invalid_argument("request exceeds capacity or generation omits layers");
    }
    for (auto token : tokens) {
      config.validate_token(token);
    }
    specferry::np101::WeightStore weights(argv[1]);
    weights.verify();
    specferry::models::opt::validate_model_weights(weights, config, report.layers);
    report.phase = "initialize";
    report.save();
    report.sample_host("before_initialize");
    const auto start = Clock::now();
    specferry::np101::Context context;
    auto model = std::make_unique<Model>(context, weights, config, report.layers, sampling);
    report.initialize_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    report.sample_host("after_initialize");
    report.phase = mode;
    report.save();
    if (mode == "benchmark") {
      benchmark(*model, tokens, maximum, warmups, repeats, report);
    } else if (mode == "generate") {
      specferry::models::opt::Generation result;
      report.observe([&] {
        result = model->generate(tokens, maximum, [](std::int32_t token) {
          std::cout << "token " << token << std::endl;
        });
      });
      report.steps = result.consumed;
      save_generation(result, report.directory);
    } else if (mode == "teacher") {
      teacher(*model, tokens, "teacher", false, report);
      const auto length = model->length();
      const auto writes = model->cache_writes();
      unsigned rejected = 0;
      for (auto invalid : {-1, static_cast<int>(config.vocabulary)}) {
        try {
          model->consume(invalid);
        } catch (const std::invalid_argument &) {
          ++rejected;
        }
      }
      if (length == config.decoder.capacity) {
        try {
          model->consume(tokens.front());
        } catch (const std::out_of_range &) {
          ++rejected;
        }
      }
      if (rejected != 2 + unsigned(length == config.decoder.capacity) ||
          length != model->length() || writes != model->cache_writes()) {
        throw std::runtime_error("rejected model input changed cache state");
      }
      report.bounds_rejected = true;
      reset(*model, report);
      teacher(*model, tokens, "reset", false, report);
      reset(*model, report);
      teacher(*model, tokens, "final", true, report);
      report.cache_writes += model->cache_writes();
      model->close();
      model = std::make_unique<Model>(context, weights, config, report.layers);
      teacher(*model, tokens, "fresh", true, report);
    }
    report.cache_writes += model->cache_writes();
    report.phase = "release";
    report.save();
    model->close();
    model.reset();
    context.close();
    report.sample_host("after_release");
    report.released = true;
    report.phase = "complete";
    report.save("executed");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    try {
      report.save("failed");
    } catch (...) {
      // Preserve the original SDK/IO exception in stderr if the report cannot be written.
    }
    return 1;
  }
}
