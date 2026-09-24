#include "models/opt/model.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using specferry::models::opt::Model;

struct Report {
  fs::path directory;
  unsigned layers = 0, steps = 0;
};

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
    for (const auto *name : {"keys", "values"}) {
      save("layer." + std::to_string(layer) + "." + name, model.read(name, layer));
    }
  }
}

void teacher(Model &model, const std::vector<std::int32_t> &tokens, const std::string &sequence,
             bool final_only, Report &report) {
  std::ofstream predictions(report.directory / (sequence + ".tokens.txt"));
  for (unsigned index = 0; index < tokens.size(); ++index) {
    std::cout << sequence << " step " << index << std::endl;
    const bool select = !final_only || index + 1 == tokens.size();
    model.consume(tokens[index]);
    ++report.steps;
    if (select) {
      predictions << model.predict() << '\n';
      capture(model, sequence + "." + std::to_string(index) + ".", report.layers, report.directory);
    }
  }
  if (!predictions) {
    throw std::runtime_error("cannot save predictions");
  }
}

void reset(Model &model) {
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
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: np101_generation_check DEPLOYMENT FIXTURE OUTPUT LAYERS STEPS\n";
    return 2;
  }
  try {
    const auto layers = specferry::np101::parse_shape(argv[4]);
    const auto steps = specferry::np101::parse_shape(argv[5]);
    if (layers.size() != 1 || steps.size() != 1) {
      throw std::invalid_argument("expected one layer count and one step count");
    }
    Report report{argv[3], layers[0]};
    fs::create_directories(report.directory);
    specferry::np101::SdkTimings timings(report.directory);
    const auto config = specferry::models::opt::read_model_config(argv[2]);
    std::ifstream source(fs::path(argv[2]) / "tokens.txt");
    std::vector<std::int32_t> tokens;
    std::int32_t token;
    while (source >> token) {
      config.validate_token(token);
      tokens.push_back(token);
    }
    if (!source.eof() || tokens.empty() || tokens.size() != steps[0] ||
        tokens.size() > config.decoder.capacity) {
      throw std::invalid_argument("invalid teacher token sequence");
    }
    specferry::np101::WeightStore weights(argv[1]);
    weights.verify();
    specferry::np101::Context context;
    auto model = std::make_unique<Model>(context, weights, config, report.layers);
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
    if (rejected != 2 + unsigned(length == config.decoder.capacity) || length != model->length() ||
        writes != model->cache_writes()) {
      throw std::runtime_error("rejected input changed model state");
    }
    reset(*model);
    teacher(*model, tokens, "reset", false, report);
    reset(*model);
    teacher(*model, tokens, "final", true, report);
    model->close();
    model = std::make_unique<Model>(context, weights, config, report.layers);
    teacher(*model, tokens, "fresh", true, report);
    model->close();
    model.reset();
    context.close();
    std::ofstream out(report.directory / "execution.json");
    out << "{\"status\":\"executed\",\"phase\":\"complete\",\"released\":true,"
           "\"bounds_rejected\":true,\"reset_rejected_stale\":true,\"steps\":"
        << report.steps << "}\n";
    return out ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
