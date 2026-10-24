#include "inference/generation_io.hpp"
#include "models/opt/model.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"
#include "platform/runtime_profile.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: specferry_opt_generate DEPLOYMENT REQUEST OUTPUT\n";
    return 2;
  }
  try {
    const std::filesystem::path output(argv[3]);
    std::filesystem::create_directories(output);
    specferry::platform::write_runtime_profile(output);
    const auto options = specferry::inference::read_run_options(argv[2], false);
    const auto config = specferry::models::opt::read_model_config(argv[2]);
    std::ifstream input(std::filesystem::path(argv[2]) / "tokens.txt");
    std::vector<std::int32_t> prompt;
    std::int32_t token;
    while (input >> token) {
      config.validate_token(token);
      prompt.push_back(token);
    }
    if (!input.eof() || prompt.empty() || prompt.size() > config.decoder.capacity) {
      throw std::invalid_argument("invalid prompt tokens or capacity");
    }
    specferry::np101::WeightStore weights(argv[1]);
    weights.verify();
    specferry::np101::SdkTimings timings(output);
    const auto start = std::chrono::steady_clock::now();
    std::unique_ptr<specferry::np101::Context> context;
    std::unique_ptr<specferry::models::opt::Model> model;
    {
      const specferry::np101::SdkScope phase("phase", "initialize");
      context = std::make_unique<specferry::np101::Context>();
      model = std::make_unique<specferry::models::opt::Model>(
          *context, weights, config, config.decoder.layers,
          specferry::np101::ops::SamplingOptions{options.sampling, options.seed});
    }
    const auto initialized =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (unsigned i = 0; i < options.warmups + options.repeats; ++i) {
      const auto name = std::string(i < options.warmups ? "warmup." : "measured.") +
                        std::to_string(i < options.warmups ? i : i - options.warmups);
      const specferry::np101::SdkScope request("request", name);
      const auto before = specferry::np101::tensor_transfers();
      const auto result = model->generate(prompt, options.maximum);
      const auto after = specferry::np101::tensor_transfers();
      std::ofstream out(output / (name + ".json"));
      out << '{';
      specferry::inference::write_generation(out, result);
      out << ",\"uploads\":" << after.uploads - before.uploads
          << ",\"upload_bytes\":" << after.upload_bytes - before.upload_bytes
          << ",\"reads\":" << after.reads - before.reads
          << ",\"read_bytes\":" << after.read_bytes - before.read_bytes << "}\n";
      if (!out) {
        throw std::runtime_error("cannot save generated output");
      }
    }
    const auto release_started = std::chrono::steady_clock::now();
    {
      const specferry::np101::SdkScope phase("phase", "release");
      model->close();
      context->close();
    }
    const auto release_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - release_started).count();
    timings.flush();
    std::ofstream out(output / "execution.json");
    out << "{\"status\":\"executed\",\"phase\":\"complete\",\"released\":true,\"initialize_"
           "seconds\":"
        << initialized << ",\"release_seconds\":" << release_seconds
        << ",\"const_live_bytes\":" << context->memory().live(true)
        << ",\"const_peak_bytes\":" << context->memory().peak(true)
        << ",\"nonconst_live_bytes\":" << context->memory().live(false)
        << ",\"nonconst_peak_bytes\":" << context->memory().peak(false) << "}\n";
    return out ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
