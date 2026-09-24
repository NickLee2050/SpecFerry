#include "inference/generation_io.hpp"
#include "models/opt/model.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/weights.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
unsigned number(const char *text) {
  const std::string value(text);
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("expected an unsigned integer");
  }
  const auto parsed = std::stoull(value);
  if (parsed > std::numeric_limits<unsigned>::max()) {
    throw std::out_of_range("integer exceeds uint32");
  }
  return unsigned(parsed);
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 7 && argc != 8) {
    std::cerr << "usage: specferry_opt_generate DEPLOYMENT REQUEST OUTPUT MAX_NEW WARMUPS REPEATS "
                 "[SEED]\n";
    return 2;
  }
  try {
    const std::filesystem::path output(argv[3]);
    std::filesystem::create_directories(output);
    const auto maximum = number(argv[4]), warmups = number(argv[5]), repeats = number(argv[6]);
    if (warmups > 5 || repeats < 1 || repeats > 20) {
      throw std::invalid_argument("expected 0..5 warmups and 1..20 repetitions");
    }
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
    std::cout << "Initializing model" << std::endl;
    specferry::np101::Context context;
    specferry::models::opt::Model model(context, weights, config, config.decoder.layers,
                                        {argc == 8, argc == 8 ? number(argv[7]) : 0});
    const auto initialized =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (unsigned i = 0; i < warmups + repeats; ++i) {
      const auto name = std::string(i < warmups ? "warmup." : "measured.") +
                        std::to_string(i < warmups ? i : i - warmups);
      std::cout << name << std::endl;
      const auto result = model.generate(prompt, maximum);
      std::ofstream out(output / (name + ".json"));
      out << '{';
      specferry::inference::write_generation(out, result);
      out << "}\n";
      if (!out) {
        throw std::runtime_error("cannot save generated output");
      }
    }
    std::cout << "Releasing model" << std::endl;
    model.close();
    context.close();
    std::ofstream out(output / "execution.json");
    out << "{\"status\":\"executed\",\"phase\":\"complete\",\"released\":true,\"initialize_"
           "seconds\":"
        << initialized << "}\n";
    return out ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
