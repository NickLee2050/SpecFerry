#include "models/opt/config.hpp"
#include "models/opt/graph_model.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace specferry;
using Clock = std::chrono::steady_clock;

unsigned number(const char *value) {
  const std::string text(value);
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("expected a positive integer");
  }
  const auto result = std::stoul(text);
  if (!result || result > 32) {
    throw std::out_of_range("experimental block/output limit must be 1..32");
  }
  return result;
}

std::vector<std::vector<std::int32_t>> read_requests(const fs::path &path,
                                                     const models::opt::ModelConfig &config) {
  std::ifstream input(path);
  std::vector<std::vector<std::int32_t>> requests;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::vector<std::int32_t> tokens;
    std::int32_t token;
    while (stream >> token) {
      config.validate_token(token);
      tokens.push_back(token);
    }
    if (!stream.eof() || tokens.empty() || tokens.size() > config.decoder.capacity) {
      throw std::invalid_argument("invalid experimental request token list");
    }
    requests.push_back(tokens);
  }
  if (!input.eof() || requests.empty() || requests.size() > 3) {
    throw std::invalid_argument("expected one to three request lines");
  }
  return requests;
}

void progress(const fs::path &directory, const std::string &phase, const char *status = "running") {
  const auto temporary = directory / "execution.json.tmp";
  std::ofstream out(temporary);
  const auto now =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  out << std::setprecision(17) << "{\"status\":\"" << status << "\",\"phase\":\"" << phase
      << "\",\"phase_started_at\":" << now << "}\n";
  out.close();
  if (!out) {
    throw std::runtime_error("cannot save experimental progress");
  }
  fs::rename(temporary, directory / "execution.json");
}

template <class T> void array(std::ostream &out, const std::vector<T> &values) {
  out << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    out << (index ? "," : "") << values[index];
  }
  out << ']';
}

void save_generation(const fs::path &path, const inference::Generation &result,
                     std::size_t launches, np101::TensorTransfers before,
                     np101::TensorTransfers after) {
  std::ofstream out(path);
  out << std::setprecision(17) << "{\"tokens\":";
  array(out, result.tokens);
  out << ",\"consumed\":" << result.consumed << ",\"stop_reason\":\"" << result.stop_reason
      << "\",\"prefill_seconds\":" << result.prefill_seconds
      << ",\"first_token_seconds\":" << result.first_token_seconds
      << ",\"total_seconds\":" << result.total_seconds << ",\"token_seconds\":";
  array(out, result.token_seconds);
  out << ",\"launches\":" << launches << ",\"uploads\":" << after.uploads - before.uploads
      << ",\"upload_bytes\":" << after.upload_bytes - before.upload_bytes
      << ",\"reads\":" << after.reads - before.reads
      << ",\"read_bytes\":" << after.read_bytes - before.read_bytes << "}\n";
  if (!out) {
    throw std::runtime_error("cannot save experimental generation");
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: specferry_opt_graph_generate DEPLOYMENT REQUEST OUTPUT BLOCK MAX_NEW "
                 "CACHE_READBACK EXPECTED_TOKENS\n"
                 "Experimental, unvalidated hardware path; CACHE_READBACK is prefix or none.\n";
    return 2;
  }
  const fs::path directory(argv[3]);
  try {
    const auto block = number(argv[4]), maximum = number(argv[5]);
    const std::string readback(argv[6]);
    if (readback != "prefix" && readback != "none") {
      throw std::invalid_argument("cache readback must be prefix or none");
    }
    const auto config = models::opt::read_model_config(argv[2]);
    const auto requests = read_requests(fs::path(argv[2]) / "requests.txt", config);
    const auto expected = read_requests(argv[7], config);
    if (expected.size() != requests.size()) {
      throw std::invalid_argument("expected token rows must match request rows");
    }
    for (const auto &tokens : expected) {
      if (tokens.size() > maximum) {
        throw std::invalid_argument("expected tokens exceed generation limit");
      }
    }
    np101::WeightStore weights(argv[1]);
    weights.verify();
    fs::create_directories(directory);
    progress(directory, "initialize");
    np101::SdkTimings timings(directory);
    np101::sample_host_memory(directory, "before_initialize");
    const auto start = Clock::now();
    np101::Context context;
    models::opt::GraphModel model(context, weights, config, block);
    const auto initialize = std::chrono::duration<double>(Clock::now() - start).count();
    const auto weight_bytes = model.weight_bytes();
    {
      std::ofstream initialized(directory / "initialization.json");
      initialized << std::setprecision(17) << "{\"initialize_seconds\":" << initialize
                  << ",\"weight_payload_bytes\":" << weight_bytes << "}\n";
      if (!initialized) {
        throw std::runtime_error("cannot save initialization observation");
      }
    }
    np101::sample_host_memory(directory, "after_initialize");
    bool tokens_match = true;

    for (unsigned index = 0; index < requests.size(); ++index) {
      const auto name = "measured." + std::to_string(index);
      progress(directory, name);
      np101::TimingLabel request(np101::TimingField::Request, name);
      const auto before = np101::tensor_transfers();
      const auto launches = model.launches();
      const auto result = model.generate(requests[index], maximum);
      save_generation(directory / (name + ".json"), result, model.launches() - launches, before,
                      np101::tensor_transfers());
      np101::sample_host_memory(directory, name);
      // Diagnostics run after generation, outside its transfer/time contract.
      if (readback == "prefix") {
        np101::TimingLabel diagnostic_request(np101::TimingField::Request,
                                              "diagnostic." + std::to_string(index));
        for (unsigned layer = 0; layer < config.decoder.layers; ++layer) {
          for (bool values : {false, true}) {
            auto bytes = model.read_cache(layer, values);
            auto path = directory / (name + ".layer." + std::to_string(layer) +
                                     (values ? ".values.bin" : ".keys.bin"));
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            if (!out) {
              throw std::runtime_error("cannot save cache diagnostic");
            }
          }
        }
      }
      timings.save();
      // Keep the first failing request's KV, then release normally. Further
      // requests cannot pass the gate once the independent token oracle fails.
      if (result.tokens != expected[index]) {
        tokens_match = false;
        std::cerr << "Request " << index << " token mismatch; expected ";
        array(std::cerr, expected[index]);
        std::cerr << ", got ";
        array(std::cerr, result.tokens);
        std::cerr << ". Skipping remaining requests and releasing the model.\n";
        break;
      }
    }
    const auto launches = model.launches();
    progress(directory, "release");
    {
      np101::TimingLabel phase(np101::TimingField::Phase, "release");
      model.close();
      context.close();
    }
    timings.save();
    np101::sample_host_memory(directory, "after_release");
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) {
      throw std::runtime_error("cannot read host resource observation");
    }
    std::ofstream out(directory / "run.json");
    out << std::setprecision(17) << "{\"status\":\""
        << (tokens_match ? "executed" : "numerical_mismatch")
        << "\",\"released\":true,\"layers\":" << config.decoder.layers
        << ",\"prefill_block\":" << block << ",\"capacity\":" << config.decoder.capacity
        << ",\"launches\":" << launches << ",\"weight_payload_bytes\":" << weight_bytes
        << ",\"initialize_seconds\":" << initialize << ",\"host_peak_rss_kib\":" << usage.ru_maxrss
        << "}\n";
    out.close();
    if (!out) {
      throw std::runtime_error("cannot save experimental lifecycle");
    }
    progress(directory, "complete", tokens_match ? "executed" : "failed");
    return tokens_match ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    if (fs::is_directory(directory)) {
      try {
        progress(directory, "failed", "failed");
      } catch (...) {
        // Preserve the original SDK/IO error.
      }
    }
    return 1;
  }
}
