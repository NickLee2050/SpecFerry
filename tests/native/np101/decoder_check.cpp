#include "binary_io.hpp"
#include "models/opt/decoder.hpp"
#include "models/qwen3_5/decoder.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace specferry;

struct DecoderCase {
  fs::path output;
  unsigned count, capacity;
  std::size_t uploads = 1, control_bytes = 0;
  std::vector<std::vector<std::uint8_t>> inputs{};
  std::map<unsigned, std::vector<std::string>> fields{};
  bool coarse = false;
  std::vector<double> sequence_seconds{};
};

bool checkpoint(unsigned step, unsigned count) {
  return step == 0 || step == 1 || step == 3 || step == 7 || step == 31 || step == 255 ||
         step == 511 || step + 1 == count;
}

template <class Exception, class Action> void rejects(Action action) {
  try {
    action();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error("decoder accepted an invalid operation");
}

template <class Model>
void sequence(Model &model, DecoderCase &test, const std::string &name, unsigned count,
              bool final_only) {
  std::cout << name << ": " << count << " steps" << std::endl;
  double seconds = 0;
  for (unsigned step = 0; step < count; ++step) {
    const auto before = np101::tensor_transfers();
    const auto start = std::chrono::steady_clock::now();
    model.step(test.inputs[step]);
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto after = np101::tensor_transfers();
    if (after.uploads - before.uploads != test.uploads ||
        after.upload_bytes - before.upload_bytes != test.inputs[step].size() + test.control_bytes ||
        after.reads != before.reads || model.length() != step + 1) {
      throw std::runtime_error("unexpected decoder position or host transfers");
    }
    if ((test.coarse || final_only) ? step + 1 != count : !checkpoint(step, count)) {
      continue;
    }
    for (const auto &[layer, fields] : test.fields) {
      for (const auto &field : fields) {
        const auto data = model.read(layer, field);
        const auto filename = name + ".layer." + std::to_string(layer) + "." + field + "." +
                              std::to_string(step) + ".bin";
        testing::write_bytes(test.output / filename, data);
      }
    }
  }
  test.sequence_seconds.push_back(seconds);
}

template <class Factory> void trajectories(Factory create, DecoderCase &test) {
  auto model = create();
  sequence(*model, test, "zero", test.count, false);
  rejects<std::invalid_argument>([&] { model->step({}); });
  if (test.count == test.capacity) {
    rejects<std::out_of_range>([&] { model->step(test.inputs.front()); });
  }
  if (model->length() != test.count) {
    throw std::runtime_error("invalid input changed decoder position");
  }
  for (const auto *name : {"reset", "final"}) {
    model->reset();
    rejects<std::logic_error>([&] { model->read(test.fields.begin()->first, "output"); });
    if (model->length()) {
      throw std::runtime_error("reset retained the old position");
    }
    const bool final_only = std::string(name) == "final";
    sequence(*model, test, name, std::min(final_only ? 32U : 8U, test.count), final_only);
  }
  model->close();
  model = create();
  sequence(*model, test, "fresh", std::min(8U, test.count), false);
  model->close();
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: np101_decoder_check opt|qwen DEPLOYMENT FIXTURE OUTPUT STEPS\n";
    return 2;
  }
  try {
    const std::string profile(argv[1]);
    const fs::path fixture(argv[3]), output(argv[4]);
    const auto counts = np101::parse_shape(argv[5]);
    if (counts.size() != 1 || counts[0] < 2) {
      throw std::invalid_argument("invalid step count");
    }
    std::vector<unsigned> layers;
    std::ifstream selected(fixture / "layers.txt");
    unsigned layer;
    while (selected >> layer) {
      layers.push_back(layer);
    }
    if (!selected.eof() || layers.empty() || layers.size() > 4) {
      throw std::invalid_argument("expected one to four layer indices");
    }
    fs::create_directories(output);
    DecoderCase test{output, counts[0], 0};
    np101::WeightStore weights(argv[2]);
    weights.verify();
    // Input/config validation occurs before the factory opens the SDK context.
    const auto load_inputs = [&](const np101::TensorSpec &spec) {
      if (test.count > test.capacity) {
        throw std::invalid_argument("steps exceed capacity");
      }
      for (unsigned i = 0; i < test.count; ++i) {
        test.inputs.push_back(
            testing::read_bytes(fixture, "input." + std::to_string(i) + ".bin", spec.bytes()));
      }
    };
    np101::SdkTimings timings(output);
    std::size_t const_peak = 0, nonconst_peak = 0;
    const auto account = [&](np101::Context &context) {
      if (context.memory().live(true) || context.memory().live(false)) {
        throw std::runtime_error("decoder payload remains after release");
      }
      const_peak = std::max(const_peak, context.memory().peak(true));
      nonconst_peak = std::max(nonconst_peak, context.memory().peak(false));
    };
    if (profile == "opt") {
      const auto config = models::opt::read_config(fixture / "components.txt");
      test.capacity = config.capacity;
      test.uploads += layers.size();
      test.control_bytes = 4 * layers.size();
      for (auto index : layers) {
        test.fields[index] = {"query",          "key",
                              "value",          "probabilities",
                              "mixer",          "attention_residual",
                              "attention_norm", "fc1",
                              "activation",     "mlp",
                              "ffn_residual",   "output",
                              "keys",           "values"};
      }
      load_inputs(config.hidden_spec());
      models::opt::validate_weights(weights, config, layers);
      np101::Context context;
      trajectories(
          [&] {
            return std::make_unique<models::opt::DecoderSlice>(context, weights, config, layers);
          },
          test);
      account(context);
      context.close();
    } else if (profile == "qwen") {
      const auto config = models::qwen3_5::read_config(fixture / "components.txt");
      std::map<unsigned, models::qwen3_5::CompiledBlock> compiled_mixers;
      std::ifstream mixer_file(fixture / "compiled-mixer-package.txt");
      if (mixer_file) {
        std::string package, extra;
        if (!(mixer_file >> std::quoted(package)) || mixer_file >> extra) {
          throw std::invalid_argument("compiled mixer fixture requires one package path");
        }
        compiled_mixers = models::qwen3_5::load_compiled_mixers(package, weights, config);
      }
      auto backend = np101::ops::LinearBackend::MatMul;
      std::ifstream backend_file(fixture / "feed-forward-backend.txt");
      std::string name, extra;
      if (backend_file) {
        if (!(backend_file >> name) || backend_file >> extra ||
            (name != "fcl" && name != "matmul")) {
          throw std::invalid_argument("invalid Qwen feed-forward backend");
        }
        backend = name == "fcl" ? np101::ops::LinearBackend::FullyConnected
                                : np101::ops::LinearBackend::MatMul;
      }
      test.coarse = fs::exists(fixture / "coarse-readback.txt");
      const auto readback =
          test.coarse ? np101::TensorReadback::MappedHost : np101::TensorReadback::SdkCopy;
      test.capacity = config.kv.capacity;
      for (auto index : layers) {
        auto &fields = test.fields[index];
        fields = test.coarse ? std::vector<std::string>{"output"}
                             : std::vector<std::string>{"normalized", "mixer", "residual",
                                                        "post_norm",  "mlp",   "output"};
        if (config.mixer(index) == models::qwen3_5::MixerKind::Attention) {
          fields.insert(fields.end(), {"keys", "values"});
          test.uploads += 2;
          test.control_bytes += 8;
        } else {
          fields.insert(fields.end(), {"recurrent", "convolution"});
        }
      }
      load_inputs(config.hidden_spec());
      std::unique_ptr<np101::Context> context;
      trajectories(
          [&] {
            if (context) {
              account(*context);
              context->close();
              context.reset();
            }
            context = std::make_unique<np101::Context>();
            return std::make_unique<models::qwen3_5::DecoderGroup>(
                *context, weights, config, layers, backend, readback, compiled_mixers);
          },
          test);
      account(*context);
      context->close();
    } else {
      throw std::invalid_argument("unknown model profile");
    }
    std::ofstream out(output / "execution.json");
    out << "{\"status\":\"executed\",\"phase\":\"complete\",\"released\":true,\"sequences\":4,"
           "\"steps\":"
        << test.count + std::min(32U, test.count) + 2 * std::min(8U, test.count)
        << ",\"const_peak_bytes\":" << const_peak << ",\"nonconst_peak_bytes\":" << nonconst_peak
        << ",\"const_live_bytes\":0,\"nonconst_live_bytes\":0,\"sequence_seconds\":[";
    for (std::size_t index = 0; index < test.sequence_seconds.size(); ++index) {
      out << (index ? "," : "") << test.sequence_seconds[index];
    }
    out << "]}\n";
    return out ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
