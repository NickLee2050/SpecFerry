#include "binary_io.hpp"
#include "inference/generation_io.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry;
using namespace np101;
using namespace np101::ops;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

std::vector<std::int32_t> read_tokens(const fs::path &path) {
  std::ifstream input(path);
  std::vector<std::int32_t> tokens;
  std::int32_t token;
  while (input >> token) {
    tokens.push_back(token);
  }
  if (!input.eof() || tokens.empty()) {
    throw std::runtime_error("invalid token fixture");
  }
  return tokens;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: np101_compiled_model_check CONTRACT REQUEST OUTPUT\n";
    return 2;
  }
  try {
    const fs::path output(argv[3]), request(argv[2]);
    fs::create_directories(output);
    std::cout << std::unitbuf;
    const auto phase = [&](const std::string &name) {
      const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
      testing::write_text(output / "phase.json",
                          "{\"phase\":\"" + name + "\",\"unix_us\":" + std::to_string(now) + "}\n");
      std::cout << "phase=" << name << '\n';
    };
    const auto prompt = read_tokens(request / "requests.txt");
    const auto expected = read_tokens(request / "expected.txt");
    if (prompt.size() != 17 || expected.size() != 16) {
      throw std::runtime_error("full-model experiment requires the accepted 17/16 fixture");
    }
    CompiledGraphSpec spec;
    std::ifstream contract(argv[1]);
    unsigned inputs_count, outputs_count;
    contract >> spec.path >> spec.constant_bytes >> inputs_count >> outputs_count;
    std::vector<std::string> input_names, output_names;
    const auto ports = [&](unsigned count, auto &ports, auto &names) {
      for (unsigned i = 0; i < count; ++i) {
        std::string name, type;
        unsigned rank;
        contract >> name >> type >> rank;
        if (rank == 0 || rank > 3) {
          throw std::runtime_error("invalid compiled port rank");
        }
        Shape shape(rank);
        for (auto &axis : shape) {
          contract >> axis;
        }
        DataType dtype;
        if (type == "F16") {
          dtype = DataType::Float16;
        } else if (type == "F32") {
          dtype = DataType::Float32;
        } else if (type == "I32") {
          dtype = DataType::Int32;
        } else {
          throw std::runtime_error("invalid compiled port dtype");
        }
        ports.push_back({dtype, shape});
        names.push_back(name);
      }
    };
    ports(inputs_count, spec.inputs, input_names);
    ports(outputs_count, spec.outputs, output_names);
    std::string extra;
    if (!contract || contract >> extra || inputs_count != 52 || outputs_count != 50) {
      throw std::runtime_error("invalid compiled full-model contract");
    }
    spec.graph_version = {1, 1, 35};
    SdkTimings timings(output);
    const auto started = Clock::now();
    phase("Context");
    Context context;
    GraphBuilder model(context, 104, 1);
    std::vector<Tensor> inputs;
    for (const auto &port : spec.inputs) {
      inputs.push_back(model.tensor(port));
    }
    const auto outputs = model.compiled(spec, inputs);
    phase("full-model.SetupVerify");
    model.compile(inputs, outputs);
    GraphBuilder commit(context, 96, 48);
    std::vector<Tensor> sources, destinations;
    for (unsigned layer = 0; layer < 24; ++layer) {
      for (const std::string part : {"key", "value"}) {
        const auto prefix = "layer" + std::to_string(layer) + ".";
        auto ip = std::find(input_names.begin(), input_names.end(), prefix + "past_" + part);
        auto op = std::find(output_names.begin(), output_names.end(), prefix + "present_" + part);
        if (ip == input_names.end() || op == output_names.end()) {
          throw std::runtime_error("missing KV contract port");
        }
        auto destination = commit.share(model, inputs[ip - input_names.begin()]);
        auto source = commit.share(model, outputs[op - output_names.begin()]);
        commit.node(VSI_NN_OP_DATACONVERT, {source}, destination);
        sources.push_back(source);
        destinations.push_back(destination);
      }
    }
    phase("kv-commit.SetupVerify");
    commit.compile(sources, destinations);
    const auto initialization = std::chrono::duration<double>(Clock::now() - started).count();
    const auto output_index = [&](const std::string &name) {
      auto found = std::find(output_names.begin(), output_names.end(), name);
      if (found == output_names.end()) {
        throw std::runtime_error("missing output port");
      }
      return std::size_t(found - output_names.begin());
    };
    const auto selected_index = output_index("selected");
    unsigned position = 0;
    std::size_t graph_runs = 0;
    const auto run = [&](GraphBuilder &graph, const std::string &role) {
      const SdkScope scope("role", role);
      check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.graph.get()); }),
            ("run " + role).c_str());
      ++graph_runs;
    };
    const auto reset = [&] {
      for (std::size_t i = 0; i < input_names.size(); ++i) {
        if (input_names[i].find(".past_") != std::string::npos) {
          upload_tensor(model.graph, inputs[i].id,
                        std::vector<std::uint8_t>(inputs[i].spec.bytes()));
        }
      }
      position = 0;
    };
    const auto consume = [&](std::int32_t token) {
      if (token < 0 || token >= 50272 || position >= 64) {
        throw std::out_of_range("token or capacity");
      }
      std::vector<std::int32_t> indices;
      for (unsigned head = 0; head < 16; ++head) {
        indices.push_back(head);
        indices.push_back(position);
      }
      std::vector<float> mask(64, -1e9f);
      std::fill_n(mask.begin(), position + 1, 0.0f);
      for (unsigned i = 0; i < inputs.size(); ++i) {
        const auto &name = input_names[i];
        if (name.find(".past_") != std::string::npos) {
          continue;
        }
        std::vector<std::uint8_t> bytes;
        if (name == "token") {
          bytes = testing::as_bytes(std::vector<std::int32_t>{token});
        } else if (name == "position") {
          bytes = testing::as_bytes(std::vector<std::int32_t>{std::int32_t(position)});
        } else if (name == "kv_indices") {
          bytes = testing::as_bytes(indices);
        } else if (name == "attention_mask") {
          bytes = testing::as_bytes(mask);
        } else {
          throw std::runtime_error("unknown control input");
        }
        upload_tensor(model.graph, inputs[i].id, bytes);
      }
      run(model, "full-model");
      run(commit, "kv-commit");
      ++position;
    };
    const auto predict = [&] {
      const auto bytes = read_tensor(model.graph, outputs[selected_index].id);
      if (bytes.size() != 4) {
        throw std::runtime_error("invalid token output size");
      }
      std::int32_t token;
      std::memcpy(&token, bytes.data(), 4);
      return token;
    };
    for (const std::string name : {"warmup", "measured"}) {
      phase(name);
      const SdkScope request_scope("request", name);
      const auto generation =
          inference::generate_tokens({50272, 64, 2, 2}, {reset, consume, predict}, prompt, 16);
      std::ofstream out(output / (name + ".json"));
      out << '{';
      inference::write_generation(out, generation);
      out << "}\n";
      out.close();
      if (generation.tokens != expected) {
        throw std::runtime_error("generated tokens differ from reference");
      }
    }
    phase("final.MappedHost");
    for (const std::string name : {"logits", "layer23.present_key", "layer23.present_value"}) {
      const auto tensor = outputs[output_index(name)];
      testing::write_bytes(output / (name + ".bin"),
                           read_tensor(model.graph, tensor.id, TensorReadback::MappedHost));
    }
    const auto constants = context.memory().peak(true),
               mutable_bytes = context.memory().peak(false);
    phase("release");
    commit.graph.close();
    model.graph.close();
    if (context.memory().live(true) || context.memory().live(false)) {
      throw std::runtime_error("payload leaked");
    }
    context.close();
    timings.flush();
    std::ofstream report(output / "execution.json");
    report << "{\"status\":\"executed\",\"phase\":\"complete\",\"released\":true,\"initialize_"
              "seconds\":"
           << initialization << ",\"graph_runs\":" << graph_runs
           << ",\"const_peak_bytes\":" << constants << ",\"nonconst_peak_bytes\":" << mutable_bytes
           << ",\"const_live_bytes\":0,\"nonconst_live_bytes\":0}\n";
    phase("complete");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
