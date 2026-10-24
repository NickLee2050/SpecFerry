#include "binary_io.hpp"
#include "inference/generation_io.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

void phase(const std::string &name) {
  std::cout << std::setprecision(17) << now() << " phase=" << name << std::endl;
}

void require_memory_reserve() {
  std::ifstream input("/proc/meminfo");
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("MemAvailable:", 0) == 0) {
      if (std::stoull(line.substr(13)) < 192 * 1024) {
        throw std::runtime_error("Linux memory reserve below 192 MiB");
      }
      return;
    }
  }
  throw std::runtime_error("cannot inspect Linux memory reserve");
}

struct ModelContract {
  CompiledGraphSpec spec;
  std::vector<std::string> inputs, outputs;
};

ModelContract read_contract(const fs::path &package) {
  ModelContract result;
  auto &spec = result.spec;
  std::ifstream contract(package / "contract.txt");
  unsigned inputs_count = 0, outputs_count = 0;
  contract >> spec.path >> spec.constant_bytes >> inputs_count >> outputs_count;
  if (!contract || inputs_count != 52 || outputs_count != 50) {
    throw std::runtime_error("invalid compiled full-model header");
  }
  auto &input_names = result.inputs;
  auto &output_names = result.outputs;
  const auto ports = [&](unsigned count, auto &ports, auto &names) {
    for (unsigned i = 0; i < count; ++i) {
      std::string name, type;
      unsigned rank = 0;
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

  return result;
}

void check_partial_write(Graph &graph, vsi_nn_tensor_id_t id,
                         const std::vector<std::uint8_t> &zeroes) {
  // Check a nonzero patch origin before trusting partial writes in generation.
  // The other 63 positions, including all head strides, must stay bit-identical.
  std::vector<float> pattern(16 * 64);
  for (unsigned i = 0; i < pattern.size(); ++i) {
    pattern[i] = float(i % 31 + 1) / 32;
  }
  const auto patch = as_bytes(pattern);
  upload_tensor_region(graph, id, {0, 63, 0}, {64, 64, 16}, patch);
  auto wanted = zeroes;
  for (unsigned head = 0; head < 16; ++head) {
    std::memcpy(wanted.data() + (head * 64 + 63) * 64 * sizeof(float),
                patch.data() + head * 64 * sizeof(float), 64 * sizeof(float));
  }
  if (read_tensor(graph, id, TensorReadback::MappedHost) != wanted) {
    throw std::runtime_error("partial write changed unrelated bytes or misplaced the patch");
  }
  upload_tensor(graph, id, zeroes);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: np101_compiled_model_transfer_check PACKAGE OUTPUT\n";
    return 2;
  }
  try {
    const fs::path package(argv[1]), output(argv[2]), fixture = package / "fixture";
    unsigned request_index = 0;
    unsetenv("SPECFERRY_SDK_TIMING");
    std::ofstream trace(output / "rungraph-calls.tsv");
    trace << "event\tsequence\tgroup\tindex\tmonotonic_seconds\tstatus\n";
    unsigned sequence = 0;
    const auto run = [&](GraphBuilder &graph, const std::string &role, unsigned step) {
      const auto id = ++sequence;
      const auto record = [&](const char *event, const std::string &status) {
        trace << event << '\t' << id << '\t' << request_index << ':' << role << '\t' << step << '\t'
              << std::setprecision(17) << now() << '\t' << status << std::endl;
        if (!trace) {
          throw std::runtime_error("RunGraph trace write failed");
        }
      };
      record("begin", "pending");
      const auto started = now();
      const auto status = vsi_nn_RunGraph(graph.graph.get());
      const auto seconds = now() - started;
      record("end", std::to_string(status));
      check(status, "RunGraph");
      return seconds;
    };
    require_memory_reserve();
    const auto prepare_start = now();
    phase("CreateContext");
    Context context;
    const auto contract = read_contract(package);
    const auto &spec = contract.spec;
    const auto &input_names = contract.inputs;
    const auto &output_names = contract.outputs;

    GraphBuilder model(context, 104, 1);
    std::vector<Tensor> inputs;
    for (const auto &port : spec.inputs) {
      inputs.push_back(model.tensor(port));
    }
    const auto outputs = model.compiled(spec, inputs);
    phase("full-model.SetupVerify.begin");
    model.compile(inputs, outputs);
    phase("full-model.SetupVerify.end");
    require_memory_reserve();
    const auto preparation_seconds = now() - prepare_start;
    const auto named_index = [](const auto &names, const std::string &name) {
      const auto found = std::find(names.begin(), names.end(), name);
      if (found == names.end()) {
        throw std::runtime_error("missing port: " + name);
      }
      return std::size_t(found - names.begin());
    };

    struct Cache {
      std::size_t input, output;
      std::vector<std::uint8_t> data;
    };

    std::vector<Cache> caches;
    for (unsigned layer = 0; layer < 24; ++layer) {
      for (const std::string kind : {"key", "value"}) {
        const auto prefix = "layer" + std::to_string(layer) + ".";
        const auto input = named_index(input_names, prefix + "past_" + kind);
        const auto output = named_index(output_names, prefix + "present_" + kind);
        caches.push_back({input, output, std::vector<std::uint8_t>(inputs[input].spec.bytes(), 0)});
      }
    }
    const auto logits_index = named_index(output_names, "logits");
    const auto selected_index = named_index(output_names, "selected");
    unsigned position = 0;
    double graph_seconds = 0, upload_seconds = 0, readback_seconds = 0;
    bool incremental = false, patch_checked = false;
    std::vector<std::uint8_t> scratch(16 * 64 * sizeof(float));
    std::vector<std::uint8_t> selected_bytes(sizeof(std::int32_t));
    const std::vector<std::uint32_t> origin{0, 0, 0}, whole{64, 64, 16};
    std::vector<std::uint32_t> slot_start{0, 0, 0}, slot_end{64, 1, 16};
    std::vector<std::int32_t> indices(32);
    std::vector<float> mask(64);
    std::vector<std::uint8_t> token_bytes(4), position_bytes(4), index_bytes(128), mask_bytes(256);
    const auto token_input = inputs[named_index(input_names, "token")].id;
    const auto position_input = inputs[named_index(input_names, "position")].id;
    const auto indices_input = inputs[named_index(input_names, "kv_indices")].id;
    const auto mask_input = inputs[named_index(input_names, "attention_mask")].id;
    TensorTransfers transfer_start;
    // Same real request: tracing off/on/off, then incremental KV bracketed by full copies.
    const std::vector<std::string> variants{"full-light-before", "full-traced", "full-light-middle",
                                            "incremental", "full-light-after"};
    const auto expected = [&] {
      std::ifstream input(fixture / "expected.txt");
      std::vector<std::int32_t> values;
      std::int32_t value;
      while (input >> value) {
        values.push_back(value);
      }
      if (!input.eof() || values.size() != 16) {
        throw std::runtime_error("expected 16 validated output tokens");
      }
      return values;
    }();
    const auto reset = [&] {
      position = 0;
      for (auto &cache : caches) {
        std::fill(cache.data.begin(), cache.data.end(), 0);
        upload_tensor(model.graph, inputs[cache.input].id, cache.data);
      }
      if (incremental && !patch_checked) {
        check_partial_write(model.graph, inputs[caches.front().input].id, caches.front().data);
        patch_checked = true;
      }
      graph_seconds = upload_seconds = readback_seconds = 0;
      transfer_start = tensor_transfers();
    };
    const auto consume = [&](std::int32_t token) {
      if (token < 0 || token >= 50272 || position >= 64) {
        throw std::out_of_range("token or capacity");
      }
      for (unsigned head = 0; head < 16; ++head) {
        indices[head * 2] = head;
        indices[head * 2 + 1] = position;
      }
      std::fill(mask.begin(), mask.end(), -1e9f);
      std::fill_n(mask.begin(), position + 1, 0.0f);
      const auto signed_position = std::int32_t(position);
      std::memcpy(token_bytes.data(), &token, sizeof(token));
      std::memcpy(position_bytes.data(), &signed_position, sizeof(signed_position));
      std::memcpy(index_bytes.data(), indices.data(), index_bytes.size());
      std::memcpy(mask_bytes.data(), mask.data(), mask_bytes.size());
      auto io_start = now();
      if (!incremental) {
        for (const auto &cache : caches) {
          upload_tensor(model.graph, inputs[cache.input].id, cache.data);
        }
      }
      upload_tensor(model.graph, token_input, token_bytes);
      upload_tensor(model.graph, position_input, position_bytes);
      upload_tensor(model.graph, indices_input, index_bytes);
      upload_tensor(model.graph, mask_input, mask_bytes);
      upload_seconds += now() - io_start;
      graph_seconds += run(model, "full-model", position);
      slot_start[1] = position;
      slot_end[1] = position + 1;
      for (auto &cache : caches) {
        io_start = now();
        if (incremental) {
          read_tensor_region_into(model.graph, outputs[cache.output].id, slot_start, slot_end,
                                  scratch);
        } else {
          read_tensor_region_into(model.graph, outputs[cache.output].id, origin, whole, cache.data);
        }
        readback_seconds += now() - io_start;
        if (incremental) {
          io_start = now();
          upload_tensor_region(model.graph, inputs[cache.input].id, slot_start, slot_end, scratch);
          upload_seconds += now() - io_start;
        }
      }
      ++position;
    };
    const auto predict = [&] {
      const auto io_start = now();
      read_tensor_region_into(model.graph, outputs[selected_index].id, {0}, {1}, selected_bytes);
      readback_seconds += now() - io_start;
      std::int32_t token;
      std::memcpy(&token, selected_bytes.data(), sizeof(token));
      return token;
    };
    for (request_index = 0; request_index < variants.size(); ++request_index) {
      const auto name = "request" + std::to_string(request_index);
      incremental = variants[request_index] == "incremental";
      fs::create_directories(output / name);
      setenv("SPECFERRY_SDK_TIMING", request_index == 1 ? "calls" : "off", 1);
      SdkTimings request_timings(output / name);
      std::ifstream input(fixture / "request0.txt");
      std::vector<std::int32_t> prompt;
      std::int32_t token;
      while (input >> token) {
        prompt.push_back(token);
      }
      if (!input.eof() || prompt.empty() || prompt.size() + 15 > 64) {
        throw std::runtime_error("invalid request");
      }
      phase(name + ".generate.begin");
      const auto result = specferry::inference::generate_tokens(
          {50272, 64, 2, 2}, {reset, consume, predict}, prompt, 16);
      phase(name + ".generate.end");
      if (result.tokens != expected) {
        throw std::runtime_error("output tokens differ from the accepted CPU reference");
      }
      const auto transfers = tensor_transfers();
      request_timings.flush();
      std::ofstream record(output / (name + ".json"));
      record << std::setprecision(17) << '{';
      specferry::inference::write_generation(record, result);
      record << ",\"graph_seconds\":" << graph_seconds << ",\"upload_seconds\":" << upload_seconds
             << ",\"readback_seconds\":" << readback_seconds << ",\"variant\":\""
             << variants[request_index] << "\""
             << ",\"uploads\":" << transfers.uploads - transfer_start.uploads
             << ",\"upload_bytes\":" << transfers.upload_bytes - transfer_start.upload_bytes
             << ",\"reads\":" << transfers.reads - transfer_start.reads
             << ",\"read_bytes\":" << transfers.read_bytes - transfer_start.read_bytes << "}\n";
      write_bytes(output / (name + ".logits.bin"),
                  read_tensor(model.graph, outputs[logits_index].id, TensorReadback::MappedHost));
      if (incremental) {
        // After the final slot update, every fixed input cache must equal the NBG's
        // full present cache, including history and unused positions in all 24 layers.
        for (const auto &cache : caches) {
          if (read_tensor(model.graph, inputs[cache.input].id, TensorReadback::MappedHost) !=
              read_tensor(model.graph, outputs[cache.output].id, TensorReadback::MappedHost)) {
            throw std::runtime_error("incremental input cache differs from complete NBG output");
          }
        }
      }
      for (unsigned layer : {0u, 23u}) {
        write_bytes(output / (name + ".layer" + std::to_string(layer) + ".keys.bin"),
                    read_tensor(model.graph, outputs[caches[layer * 2].output].id,
                                TensorReadback::MappedHost));
        write_bytes(output / (name + ".layer" + std::to_string(layer) + ".values.bin"),
                    read_tensor(model.graph, outputs[caches[layer * 2 + 1].output].id,
                                TensorReadback::MappedHost));
      }
      record.close();
      if (!record) {
        throw std::runtime_error("cannot save generation record");
      }
    }
    const auto const_peak = context.memory().peak(true),
               nonconst_peak = context.memory().peak(false);
    phase("ReleaseGraph_ReleaseContext");
    const auto release_start = now();
    model.graph.close();
    context.close();
    const auto release_seconds = now() - release_start;
    if (context.memory().live(true) || context.memory().live(false)) {
      throw std::runtime_error("application allocation remained after release");
    }

    std::ofstream report(output / "execution.json");
    report << std::setprecision(17) << "{\"generation_complete\":true,\"released\":true"
           << ",\"const_peak_bytes\":" << const_peak << ",\"nonconst_peak_bytes\":" << nonconst_peak
           << ",\"const_live_bytes\":0,\"nonconst_live_bytes\":0,\"initialization_seconds\":"
           << preparation_seconds << ",\"release_seconds\":" << release_seconds
           << ",\"patch_contract_passed\":" << (patch_checked ? "true" : "false")
           << ",\"graph_calls\":" << sequence << "}\n";
    report.close();
    if (!report) {
      throw std::runtime_error("cannot save execution record");
    }
    phase("complete");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
