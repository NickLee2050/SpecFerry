#include "allocation_support.hpp"
#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "np101/weights.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::testing;

struct AllocationProgress {
  AllocationOutcome outcome;
  Storage storage = Storage::Constant;
  std::string current_weight;
  std::size_t tensors = 0;
  std::size_t payload_bytes = 0;
  std::size_t expected_weights = 0;
  std::size_t completed_weights = 0;
  std::size_t expected_weight_bytes = 0;
  std::size_t uploaded_weight_bytes = 0;
  std::size_t verified_weight_bytes = 0;
  std::size_t expected_state_bytes = 0;
  std::size_t uploaded_state_bytes = 0;
  std::size_t verified_state_bytes = 0;
  std::size_t chunk_offset = 0;
  std::size_t chunk_bytes = 0;
  bool states_complete = false;

  void save(const std::string &status = "running") const {
    std::ofstream report(outcome.report);
    report << std::boolalpha << "{\"status\":";
    json_string(report, status);
    report << ",\"allocated_tensors\":" << tensors << ",\"payload_bytes\":" << payload_bytes
           << ",\"weight_storage\":";
    json_string(report, storage_name(storage));
    report << ",\"is_const\":" << (storage == Storage::Constant) << ",\"current_weight\":";
    json_string(report, current_weight);
    report << ",\"chunk_offset_bytes\":" << chunk_offset << ",\"chunk_bytes\":" << chunk_bytes
           << ",\"expected_weights\":" << expected_weights
           << ",\"completed_weights\":" << completed_weights
           << ",\"expected_weight_bytes\":" << expected_weight_bytes
           << ",\"uploaded_weight_bytes\":" << uploaded_weight_bytes
           << ",\"verified_weight_bytes\":" << verified_weight_bytes
           << ",\"expected_state_bytes\":" << expected_state_bytes
           << ",\"uploaded_state_bytes\":" << uploaded_state_bytes
           << ",\"verified_state_bytes\":" << verified_state_bytes
           << ",\"state_allocation_complete\":" << states_complete << ',';
    outcome.write_fields(report);
    report << "}\n";
    report.close();
    if (!report) {
      throw std::runtime_error("cannot write allocation report");
    }
  }

  std::string status() const {
    if (!outcome.error.empty() || !outcome.released || !outcome.release_error.empty()) {
      return "failed";
    }
    return outcome.mismatch ? "readback_mismatch" : "allocation_pass";
  }
};

struct WeightChunk {
  vsi_nn_tensor_id_t tensor;
  const WeightRecord *record; // The WeightStore outlives every retained chunk.
  std::size_t offset;
  std::size_t bytes;
};

std::vector<WeightChunk> allocate_weights(Graph &graph, const WeightStore &weights,
                                          AllocationProgress &progress) {
  std::vector<WeightChunk> chunks;
  for (const auto &record : weights.records()) {
    progress.current_weight = record.name;
    auto shape = record.spec.shape;
    const auto total_rows = shape.back();
    const auto bytes_per_row = record.bytes / total_rows;
    const auto rows_per_chunk =
        std::min<std::uint64_t>(4096, allocation_block_bytes / bytes_per_row);
    if (rows_per_chunk == 0) {
      throw std::runtime_error("one weight row exceeds the bounded host staging buffer");
    }
    for (std::uint64_t row = 0; row < total_rows; row += rows_per_chunk) {
      const auto count = std::min<std::uint64_t>(rows_per_chunk, total_rows - row);
      shape.back() = count;
      const auto data = weights.read(record, row * bytes_per_row, count * bytes_per_row);
      progress.outcome.phase = "allocate_weight";
      progress.chunk_offset = row * bytes_per_row;
      progress.chunk_bytes = data.size();
      progress.save();
      const auto tensor = progress.storage == Storage::Constant
                              ? add_tensor(graph, {record.spec.type, shape}, true, data)
                              : add_tensor(graph, {record.spec.type, shape});
      progress.payload_bytes += data.size();
      ++progress.tensors;
      if (progress.storage == Storage::Mutable) {
        progress.outcome.phase = "upload_weight";
        progress.save();
        upload_tensor(graph, tensor, data);
      }
      progress.uploaded_weight_bytes += data.size();
      chunks.push_back({tensor, &record, progress.chunk_offset, data.size()});
      progress.save();
    }
    ++progress.completed_weights;
  }
  return chunks;
}

void verify_weights(Graph &graph, const WeightStore &weights,
                    const std::vector<WeightChunk> &chunks, AllocationProgress &progress) {
  // Read back only after all weights and states coexist.
  for (unsigned index = 0; index < chunks.size(); ++index) {
    const auto &chunk = chunks[index];
    progress.outcome.phase = "verify_weight";
    progress.current_weight = chunk.record->name;
    progress.chunk_offset = chunk.offset;
    progress.chunk_bytes = chunk.bytes;
    progress.save();
    const auto expected = weights.read(*chunk.record, chunk.offset, chunk.bytes);
    const auto actual = read_tensor(graph, chunk.tensor);
    if (!progress.outcome.verify(
            {chunk.record->name, chunk.tensor, index, chunk.record->spec.type, chunk.offset},
            actual, expected)) {
      std::cout << "weight readback mismatch: " << chunk.record->name
                << " byte=" << chunk.offset + progress.outcome.mismatch->offset << std::endl;
      break;
    }
    progress.verified_weight_bytes += chunk.bytes;
  }
}

struct StateAllocation {
  TensorSpec spec;
  unsigned copies;
};

std::vector<StateAllocation> read_states(const std::filesystem::path &path) {
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line) || line != "specferry-allocation-states 1") {
    throw std::invalid_argument("missing or unsupported allocation state specification");
  }
  std::vector<StateAllocation> states;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string dtype, shape, copies, extra;
    if (!(fields >> dtype >> shape >> copies) || fields >> extra || copies.empty() ||
        copies.find_first_not_of("0123456789") != std::string::npos) {
      throw std::invalid_argument("invalid allocation state record");
    }
    const auto count = std::stoul(copies);
    TensorSpec spec{parse_dtype(dtype), parse_shape(shape)};
    if (!count || count > 1024 || spec.bytes() > allocation_block_bytes) {
      throw std::invalid_argument("state fixture exceeds bounded allocation parameters");
    }
    states.push_back({spec, static_cast<unsigned>(count)});
  }
  if (states.empty()) {
    throw std::invalid_argument("empty allocation state specification");
  }
  return states;
}

struct StateTensor {
  vsi_nn_tensor_id_t tensor;
  TensorSpec spec;
};

std::vector<StateTensor> allocate_states(Graph &graph, AllocationProgress &progress,
                                         const std::vector<StateAllocation> &states) {
  std::vector<StateTensor> retained;
  for (const auto &state : states) {
    // Preserve the historical reproducer's zero initialization and allocation order.
    const std::vector<std::uint8_t> zeros(state.spec.bytes());
    for (unsigned copy = 0; copy < state.copies; ++copy) {
      progress.outcome.phase = "allocate_state";
      progress.chunk_bytes = zeros.size();
      progress.save();
      const auto tensor = add_tensor(graph, state.spec);
      ++progress.tensors;
      progress.payload_bytes += zeros.size();
      progress.outcome.phase = "upload_state";
      progress.save();
      upload_tensor(graph, tensor, zeros);
      progress.uploaded_state_bytes += zeros.size();
      retained.push_back({tensor, state.spec});
      progress.save();
    }
  }
  progress.states_complete = true;
  return retained;
}

void verify_states(Graph &graph, const std::vector<StateTensor> &states,
                   AllocationProgress &progress) {
  progress.outcome.phase = "verify_state";
  progress.current_weight.clear();
  progress.chunk_offset = 0;
  for (unsigned index = 0; index < states.size(); ++index) {
    const auto &state = states[index];
    progress.chunk_bytes = state.spec.bytes();
    progress.save();
    const std::vector<std::uint8_t> expected(state.spec.bytes());
    const auto actual = read_tensor(graph, state.tensor);
    if (!progress.outcome.verify(
            {"state." + std::to_string(index), state.tensor, index, state.spec.type, 0}, actual,
            expected)) {
      std::cout << "state readback mismatch: state." << index << std::endl;
      break;
    }
    progress.verified_state_bytes += expected.size();
  }
}

void probe(const WeightStore &weights, const std::vector<StateAllocation> &states,
           AllocationProgress &progress) {
  progress.outcome.phase = "initialize";
  progress.save();
  Context context;
  Graph graph(context, 1024, 1);
  try {
    const auto chunks = allocate_weights(graph, weights, progress);
    progress.current_weight.clear();
    progress.chunk_offset = 0;
    const auto state_tensors = allocate_states(graph, progress, states);
    verify_weights(graph, weights, chunks, progress);
    // A byte mismatch is not an SDK exception: still check state storage and
    // explicitly release. The first mismatch remains the primary evidence.
    verify_states(graph, state_tensors, progress);
  } catch (const std::exception &error) {
    progress.outcome.fail(error.what());
  }
  progress.outcome.release(graph, context, [&] { progress.save(); });
  progress.save(progress.status());
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 3 || argc > 7 || argc % 2 == 0) {
    std::cerr << "usage: np101_weight_allocation_check DEPLOYMENT_DIRECTORY REPORT_JSON "
                 "[--state-spec STATE_FILE] [--weight-storage constant|mutable]\n";
    return 2;
  }
  AllocationProgress progress;
  progress.outcome.report = argv[2];
  try {
    std::filesystem::path state_path;
    bool storage_specified = false;
    for (int index = 3; index < argc; index += 2) {
      const std::string option(argv[index]);
      if (option == "--state-spec" && state_path.empty()) {
        state_path = argv[index + 1];
        if (state_path.empty()) {
          throw std::invalid_argument("state file path is empty");
        }
      } else if (option == "--weight-storage" && !storage_specified) {
        progress.storage = parse_storage(argv[index + 1]);
        storage_specified = true;
      } else {
        throw std::invalid_argument("unknown or repeated allocation option: " + option);
      }
    }
    progress.outcome.phase = "validate_weights";
    progress.save();
    const WeightStore weights(argv[1]);
    weights.verify();
    progress.expected_weights = weights.records().size();
    for (const auto &record : weights.records()) {
      progress.expected_weight_bytes += record.bytes;
    }
    const auto states =
        state_path.empty() ? std::vector<StateAllocation>{} : read_states(state_path);
    for (const auto &state : states) {
      progress.expected_state_bytes += state.spec.bytes() * state.copies;
    }
    probe(weights, states, progress);
    return progress.status() == "allocation_pass" ? 0 : 1;
  } catch (const std::exception &error) {
    progress.outcome.fail(error.what());
    std::cerr << progress.outcome.phase << ": " << error.what() << '\n';
    try {
      progress.save("failed");
    } catch (const std::exception &report_error) {
      std::cerr << report_error.what() << '\n';
    }
    return 1;
  }
}
