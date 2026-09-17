#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/resource.h>

namespace {
using namespace specferry::np101;

struct AllocationProgress {
  std::filesystem::path output;
  std::string phase = "validate_weights";
  std::string current_weight;
  bool constant_weights = true;
  std::size_t tensors = 0;
  std::size_t payload_bytes = 0;
  std::size_t expected_weights = 0;
  std::size_t completed_weights = 0;
  std::size_t expected_weight_bytes = 0;
  std::size_t uploaded_weight_bytes = 0;
  std::size_t verified_weight_bytes = 0;
  std::size_t chunk_offset = 0;
  std::size_t chunk_bytes = 0;
  bool states_complete = false;

  void save(const std::string &status) const {
    struct rusage usage {};

    if (getrusage(RUSAGE_SELF, &usage) != 0) {
      throw std::runtime_error("getrusage failed");
    }
    std::ofstream report(output);
    report << std::boolalpha << "{\"status\":\"" << status << "\",\"phase\":\"" << phase
           << "\",\"allocated_tensors\":" << tensors << ",\"payload_bytes\":" << payload_bytes
           << ",\"weight_storage\":\"" << (constant_weights ? "constant" : "mutable") << '"'
           << ",\"is_const\":" << constant_weights
           << ",\"current_weight\":" << std::quoted(current_weight)
           << ",\"chunk_offset_bytes\":" << chunk_offset << ",\"chunk_bytes\":" << chunk_bytes
           << ",\"expected_weights\":" << expected_weights
           << ",\"completed_weights\":" << completed_weights
           << ",\"expected_weight_bytes\":" << expected_weight_bytes
           << ",\"uploaded_weight_bytes\":" << uploaded_weight_bytes
           << ",\"verified_weight_bytes\":" << verified_weight_bytes
           << ",\"state_allocation_complete\":" << states_complete
           << ",\"host_peak_rss_bytes\":" << usage.ru_maxrss * 1024ULL
           << ",\"device_residency_verified\":false,\"model_memory_fit_verified\":false,"
              "\"graph_workspace_included\":false}\n";
    if (!report) {
      throw std::runtime_error("cannot write allocation report");
    }
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
    auto total_rows = shape.back();
    auto bytes_per_row = record.bytes / total_rows;
    auto rows_per_chunk = std::min<std::uint64_t>(4096, 8 * 1024 * 1024 / bytes_per_row);
    if (rows_per_chunk == 0) {
      throw std::runtime_error("one weight row exceeds the bounded host staging buffer");
    }
    for (std::uint64_t row = 0; row < total_rows; row += rows_per_chunk) {
      auto count = std::min<std::uint64_t>(rows_per_chunk, total_rows - row);
      shape.back() = count;
      auto data = weights.read(record, row * bytes_per_row, count * bytes_per_row);
      progress.phase = "allocate_weight";
      progress.chunk_offset = row * bytes_per_row;
      progress.chunk_bytes = data.size();
      progress.save("running");
      vsi_nn_tensor_id_t tensor;
      if (progress.constant_weights) {
        tensor = add_tensor(graph, {record.spec.type, shape}, true, data);
      } else {
        // Follow the documented mutable-tensor path: create, then explicitly
        // upload bytes. Creating an empty tensor alone is not a loading test.
        tensor = add_tensor(graph, {record.spec.type, shape});
      }
      progress.payload_bytes += data.size();
      ++progress.tensors;
      if (!progress.constant_weights) {
        progress.phase = "upload_weight";
        progress.save("running");
        upload_tensor(graph, tensor, data);
      }
      progress.uploaded_weight_bytes += data.size();
      chunks.push_back({tensor, &record, progress.chunk_offset, data.size()});
      progress.save("running");
    }
    ++progress.completed_weights;
  }
  return chunks;
}

void verify_weights(Graph &graph, const WeightStore &weights,
                    const std::vector<WeightChunk> &chunks, AllocationProgress &progress) {
  // Read back only after all weights and states coexist, so reusing or
  // overwriting an earlier allocation cannot silently pass the loading check.
  for (const auto &chunk : chunks) {
    progress.phase = "verify_weight";
    progress.current_weight = chunk.record->name;
    progress.chunk_offset = chunk.offset;
    progress.chunk_bytes = chunk.bytes;
    progress.save("running");
    auto expected = weights.read(*chunk.record, chunk.offset, chunk.bytes);
    if (read_tensor(graph, chunk.tensor) != expected) {
      throw std::runtime_error("weight readback differs from the exported bytes");
    }
    progress.verified_weight_bytes += chunk.bytes;
  }
}

void allocate_state(Graph &graph, AllocationProgress &progress) {
  struct StateAllocation {
    TensorSpec spec;
    unsigned copies;
  };

  // Explicit double buffers and compact 512-token KV. Activations and SDK graph
  // packing are unknown until full model graphs exist and are deliberately excluded.
  const std::vector<StateAllocation> states{
      {{DataType::Float32, {128, 128, 16}}, 36},
      {{DataType::Float16, {4, 6144}}, 36},
      {{DataType::Float16, {256, 2, 512}}, 12},
      {{DataType::Float32, {64, 512}}, 2},
  };
  for (const auto &state : states) {
    std::vector<std::uint8_t> zeros(state.spec.bytes());
    for (unsigned copy = 0; copy < state.copies; ++copy) {
      progress.phase = "allocate_state";
      progress.chunk_bytes = zeros.size();
      progress.save("running");
      auto id = add_tensor(graph, state.spec);
      ++progress.tensors;
      progress.payload_bytes += zeros.size();
      progress.phase = "upload_state";
      progress.save("running");
      upload_tensor(graph, id, zeros);
      progress.save("running");
    }
  }
  progress.states_complete = true;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3 && argc != 5) {
    std::cerr << "usage: np101_weight_allocation_check DEPLOYMENT_DIRECTORY REPORT_JSON "
                 "[--weight-storage constant|mutable]\n";
    return 2;
  }
  AllocationProgress progress;
  progress.output = argv[2];
  if (argc == 5) {
    const std::string option = argv[3], storage = argv[4];
    if (option != "--weight-storage" || (storage != "constant" && storage != "mutable")) {
      std::cerr << "invalid weight storage option\n";
      return 2;
    }
    progress.constant_weights = storage == "constant";
  }
  try {
    progress.save("running");
    const WeightStore weights(argv[1]);
    weights.verify();
    progress.expected_weights = weights.records().size();
    for (const auto &record : weights.records()) {
      progress.expected_weight_bytes += record.bytes;
    }
    progress.phase = "initialize";
    progress.save("running");
    Context context;
    Graph graph(context, 1024, 1);
    const auto chunks = allocate_weights(graph, weights, progress);
    progress.current_weight.clear();
    progress.chunk_offset = 0;
    allocate_state(graph, progress);
    verify_weights(graph, weights, chunks, progress);
    progress.phase = "release";
    progress.save("running");
    graph.close();
    context.close();
    progress.phase = "complete";
    progress.save("allocation_pass");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << progress.phase << ": " << error.what() << '\n';
    try {
      progress.save("failed");
    } catch (const std::exception &report_error) {
      std::cerr << report_error.what() << '\n';
    }
    return 1;
  }
}
