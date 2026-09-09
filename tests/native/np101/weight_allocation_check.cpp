#include "np101/tensor.hpp"
#include "np101/weights.hpp"

#include <fstream>
#include <iostream>
#include <sys/resource.h>

namespace {
using namespace specferry::np101;

struct AllocationProgress {
  std::filesystem::path output;
  std::string phase = "validate_weights";
  std::size_t tensors = 0;
  std::size_t payload_bytes = 0;

  void save(const std::string &status) const {
    struct rusage usage {};

    if (getrusage(RUSAGE_SELF, &usage) != 0) {
      throw std::runtime_error("getrusage failed");
    }
    std::ofstream report(output);
    report << "{\"status\":\"" << status << "\",\"phase\":\"" << phase
           << "\",\"allocated_tensors\":" << tensors << ",\"payload_bytes\":" << payload_bytes
           << ",\"host_peak_rss_bytes\":" << usage.ru_maxrss * 1024ULL
           << ",\"device_residency_verified\":false,\"model_memory_fit_verified\":false,"
              "\"graph_workspace_included\":false}\n";
    if (!report) {
      throw std::runtime_error("cannot write allocation report");
    }
  }
};

void allocate_weights(Graph &graph, const WeightStore &weights, AllocationProgress &progress) {
  for (const auto &record : weights.records()) {
    progress.phase = record.name;
    progress.save("running");
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
      add_tensor(graph, {record.spec.type, shape}, true, data);
      progress.payload_bytes += data.size();
      ++progress.tensors;
    }
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
      auto id = add_tensor(graph, state.spec);
      upload_tensor(graph, id, zeros);
      ++progress.tensors;
      progress.payload_bytes += zeros.size();
      progress.save("running");
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: np101_weight_allocation_check DEPLOYMENT_DIRECTORY REPORT_JSON\n";
    return 2;
  }
  AllocationProgress progress;
  progress.output = argv[2];
  try {
    progress.save("running");
    const WeightStore weights(argv[1]);
    weights.verify();
    Context context;
    Graph graph(context, 1024, 1);
    allocate_weights(graph, weights, progress);
    progress.phase = "allocate_states";
    allocate_state(graph, progress);
    progress.phase = "release";
    graph.close();
    context.close();
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
