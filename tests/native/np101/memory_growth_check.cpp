#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;

std::uint64_t rss_bytes() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::uint64_t kib;
      std::istringstream value(line.substr(6));
      if (value >> kib) {
        return kib * 1024;
      }
    }
  }
  throw std::runtime_error("cannot read host RSS");
}

// Views and graph are allocated once; the loop holds no growing containers.
struct CopyGraph {
  vx_graph graph = nullptr;
  std::array<vx_tensor, 8> views{};

  CopyGraph() = default;
  CopyGraph(const CopyGraph &) = delete;
  CopyGraph &operator=(const CopyGraph &) = delete;

  void close() {
    if (graph) {
      check(vxReleaseGraph(&graph), "release copy graph");
    }
    for (auto &view : views) {
      if (view) {
        check(vxReleaseTensor(&view), "release view");
      }
    }
  }

  ~CopyGraph() {
    if (graph) {
      vxReleaseGraph(&graph);
    }
    for (auto &view : views) {
      if (view) {
        vxReleaseTensor(&view);
      }
    }
  }
};

void initialize_copy(CopyGraph &copy, Context &context, Graph &storage, vsi_nn_tensor_id_t source,
                     vsi_nn_tensor_id_t parent) {
  copy.graph = vxCreateGraph(context.get()->c);
  check(vxGetStatus(reinterpret_cast<vx_reference>(copy.graph)), "create graph");
  for (unsigned slot = 0; slot < copy.views.size(); ++slot) {
    std::array<vsi_size_t, 3> start{0, slot, 0}, end{64, slot + 1, 16};
    copy.views[slot] = vsi_nn_CreateViewTensor(storage.get(), start.data(), end.data(),
                                               vsi_nn_GetTensor(storage.get(), parent));
    check(vxGetStatus(reinterpret_cast<vx_reference>(copy.views[slot])), "create view");
  }
  auto node =
      vxTensorCopyNode(copy.graph, vsi_nn_GetTensor(storage.get(), source)->t, copy.views[0]);
  check(vxGetStatus(reinterpret_cast<vx_reference>(node)), "create copy node");
  auto parameter = vxGetParameterByIndex(node, 1);
  if (!parameter) {
    vxReleaseNode(&node);
    throw std::runtime_error("copy destination parameter is null");
  }
  auto status = vxAddParameterToGraph(copy.graph, parameter);
  vxReleaseParameter(&parameter);
  vxReleaseNode(&node);
  check(status, "expose destination");
  check(vxVerifyGraph(copy.graph), "initial verify");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: np101_memory_growth_check fixed|same|advance ITERATIONS[1..4096]");
    }
    const std::string mode(argv[1]);
    if (mode != "fixed" && mode != "same" && mode != "advance") {
      throw std::invalid_argument("unknown binding mode");
    }
    const auto dimensions = parse_shape(argv[2]);
    if (dimensions.size() != 1 || dimensions[0] > 4096) {
      throw std::invalid_argument("iterations must be in [1,4096]");
    }
    const auto iterations = dimensions[0];
    // Keep tensor storage and all views fixed throughout the experiment.
    Context context;
    Graph storage(context, 2, 0);
    std::vector<std::uint8_t> input(64 * 16 * 2, 0x30), expected(input.size() * 8);
    const auto source = add_tensor(storage, {DataType::Float16, {64, 1, 16}}, false, input);
    const auto parent = add_tensor(storage, {DataType::Float16, {64, 8, 16}}, false, expected);
    const auto initial_rss = rss_bytes();
    std::cout << "One copy node, 8 preallocated views, " << input.size()
              << " bytes copied per iteration. Mode: " << mode << '\n'
              << "Columns: iteration phase host_RSS_bytes phase_seconds\n";
    {
      CopyGraph copy;
      initialize_copy(copy, context, storage, source, parent);

      for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        auto phase = [&](const char *name, auto action) {
          const auto start = std::chrono::steady_clock::now();
          action();
          const auto seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          std::cout << iteration << ' ' << name << ' ' << rss_bytes() << ' ' << seconds
                    << std::endl;
        };
        const auto slot = mode == "advance" ? iteration % copy.views.size() : 0;
        phase("bind", [&] {
          if (mode != "fixed") {
            check(vxSetGraphParameterByIndex(copy.graph, 0,
                                             reinterpret_cast<vx_reference>(copy.views[slot])),
                  "bind slot");
          }
        });
        phase("verify", [&] {
          if (!vxIsGraphVerified(copy.graph)) {
            check(vxVerifyGraph(copy.graph), "reverify");
          }
        });
        phase("process", [&] { check(vxProcessGraph(copy.graph), "copy slot"); });
        for (unsigned head = 0; head < 16; ++head) {
          std::fill_n(expected.begin() + (head * 8 + slot) * 64 * 2, 64 * 2, 0x30);
        }
      }

      // Read only after the final copy; untouched slots must remain zero.
      auto actual = read_tensor(storage, parent);
      if (actual != expected) {
        throw std::runtime_error("final parent byte readback differs");
      }
      copy.close();
    }
    std::cout << "After copy graph/views release RSS: " << rss_bytes() << " bytes\n";
    storage.close();
    context.close();
    std::cout << "Initial / final host RSS: " << initial_rss << " / " << rss_bytes()
              << " bytes\nByte readback and graph/context release: PASS\n"
              << "RSS includes allocator retention; it does not prove a leak or device usage.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
  }
}
