#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace specferry::np101;

struct Allocation {
  vsi_nn_tensor_id_t id;
  std::size_t bytes;
  std::uint32_t seed;
};

std::vector<std::uint8_t> pattern(std::size_t bytes, std::uint32_t seed) {
  std::vector<std::uint8_t> result(bytes);
  for (std::size_t offset = 0; offset < bytes; offset += sizeof(std::uint32_t)) {
    // Finite FP32 values with allocation-specific and position-dependent mantissas.
    const std::uint32_t value = 0x3f000000u | ((offset / 4 + seed * 104729u) & 0x007fffffu);
    std::memcpy(result.data() + offset, &value, sizeof(value));
  }
  return result;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: np101_memory_check OUTPUT\n";
    return 2;
  }
  try {
    std::filesystem::create_directories(argv[1]);
    SdkTimings timings(argv[1]);
    std::cout << "creating context\n" << std::flush;
    Context context;
    // This graph only owns buffers. No operators, Setup, Verify or RunGraph are used.
    Graph graph(context, 6, 1);
    std::vector<Allocation> allocations;
    for (const unsigned mib : {1u, 4u, 16u}) {
      for (const bool constant : {false, true}) {
        const std::size_t bytes = mib * 1024u * 1024u;
        const auto seed = static_cast<std::uint32_t>(allocations.size() + 1);
        std::cout << "allocating const=" << constant << " bytes=" << bytes << '\n' << std::flush;
        const auto id = add_tensor(graph, {DataType::Float32, {1024, mib * 256}}, constant,
                                   pattern(bytes, seed));
        allocations.push_back({id, bytes, seed});
        // Recheck every retained block to detect aliasing or damage from later allocations.
        for (const auto &allocation : allocations) {
          if (read_tensor(graph, allocation.id, TensorReadback::MappedHost) !=
              pattern(allocation.bytes, allocation.seed)) {
            throw std::runtime_error("retained tensor readback differs");
          }
        }
        std::cout << "readback passed const_live=" << context.memory().live(true)
                  << " nonconst_live=" << context.memory().live(false) << '\n'
                  << std::flush;
      }
    }
    graph.close();
    context.close();
    timings.flush();
    if (context.memory().live(true) || context.memory().live(false)) {
      throw std::runtime_error("application allocation budget did not return to zero");
    }
    std::cout << "released const_peak=" << context.memory().peak(true)
              << " nonconst_peak=" << context.memory().peak(false) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
