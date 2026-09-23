#include "allocation_support.hpp"
#include "gc_hal_base.h"
#include "np101/context.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::testing;

memory_profile_info counters() {
  memory_profile_info info;
  std::memset(&info, 0xa5, sizeof(info));
  check(gcoOS_GetMemoryProfileInfo(sizeof(info), &info), "query SDK memory counters");
  const auto *bytes = reinterpret_cast<const unsigned char *>(&info.gpu_memory.currentSize);
  if (std::all_of(bytes, bytes + sizeof(info.gpu_memory.currentSize),
                  [](auto byte) { return byte == 0xa5; })) {
    throw std::runtime_error("SDK did not populate gpu_memory; run with VIV_MEMORY_PROFILE=1");
  }
  return info;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3 && argc != 4) {
      throw std::invalid_argument(
          "usage: np101_memory_accounting_check MIB[1..64] constant|mutable [F16|F32]");
    }
    const auto dimensions = parse_shape(argv[1]);
    if (dimensions.size() != 1 || dimensions[0] > 64) {
      throw std::invalid_argument("payload must be in [1,64] MiB");
    }
    const auto size_mib = dimensions[0];
    const bool constant = parse_storage(argv[2]) == Storage::Constant;
    const auto dtype = argc == 4 ? parse_dtype(argv[3]) : DataType::Float16;
    if (dtype != DataType::Float16 && dtype != DataType::Float32) {
      throw std::invalid_argument("accounting control requires F16 or F32");
    }

    Context context;
    Graph storage(context, 1, 0);
    std::vector<std::uint8_t> data(size_mib * mib, 0x30);
    const auto before = counters();
    const auto rows = static_cast<std::uint32_t>(data.size() / (1024 * dtype_bytes(dtype)));
    const auto id = add_tensor(storage, {dtype, {1024, rows}}, constant, data);
    if (!constant) {
      // Match the capacity probe's explicit mutable-upload path before measuring.
      upload_tensor(storage, id, data);
    }
    const auto allocated = counters();

    const bool equal = read_tensor(storage, id) == data;
    storage.close();
    context.close();
    const auto released = counters();

    const auto delta = static_cast<std::int64_t>(allocated.gpu_memory.currentSize) -
                       static_cast<std::int64_t>(before.gpu_memory.currentSize);
    std::cout << "Tensor payload: " << data.size() << " bytes (" << size_mib << " MiB)\n"
              << "Storage: " << argv[2] << "; dtype: " << dtype_name(dtype)
              << "\nSDK gpu_memory before / allocated / released: " << before.gpu_memory.currentSize
              << " / " << allocated.gpu_memory.currentSize << " / "
              << released.gpu_memory.currentSize << " bytes\n"
              << "SDK counter delta / payload: " << delta << " / " << data.size() << " = "
              << std::setprecision(8) << double(delta) / data.size() << "\n"
              << "Byte readback: " << (equal ? "PASS" : "FAIL") << "\n"
              << "Graph/context release: PASS\n"
              << "Counter semantics are unconfirmed; this is NOT a physical-memory measurement.\n";
    return equal ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
  }
}
