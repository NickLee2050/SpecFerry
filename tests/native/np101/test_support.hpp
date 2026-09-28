#pragma once

#include "np101/diagnostics.hpp"
#include "np101/tensor_spec.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>

namespace specferry::testing {
std::vector<std::uint8_t> encode_floats(const std::vector<float> &values,
                                        np101::DataType type = np101::DataType::Float16);

// Argument validation stays in main, before any device access. The test owns and
// explicitly closes its SDK objects; this helper only handles output and errors.
template <class Action> int run_test(const std::filesystem::path &output, Action action) {
  try {
    std::cout << std::unitbuf;
    std::filesystem::create_directories(output);
    np101::SdkTimings timings(output);
    action();
    std::cout << "Completed; graph/context released\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
} // namespace specferry::testing
