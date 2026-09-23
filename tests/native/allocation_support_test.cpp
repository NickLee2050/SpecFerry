#include "np101/allocation_support.hpp"
#include "np101/tensor_spec.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::testing;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_pattern(DataType dtype) {
  const auto expected = allocation_data(dtype, 109, 16 * 1024);
  require(expected == allocation_data(dtype, 109, expected.size()), "pattern is not reproducible");
  require(expected != allocation_data(dtype, 110, expected.size()), "blocks alias in the pattern");
  auto aliased_page = expected;
  std::copy_n(expected.begin(), 4096, aliased_page.begin() + 4096);
  require(compare_bytes(aliased_page, expected).has_value(), "4 KiB page alias escaped comparison");
  const auto width = dtype_bytes(dtype);
  for (std::size_t offset = 0; offset < expected.size(); offset += width) {
    if (dtype == DataType::Float16) {
      const auto bits = unsigned(expected[offset]) | (unsigned(expected[offset + 1]) << 8);
      require((bits & 0x7c00) != 0x7c00 && (bits & 0x7fff) != 0,
              "FP16 pattern contains nonfinite or zero values");
    } else {
      float value;
      std::memcpy(&value, expected.data() + offset, sizeof(value));
      require(std::isfinite(value) && value != 0, "FP32 pattern contains nonfinite or zero values");
    }
  }
}

void check_comparison() {
  const std::vector<std::uint8_t> expected{1, 2, 3, 4};
  require(!compare_bytes(expected, expected), "identical bytes failed");
  const auto shorter = compare_bytes({1, 2}, expected);
  require(shorter && shorter->reason == "size_mismatch" && shorter->offset == 2 &&
              shorter->expected == 3 && !shorter->actual,
          "short readback was accepted or incorrectly indexed");
  const auto longer = compare_bytes({1, 2, 3, 4, 5}, expected);
  require(longer && longer->offset == 4 && !longer->expected && longer->actual == 5,
          "oversized readback was accepted or incorrectly indexed");
  const auto changed = compare_bytes({1, 0, 3, 4}, expected);
  require(changed && changed->reason == "byte_mismatch" && changed->offset == 1 &&
              changed->expected == 2 && changed->actual == 0,
          "changed byte was not reported precisely");
  require(compare_bytes({}, expected).has_value(), "empty readback was accepted");
}

void check_report_escaping() {
  std::ostringstream output;
  json_string(output, "SDK \"error\"\\\n\t");
  require(output.str() == "\"SDK \\\"error\\\"\\\\\\u000a\\u0009\"", "invalid JSON escaping");
  AllocationOutcome outcome;
  outcome.phase = "upload";
  outcome.fail("report write failed");
  require(outcome.failure_phase == "upload" && !outcome.released,
          "program failure lost its phase or claimed cleanup");
}

void check_scan() {
  std::vector<std::uint8_t> expected(3 * 4096, 1);
  auto actual = expected;
  actual[0] = 0;
  actual[4095] = 0;
  actual[4096] = 0;
  actual.back() = 0;
  const auto scan = scan_readback(actual, expected);
  require(scan.mismatched_bytes == 4 && scan.range_count == 3 && scan.ranges.size() == 3,
          "scan lost separated mismatches or a page-crossing interval");
  require(scan.ranges[1].begin == 4095 && scan.ranges[1].end == 4097 &&
              scan.page_ranges.size() == 1 && scan.page_ranges[0].begin == 0 &&
              scan.page_ranges[0].end == 3,
          "scan page coverage is incomplete");
  const auto bounded = scan_readback(actual, expected, 1);
  require(bounded.ranges.size() == 1 && bounded.range_count == 3 && bounded.mismatched_bytes == 4 &&
              bounded.page_ranges[0].end == 3,
          "bounded detail truncated total counts or page coverage");
  const auto short_read = scan_readback({1, 2}, {1, 2, 3, 4});
  require(short_read.mismatched_bytes == 2 && short_read.ranges[0].begin == 2 &&
              short_read.ranges[0].end == 4,
          "short readback lost the missing tail");
  require(scan_readback(expected, expected).page_ranges.empty(), "intact data produced bad pages");
}
} // namespace

int main() {
  try {
    check_pattern(DataType::Float16);
    check_pattern(DataType::Float32);
    check_comparison();
    check_report_escaping();
    check_scan();
    std::cout << "Allocation pattern and readback contracts passed without opening the device.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
