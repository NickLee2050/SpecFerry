#pragma once

#include "np101/tensor_spec.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace specferry::np101 {
class Context;
class Graph;
} // namespace specferry::np101

namespace specferry::testing {
constexpr std::size_t mib = 1024 * 1024;
constexpr std::size_t allocation_block_bytes = 8 * mib;
constexpr unsigned maximum_capacity_mib = 4096;
constexpr unsigned allocation_report_version = 2;
constexpr const char *allocation_pattern = "splitmix64-finite-v1";

enum class Storage { Constant, Mutable };
Storage parse_storage(const std::string &value);
const char *storage_name(Storage storage);

// Positive, finite values derived from both the block and full element position.
std::vector<std::uint8_t> allocation_data(np101::DataType dtype, unsigned block, std::size_t bytes);

struct ByteMismatch {
  std::string reason;
  std::size_t offset;
  std::size_t expected_size;
  std::size_t actual_size;
  std::optional<std::uint8_t> expected;
  std::optional<std::uint8_t> actual;
};

std::optional<ByteMismatch> compare_bytes(const std::vector<std::uint8_t> &actual,
                                          const std::vector<std::uint8_t> &expected);

struct ByteRange {
  std::size_t begin;
  std::size_t end;
};

struct ReadbackScan {
  std::size_t mismatched_bytes = 0;
  std::size_t range_count = 0;
  // Bound detailed output even if every other byte is corrupt. Page intervals
  // still cover every affected logical 4 KiB page; they are not physical pages.
  std::vector<ByteRange> ranges;
  std::vector<ByteRange> page_ranges;
};

ReadbackScan scan_readback(const std::vector<std::uint8_t> &actual,
                           const std::vector<std::uint8_t> &expected,
                           std::size_t maximum_ranges = 256);
void json_string(std::ostream &output, const std::string &value);

struct ReadbackLocation {
  std::string name;
  std::uint32_t tensor_id;
  unsigned block;
  np101::DataType dtype;
  std::size_t chunk_offset;
};

// Common report fields for the synthetic and real-weight allocation diagnostics.
struct AllocationOutcome {
  std::filesystem::path report;
  std::string phase = "initialize";
  std::string failure_phase;
  std::string error;
  std::string release_error;
  bool released = false;
  std::optional<ByteMismatch> mismatch;
  ReadbackLocation location{};

  void fail(const std::string &message);
  bool verify(const ReadbackLocation &where, const std::vector<std::uint8_t> &actual,
              const std::vector<std::uint8_t> &expected);
  void release(np101::Graph &graph, np101::Context &context,
               const std::function<void()> &record_progress);
  // Writes fields inside an already opened JSON object, without a trailing comma.
  void write_fields(std::ostream &output) const;
};
} // namespace specferry::testing
