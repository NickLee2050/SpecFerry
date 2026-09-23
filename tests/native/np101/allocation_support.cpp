#include "allocation_support.hpp"
#include "np101/context.hpp"
#include "sys/resource.h"

#include <algorithm>
#include <exception>
#include <fstream>
#include <ios>
#include <ostream>
#include <stdexcept>

namespace specferry::testing {
namespace {
std::uint64_t mix_bits(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

void save_bytes(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.close();
  if (!output) {
    throw std::runtime_error("cannot save readback evidence: " + path.string());
  }
}

void optional_byte(std::ostream &output, const std::optional<std::uint8_t> &value) {
  if (value) {
    output << unsigned(*value);
  } else {
    output << "null";
  }
}
} // namespace

Storage parse_storage(const std::string &value) {
  if (value == "constant") {
    return Storage::Constant;
  }
  if (value == "mutable") {
    return Storage::Mutable;
  }
  throw std::invalid_argument("storage must be constant or mutable");
}

const char *storage_name(Storage storage) {
  return storage == Storage::Constant ? "constant" : "mutable";
}

std::vector<std::uint8_t> allocation_data(np101::DataType dtype, unsigned block,
                                          std::size_t bytes) {
  if (dtype != np101::DataType::Float16 && dtype != np101::DataType::Float32) {
    throw std::invalid_argument("allocation pattern requires F16 or F32");
  }
  const unsigned width = dtype == np101::DataType::Float16 ? 2 : 4;
  if (bytes == 0 || bytes > allocation_block_bytes || bytes % width) {
    throw std::invalid_argument("invalid allocation pattern byte count");
  }
  std::vector<std::uint8_t> result(bytes);
  for (std::size_t offset = 0; offset < bytes; offset += width) {
    const auto mixed = mix_bits((std::uint64_t(block) << 32) | (offset / width));
    const auto word =
        width == 2 ? 0x3000U | (mixed & 0x0fffU) : 0x3f000000U | (mixed & 0x007fffffU);
    for (unsigned byte = 0; byte < width; ++byte) {
      result[offset + byte] = (word >> (8 * byte)) & 0xff;
    }
  }
  return result;
}

std::optional<ByteMismatch> compare_bytes(const std::vector<std::uint8_t> &actual,
                                          const std::vector<std::uint8_t> &expected) {
  const auto common = std::min(actual.size(), expected.size());
  std::size_t offset = 0;
  while (offset < common && actual[offset] == expected[offset]) {
    ++offset;
  }
  if (actual.size() == expected.size() && offset == common) {
    return std::nullopt;
  }
  ByteMismatch mismatch{actual.size() == expected.size() ? "byte_mismatch" : "size_mismatch",
                        offset,
                        expected.size(),
                        actual.size(),
                        std::nullopt,
                        std::nullopt};
  if (offset < expected.size()) {
    mismatch.expected = expected[offset];
  }
  if (offset < actual.size()) {
    mismatch.actual = actual[offset];
  }
  return mismatch;
}

ReadbackScan scan_readback(const std::vector<std::uint8_t> &actual,
                           const std::vector<std::uint8_t> &expected, std::size_t maximum_ranges) {
  ReadbackScan result;
  const auto size = std::max(actual.size(), expected.size());
  const auto differs = [&](std::size_t offset) {
    return offset >= actual.size() || offset >= expected.size() ||
           actual[offset] != expected[offset];
  };
  std::size_t offset = 0;
  while (offset < size) {
    if (!differs(offset)) {
      ++offset;
      continue;
    }
    const auto begin = offset;
    while (offset < size && differs(offset)) {
      ++offset;
    }
    result.mismatched_bytes += offset - begin;
    ++result.range_count;
    if (result.ranges.size() < maximum_ranges) {
      result.ranges.push_back({begin, offset});
    }
    const ByteRange pages{begin / 4096, (offset - 1) / 4096 + 1};
    if (!result.page_ranges.empty() && pages.begin <= result.page_ranges.back().end) {
      result.page_ranges.back().end = pages.end;
    } else {
      result.page_ranges.push_back(pages);
    }
  }
  return result;
}

void json_string(std::ostream &output, const std::string &value) {
  constexpr char hex[] = "0123456789abcdef";
  output << '"';
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') {
      output << '\\' << character;
    } else if (character < 0x20) {
      output << "\\u00" << hex[character >> 4] << hex[character & 15];
    } else {
      output << character;
    }
  }
  output << '"';
}

void AllocationOutcome::fail(const std::string &message) {
  if (failure_phase.empty()) {
    failure_phase = phase;
  }
  error = message;
}

bool AllocationOutcome::verify(const ReadbackLocation &where,
                               const std::vector<std::uint8_t> &actual,
                               const std::vector<std::uint8_t> &expected) {
  auto difference = compare_bytes(actual, expected);
  if (!difference) {
    return true;
  }
  if (!mismatch) {
    mismatch = difference;
    location = where;
    failure_phase = phase;
    save_bytes(report.parent_path() / "first-mismatch-expected.bin", expected);
    save_bytes(report.parent_path() / "first-mismatch-actual.bin", actual);
  }
  return false;
}

void AllocationOutcome::release(np101::Graph &graph, np101::Context &context,
                                const std::function<void()> &record_progress) {
  phase = "release";
  try {
    record_progress();
  } catch (const std::exception &exception) {
    // Report IO failure must not skip device teardown or become a capacity bound.
    fail(exception.what());
  }
  try {
    // Do not destroy the context ahead of a graph whose explicit release failed.
    graph.close();
    context.close();
    released = true;
  } catch (const std::exception &exception) {
    release_error = exception.what();
    if (failure_phase.empty()) {
      failure_phase = phase;
    }
  }
  phase = "complete";
}

void AllocationOutcome::write_fields(std::ostream &output) const {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    throw std::runtime_error("getrusage failed");
  }
  output << "\"report_version\":" << allocation_report_version << ",\"phase\":";
  json_string(output, phase);
  output << ",\"failure_phase\":";
  json_string(output, failure_phase);
  output << ",\"error\":";
  json_string(output, error);
  output << ",\"release_error\":";
  json_string(output, release_error);
  output << ",\"released\":" << (released ? "true" : "false") << ",\"readback_error\":";
  json_string(output, mismatch ? mismatch->reason : "");
  output << ",\"first_mismatch\":";
  if (mismatch) {
    output << "{\"tensor_name\":";
    json_string(output, location.name);
    output << ",\"dtype\":";
    json_string(output, np101::dtype_name(location.dtype));
    output << ",\"tensor_id\":" << location.tensor_id << ",\"block_index\":" << location.block
           << ",\"chunk_offset_bytes\":" << location.chunk_offset
           << ",\"byte_offset\":" << mismatch->offset
           << ",\"tensor_byte_offset\":" << location.chunk_offset + mismatch->offset
           << ",\"expected_bytes\":" << mismatch->expected_size
           << ",\"actual_bytes\":" << mismatch->actual_size << ",\"expected_byte\":";
    optional_byte(output, mismatch->expected);
    output << ",\"actual_byte\":";
    optional_byte(output, mismatch->actual);
    output << ",\"expected_file\":\"first-mismatch-expected.bin\","
              "\"actual_file\":\"first-mismatch-actual.bin\"}";
  } else {
    output << "null";
  }
  output << ",\"host_peak_rss_bytes\":" << usage.ru_maxrss * 1024ULL
         << ",\"device_residency_verified\":false,\"model_memory_fit_verified\":false,"
            "\"graph_workspace_included\":false";
}
} // namespace specferry::testing
