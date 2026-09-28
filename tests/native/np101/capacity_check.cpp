#include "allocation_support.hpp"
#include "np101/context.hpp"
#include "np101/memory_budget.hpp"
#include "np101/tensor.hpp"
#include "np101/tensor_spec.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::testing;

struct CapacityProgress {
  AllocationOutcome outcome;
  Storage storage = Storage::Constant;
  DataType dtype = DataType::Float16;
  std::string readback = "first";
  std::string allocation_error;
  std::string rejection_phase;
  std::size_t target = 0;
  std::size_t uploaded = 0;
  std::size_t verified = 0;
  std::size_t rejected = 0;
  std::size_t tensors = 0;
  std::size_t scanned = 0;
  std::size_t scanned_blocks = 0;
  std::size_t mismatched_bytes = 0;
  std::size_t mismatched_blocks = 0;
  std::size_t mismatched_ranges = 0;
  std::size_t size_mismatch_blocks = 0;
  bool probe_limit = false;

  void save(const std::string &status = "running") const {
    std::ofstream output(outcome.report);
    output << "{\"status\":";
    json_string(output, status);
    output << ",\"storage\":";
    json_string(output, storage_name(storage));
    output << ",\"dtype\":";
    json_string(output, dtype_name(dtype));
    output << ",\"pattern\":";
    json_string(output, allocation_pattern);
    output << ",\"readback_mode\":";
    json_string(output, readback);
    output << ",\"target_bytes\":" << target << ",\"block_bytes\":" << allocation_block_bytes
           << ",\"uploaded_bytes\":" << uploaded << ",\"verified_bytes\":" << verified
           << ",\"scanned_bytes\":" << scanned << ",\"scanned_blocks\":" << scanned_blocks
           << ",\"mismatched_bytes\":" << mismatched_bytes
           << ",\"mismatched_blocks\":" << mismatched_blocks
           << ",\"mismatched_ranges\":" << mismatched_ranges
           << ",\"size_mismatch_blocks\":" << size_mismatch_blocks
           << ",\"probe_limit\":" << (probe_limit ? "true" : "false")
           << ",\"rejected_block_bytes\":" << rejected << ",\"retained_tensors\":" << tensors
           << ",\"rejection_phase\":";
    json_string(output, rejection_phase);
    output << ",\"allocation_error\":";
    json_string(output, allocation_error);
    output << ',';
    outcome.write_fields(output);
    output << "}\n";
    output.close();
    if (!output) {
      throw std::runtime_error("cannot save capacity progress");
    }
  }

  void reject(std::size_t bytes, const std::string &message) {
    rejected = bytes;
    allocation_error = message;
    rejection_phase = outcome.phase;
    std::cout << outcome.phase << " rejected after " << uploaded / mib << " MiB: " << message
              << std::endl;
  }

  std::string status() const {
    if (!outcome.error.empty() || !outcome.released || !outcome.release_error.empty()) {
      return "failed";
    }
    if (outcome.mismatch) {
      return "readback_mismatch";
    }
    return rejected ? "allocation_limit" : "target_reached";
  }
};

struct Block {
  vsi_nn_tensor_id_t tensor;
  std::size_t bytes;
};

void write_ranges(std::ostream &output, const std::vector<ByteRange> &ranges) {
  output << '[';
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << '[' << ranges[index].begin << ',' << ranges[index].end << ']';
  }
  output << ']';
}

void write_scan(std::ostream &output, const Block &block, unsigned index,
                const std::vector<std::uint8_t> &actual, const ReadbackScan &scan) {
  output << "{\"block_index\":" << index << ",\"tensor_id\":" << block.tensor
         << ",\"payload_offset_bytes\":" << std::size_t(index) * allocation_block_bytes
         << ",\"expected_bytes\":" << block.bytes << ",\"actual_bytes\":" << actual.size()
         << ",\"mismatched_bytes\":" << scan.mismatched_bytes
         << ",\"byte_range_count\":" << scan.range_count << ",\"byte_ranges\":";
  write_ranges(output, scan.ranges);
  output << ",\"byte_ranges_truncated\":"
         << (scan.ranges.size() < scan.range_count ? "true" : "false")
         << ",\"logical_page_size\":4096,\"bad_page_ranges\":";
  write_ranges(output, scan.page_ranges);
  output << "}\n";
  output.flush();
  if (!output) {
    throw std::runtime_error("cannot save readback scan");
  }
}

std::vector<Block> allocate_blocks(Graph &graph, CapacityProgress &progress) {
  std::vector<Block> retained;
  while (progress.uploaded < progress.target) {
    const auto bytes = std::min(allocation_block_bytes, progress.target - progress.uploaded);
    const auto data = allocation_data(progress.dtype, retained.size(), bytes);
    const TensorSpec spec{progress.dtype,
                          {1024, std::uint32_t(bytes / (1024 * dtype_bytes(progress.dtype)))}};
    progress.outcome.phase = "allocate";
    progress.save();
    vsi_nn_tensor_id_t tensor;
    // Only SDK calls are caught as allocation rejections. Pattern generation,
    // file IO and report errors must fail the experiment, not establish a bound.
    try {
      tensor = progress.storage == Storage::Constant ? add_tensor(graph, spec, true, data)
                                                     : add_tensor(graph, spec);
    } catch (const std::runtime_error &error) {
      progress.reject(bytes, error.what());
      break;
    }
    if (progress.storage == Storage::Mutable) {
      progress.outcome.phase = "upload";
      progress.save();
      try {
        upload_tensor(graph, tensor, data);
      } catch (const std::runtime_error &error) {
        progress.reject(bytes, error.what());
        break;
      }
    }
    retained.push_back({tensor, bytes});
    progress.uploaded += bytes;
    progress.tensors = retained.size();
    progress.save();
    if (progress.uploaded % (128 * mib) == 0 || progress.uploaded == progress.target) {
      std::cout << "retained " << progress.uploaded / mib << " MiB" << std::endl;
    }
  }
  return retained;
}

void verify_blocks(Graph &graph, const std::vector<Block> &retained, CapacityProgress &progress) {
  if (progress.readback == "none") {
    return;
  }
  std::ofstream scans;
  if (progress.readback == "all") {
    scans.open(progress.outcome.report.parent_path() / "readback-blocks.jsonl");
    if (!scans) {
      throw std::runtime_error("cannot open readback scan");
    }
  }
  // Scan every retained block after allocation stops, including after SDK rejection.
  // Never stop at the first corrupt block in all mode: later bytes are part of P5.
  progress.outcome.phase = "readback";
  std::cout << "Readback: " << retained.size() << " retained blocks, " << progress.uploaded / mib
            << " MiB; "
            << (progress.readback == "all" ? "scan all bytes, continue after mismatches"
                                           : "stop at the first mismatched block")
            << std::endl;
  progress.save();
  for (unsigned index = 0; index < retained.size(); ++index) {
    const auto &block = retained[index];
    const auto actual = read_tensor(graph, block.tensor);
    const auto expected = allocation_data(progress.dtype, index, block.bytes);
    const bool matched = progress.outcome.verify(
        {"block." + std::to_string(index), block.tensor, index, progress.dtype, 0}, actual,
        expected);
    progress.scanned += block.bytes;
    ++progress.scanned_blocks;
    progress.size_mismatch_blocks += actual.size() != block.bytes;
    if (progress.readback == "all") {
      const auto scan = scan_readback(actual, expected);
      write_scan(scans, block, index, actual, scan);
      progress.mismatched_bytes += scan.mismatched_bytes;
      progress.mismatched_ranges += scan.range_count;
      for (const auto &range : scan.ranges) {
        std::cout << "block " << index << ": local bytes [" << range.begin << ", " << range.end
                  << "), length " << range.end - range.begin << "; logical payload offset "
                  << std::size_t(index) * allocation_block_bytes + range.begin << std::endl;
      }
    }
    if (!matched) {
      ++progress.mismatched_blocks;
      std::cout << "readback mismatch: block=" << index << " (see retained scan/evidence)"
                << std::endl;
      if (progress.readback == "first") {
        break;
      }
    } else {
      progress.verified += block.bytes;
    }
    progress.save();
  }
}

void probe(CapacityProgress &progress) {
  progress.save();
  std::cout << "Requested payload: " << progress.target / mib
            << " MiB; storage: " << storage_name(progress.storage)
            << "; dtype: " << dtype_name(progress.dtype)
            << "; block: " << allocation_block_bytes / mib << " MiB\n"
            << "Allocate/write all blocks before readback; retain them until the scan finishes.\n"
            << "Offsets refer to application payload, not physical addresses." << std::endl;
  // Validate the diagnostic exception before opening the device. Ordinary model
  // budgets cannot exceed 1 GiB; only --probe-limit uses this separate constructor.
  auto budget =
      progress.probe_limit ? MemoryBudget::for_capacity_probe(progress.target) : MemoryBudget();
  Context context;
  context.memory() = budget;
  Graph graph(context, (progress.target + allocation_block_bytes - 1) / allocation_block_bytes + 1,
              1);
  try {
    const auto retained = allocate_blocks(graph, progress);
    verify_blocks(graph, retained, progress);
  } catch (const std::exception &error) {
    progress.outcome.fail(error.what());
  }
  progress.outcome.release(graph, context, [&] { progress.save(); });
  progress.save(progress.status());
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 5 || argc > 7) {
    std::cerr << "usage: np101_capacity_check REPORT_JSON constant|mutable F16|F32 TARGET_MIB "
                 "[first|all|none] [--probe-limit]\n";
    return 2;
  }
  CapacityProgress progress;
  progress.outcome.report = argv[1];
  try {
    progress.storage = parse_storage(argv[2]);
    progress.dtype = parse_dtype(argv[3]);
    if (argc >= 6) {
      progress.readback = argv[5];
      if (progress.readback != "first" && progress.readback != "all" &&
          progress.readback != "none") {
        throw std::invalid_argument("readback must be first, all or none");
      }
    }
    if (argc == 7) {
      if (std::string(argv[6]) != "--probe-limit") {
        throw std::invalid_argument("expected --probe-limit");
      }
      progress.probe_limit = true;
    }
    const std::string target(argv[4]);
    if (target.empty() || target.find_first_not_of("0123456789") != std::string::npos ||
        (progress.dtype != DataType::Float16 && progress.dtype != DataType::Float32)) {
      throw std::invalid_argument("expected F16/F32 and an integer target");
    }
    const auto size = std::stoul(target);
    const auto limit_mib = progress.probe_limit ? 4096 : segment_payload_limit / mib;
    if (size == 0 || size > limit_mib) {
      throw std::invalid_argument(
          "target must be 1..1024 MiB, or up to 4096 MiB with --probe-limit");
    }
    progress.target = size * mib;
    probe(progress);
    return progress.status() == "target_reached" || progress.status() == "allocation_limit" ? 0 : 1;
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
