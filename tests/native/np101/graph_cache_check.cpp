#include "binary_io.hpp"
#include "np101/diagnostics.hpp"
#include "np101/ops/cache_update.hpp"
#include "np101/ops/graph_builder.hpp"
#include "np101/ops/mixers.hpp"
#include "np101/tensor.hpp"
#include "test_support.hpp"
#include "vsi_nn_pub.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace specferry::np101;
using namespace specferry::np101::ops;
using namespace specferry::testing;
constexpr auto f16 = DataType::Float16;
constexpr unsigned packed_dimension = 64, packed_heads = 16;
constexpr unsigned packed_width = packed_dimension * packed_heads;

std::vector<float> packed_queries(unsigned block, bool legacy_single_query = false) {
  std::vector<float> result(packed_width * block);
  for (unsigned token = 0; token < block; ++token) {
    for (unsigned head = 0; head < packed_heads; ++head) {
      const auto dimension = (head * 3 + token * 5) % packed_dimension;
      result[token * packed_width + head * packed_dimension + dimension] =
          legacy_single_query ? 0.5f : 0.25f * (1 + (head + token) % 3);
    }
  }
  return result;
}

class PackedCacheGraph : public GraphBuilder {
public:
  Tensor key_column{}, value_column{}, slot{}, limit, attended, reordered{};
  std::vector<Tensor> declared_inputs;
  const bool writes;

  PackedCacheGraph(Context &context, Graph &storage, const std::array<Tensor, 2> &parents,
                   unsigned cache_capacity, bool writer)
      : GraphBuilder(context, 64, 32), limit(tensor({DataType::Int32, {1}})), writes(writer) {
    if (writes) {
      key_column = tensor({f16, {packed_width, 1}});
      value_column = tensor(key_column.spec);
      slot = tensor({DataType::Int32, {1}});
      declared_inputs = {key_column, value_column, slot, limit};
    } else {
      declared_inputs = {limit};
    }
    std::array<Tensor, 2> batches;
    for (unsigned kind = 0; kind < parents.size(); ++kind) {
      auto backing = bind({storage, parents[kind].id}, parents[kind].spec);
      if (writes) {
        backing = indexed_append(*this, kind ? value_column : key_column, slot, backing);
      } else {
        declared_inputs.push_back(backing);
      }
      // These are graph edges, so attention observes the append before its reads.
      batches[kind] =
          permute(reshape(backing, {packed_dimension, packed_heads, cache_capacity}), {0, 2, 1});
    }
    attended = attention_core(
                   *this, floats(packed_queries(1, true), {packed_dimension, 1, packed_heads}, f16),
                   batches[0], batches[1], limit, 1)
                   .attended;
    if (writes) {
      // One coarse read checks both full reordered caches, including untouched slots.
      reordered = concat(batches[0], batches[1], 2);
    }
  }

  void verify() {
    compile(declared_inputs,
            writes ? std::vector<Tensor>{attended, reordered} : std::vector<Tensor>{attended});
  }

  void run() {
    if (!vxIsGraphVerified(graph.get()->g)) {
      throw std::runtime_error("packed cache graph lost its fixed bindings");
    }
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.get()); }),
          "run packed cache graph");
  }
};

class PackedCacheReport {
  const std::filesystem::path directory_;
  const unsigned planned_checks_;
  std::ofstream checks_;
  unsigned completed_ = 0;

  void record(const std::string &name, bool passed, std::size_t bytes, float error) {
    checks_ << name << '\t' << passed << '\t' << bytes << '\t' << error << '\n';
    checks_.flush();
    if (!checks_) {
      throw std::runtime_error("cannot save packed cache comparison");
    }
    ++completed_;
    if (!passed) {
      phase("numerical_failure");
      throw std::runtime_error("packed cache comparison failed: " + name);
    }
  }

public:
  explicit PackedCacheReport(std::filesystem::path directory, unsigned planned_checks = 20)
      : directory_(std::move(directory)), planned_checks_(planned_checks),
        checks_(directory_ / "checks.tsv") {
    checks_ << "check\tpassed\tbytes\tmax_abs_error\n";
  }

  void phase(const std::string &name, bool released = false) const {
    write_text(directory_ / "execution.json",
               "{\"phase\":\"" + name + "\",\"checks\":" + std::to_string(completed_) +
                   ",\"planned_checks\":" + std::to_string(planned_checks_) +
                   ",\"released\":" + (released ? "true" : "false") + "}\n");
  }

  void condition(const std::string &name, bool passed) { record(name, passed, 0, 0); }

  void exact(const std::string &name, const std::vector<std::uint8_t> &actual,
             const std::vector<std::uint8_t> &expected) {
    write_bytes(directory_ / (name + ".bin"), actual);
    write_bytes(directory_ / (name + ".expected.bin"), expected);
    record(name, actual == expected, actual.size(), 0);
  }

  void attention(const std::string &name, const std::vector<std::uint8_t> &actual,
                 const std::vector<float> &expected) {
    write_bytes(directory_ / (name + ".bin"), actual);
    write_bytes(directory_ / (name + ".expected.bin"), encode_floats(expected));
    bool passed = actual.size() == expected.size() * sizeof(std::uint16_t);
    float maximum_error = 0;
    if (passed) {
      for (std::size_t index = 0; index < expected.size(); ++index) {
        std::uint16_t bits;
        std::memcpy(&bits, actual.data() + index * sizeof(bits), sizeof(bits));
        const auto value = vsi_nn_Fp16ToFp32(bits);
        const auto error = std::abs(value - expected[index]);
        maximum_error = std::max(maximum_error, error);
        // Storage checks remain byte-exact. Only arithmetic uses FP16 tolerance.
        if (!std::isfinite(value) || error > 0.008f + 0.005f * std::abs(expected[index])) {
          passed = false;
        }
      }
    }
    record(name, passed, actual.size(), maximum_error);
  }

  void complete(Context &context, std::ostringstream &summary, TensorTransfers before,
                TensorTransfers expected, std::size_t const_peak, std::size_t nonconst_peak,
                const char *failure) {
    const auto after = tensor_transfers();
    const std::pair<const char *, std::size_t TensorTransfers::*> fields[]{
        {"reads", &TensorTransfers::reads},
        {"read_bytes", &TensorTransfers::read_bytes},
        {"uploads", &TensorTransfers::uploads},
        {"upload_bytes", &TensorTransfers::upload_bytes}};
    for (const auto &[name, field] : fields) {
      if (after.*field - before.*field != expected.*field) {
        phase("transfer_count_failure", true);
        throw std::runtime_error(failure);
      }
    }
    for (const bool planned : {false, true}) {
      for (const auto &[name, field] : fields) {
        summary << (planned ? "expected_" : "") << name << ": "
                << (planned ? expected.*field : after.*field - before.*field) << '\n';
      }
    }
    summary << "const_peak_bytes: " << const_peak << "\nnonconst_peak_bytes: " << nonconst_peak
            << "\nconst_live_after_release: " << context.memory().live(true)
            << "\nnonconst_live_after_release: " << context.memory().live(false)
            << "\ngraph_released: true\n";
    write_text(directory_ / "cache.txt", summary.str());
    phase("complete", true);
  }
};

std::uint16_t packed_pattern(unsigned kind, unsigned position, unsigned head, unsigned dim,
                             unsigned revision) {
  const auto index = (position * packed_heads + head) * packed_dimension + dim;
  // Each 16-slot block permutes 16384 finite normal FP16 values below four.
  // Larger caches offset each block so a 16-slot addressing wrap cannot pass.
  // Block zero preserves the original C16 pattern exactly.
  const auto block_offset = (index / 16384) * 257;
  const auto payload = (index * 4051 + block_offset + revision * 997 + kind * 7919) & 0x3fff;
  return std::uint16_t(0x0400 + payload + (((kind + revision) & 1) ? 0x8000 : 0));
}

void fill_packed_cache(std::vector<std::uint16_t> &cache, unsigned kind, unsigned first,
                       unsigned count, unsigned revision) {
  for (unsigned position = first; position < first + count; ++position) {
    for (unsigned head = 0; head < packed_heads; ++head) {
      for (unsigned dim = 0; dim < packed_dimension; ++dim) {
        cache[position * packed_width + head * packed_dimension + dim] =
            packed_pattern(kind, position, head, dim, revision);
      }
    }
  }
}

std::vector<float>
packed_attention_reference(const std::array<std::vector<std::uint16_t>, 2> &cache,
                           const std::vector<float> &query, unsigned position) {
  const auto block = query.size() / packed_width;
  std::vector<float> output(query.size());
  // Compute the mathematical attention directly in token/head/slot order.
  // This reference uses no SDK tensor layout, reshape, permutation or append.
  for (unsigned token = 0; token < block; ++token) {
    const auto limit = position + token + 1;
    for (unsigned head = 0; head < packed_heads; ++head) {
      std::vector<float> probabilities(limit);
      float maximum = -1e9f;
      for (unsigned slot = 0; slot < limit; ++slot) {
        float score = 0;
        for (unsigned dimension = 0; dimension < packed_dimension; ++dimension) {
          const auto query_index = token * packed_width + head * packed_dimension + dimension;
          const auto cache_index = slot * packed_width + head * packed_dimension + dimension;
          score += query[query_index] * vsi_nn_Fp16ToFp32(cache[0][cache_index]);
        }
        // Match only the public precision boundaries: scores and probabilities
        // are FP16 around the FP32 softmax, and the final GEMM stores FP16.
        probabilities[slot] = vsi_nn_Fp16ToFp32(vsi_nn_Fp32ToFp16(score));
        maximum = std::max(maximum, probabilities[slot]);
      }
      float denominator = 0;
      for (auto &probability : probabilities) {
        probability = std::exp(probability - maximum);
        denominator += probability;
      }
      for (auto &probability : probabilities) {
        probability = vsi_nn_Fp16ToFp32(vsi_nn_Fp32ToFp16(probability / denominator));
      }
      for (unsigned dimension = 0; dimension < packed_dimension; ++dimension) {
        float value = 0;
        for (unsigned slot = 0; slot < limit; ++slot) {
          const auto cache_index = slot * packed_width + head * packed_dimension + dimension;
          value += probabilities[slot] * vsi_nn_Fp16ToFp32(cache[1][cache_index]);
        }
        output[token * packed_width + head * packed_dimension + dimension] =
            vsi_nn_Fp16ToFp32(vsi_nn_Fp32ToFp16(value));
      }
    }
  }
  return output;
}

std::vector<std::uint8_t>
packed_reordered_reference(const std::array<std::vector<std::uint16_t>, 2> &expected,
                           unsigned cache_capacity) {
  std::vector<std::uint16_t> output;
  output.reserve(2 * packed_width * cache_capacity);
  for (const auto &cache : expected) {
    for (unsigned head = 0; head < packed_heads; ++head) {
      for (unsigned position = 0; position < cache_capacity; ++position) {
        const auto offset = position * packed_width + head * packed_dimension;
        output.insert(output.end(), cache.begin() + offset,
                      cache.begin() + offset + packed_dimension);
      }
    }
  }
  return as_bytes(output);
}

void check_packed_cache(Context &context, const std::filesystem::path &output,
                        unsigned cache_capacity) {
  if (cache_capacity != 16 && cache_capacity != 64) {
    throw std::invalid_argument("packed cache calibration requires capacity 16 or 64");
  }
  PackedCacheReport report(output);
  report.phase("setup.storage");
  const auto transfers_before = tensor_transfers();
  Graph storage(context, 2, 0);
  const TensorSpec backing_spec{f16, {packed_width, cache_capacity}};
  const std::array<unsigned, 4> slots{0, cache_capacity / 2 - 1, cache_capacity - 1,
                                      cache_capacity / 2 - 1};
  const auto column_bytes = std::size_t(packed_width) * sizeof(std::uint16_t);
  const auto expected_reads = slots.size() * 5;
  const auto expected_read_bytes = slots.size() * (2 * column_bytes + 4 * backing_spec.bytes());
  const auto expected_uploads = slots.size() * 5;
  const auto expected_upload_bytes = slots.size() * (2 * column_bytes + 3 * sizeof(std::int32_t));
  std::array<Tensor, 2> parents;
  std::array<std::vector<std::uint16_t>, 2> expected;
  for (unsigned kind = 0; kind < parents.size(); ++kind) {
    expected[kind].resize(backing_spec.elements());
    fill_packed_cache(expected[kind], kind, 0, cache_capacity, 0);
    parents[kind] = {add_tensor(storage, backing_spec, false, as_bytes(expected[kind])),
                     backing_spec};
  }
  PackedCacheGraph producer(context, storage, parents, cache_capacity, true);
  PackedCacheGraph reader(context, storage, parents, cache_capacity, false);
  report.phase("verify.reader.before_producer");
  reader.verify();
  report.phase("verify.producer");
  producer.verify();
  const auto const_peak = context.memory().peak(true);
  const auto nonconst_peak = context.memory().peak(false);

  unsigned revision = 0;
  for (unsigned position : slots) {
    ++revision;
    const auto prefix = "step." + std::to_string(revision - 1);
    report.phase(prefix + ".upload");
    for (unsigned kind = 0; kind < parents.size(); ++kind) {
      fill_packed_cache(expected[kind], kind, position, 1, revision);
      const auto begin = expected[kind].begin() + position * packed_width;
      const std::vector<std::uint16_t> column(begin, begin + packed_width);
      upload_tensor(producer.graph, kind ? producer.value_column.id : producer.key_column.id,
                    as_bytes(column));
    }
    const auto limit = position + 1;
    upload_tensor(producer.graph, producer.slot.id,
                  as_bytes<std::int32_t>({std::int32_t(position)}));
    upload_tensor(producer.graph, producer.limit.id, as_bytes<std::int32_t>({std::int32_t(limit)}));
    upload_tensor(reader.graph, reader.limit.id, as_bytes<std::int32_t>({std::int32_t(limit)}));
    report.phase(prefix + ".run.producer");
    producer.run();
    report.phase(prefix + ".run.preverified_reader");
    reader.run();

    // Both computations finish before any readback. Validate them first so host
    // reading either cache cannot accidentally provide the required visibility.
    report.phase(prefix + ".compare.attention");
    const auto reference = packed_attention_reference(expected, packed_queries(1, true), position);
    report.attention(prefix + ".producer_attention",
                     read_tensor(producer.graph, producer.attended.id), reference);
    report.attention(prefix + ".preverified_reader_attention",
                     read_tensor(reader.graph, reader.attended.id), reference);
    report.phase(prefix + ".compare.layout_and_backing");
    report.exact(prefix + ".reordered_keys_values",
                 read_tensor(producer.graph, producer.reordered.id),
                 packed_reordered_reference(expected, cache_capacity));
    for (unsigned kind = 0; kind < parents.size(); ++kind) {
      report.exact(prefix + (kind ? ".values_backing" : ".keys_backing"),
                   read_tensor(storage, parents[kind].id), as_bytes(expected[kind]));
    }
  }
  report.phase("release");
  reader.graph.close();
  producer.graph.close();
  storage.close();
  std::ostringstream summary;
  summary << "status: numerical_pass\nmode: " << (cache_capacity == 16 ? "packed" : "packed64")
          << "\nheads: " << packed_heads << "\nhead_dimension: " << packed_dimension
          << "\ncapacity: " << cache_capacity << "\nslots: " << slots[0] << ',' << slots[1] << ','
          << slots[2] << ',' << slots[3] << "\nchecks: 20\nlaunches: 8\nrevalidations: 0\n"
          << "reader_verified_before_producer: true\ncomputations_before_cache_readback: true\n";
  report.complete(context, summary, transfers_before,
                  {expected_uploads, expected_upload_bytes, expected_reads, expected_read_bytes},
                  const_peak, nonconst_peak,
                  "packed cache transfer counts differ from the planned scope");
}

class PackedBlockGraph : public GraphBuilder {
public:
  const unsigned block;
  const bool writes;
  Tensor keys{}, values{}, index{}, limits, attended, reordered{};
  std::vector<Tensor> declared_inputs;

  PackedBlockGraph(Context &context, Graph &storage, const std::array<Tensor, 2> &parents,
                   unsigned cache_capacity, unsigned tokens, bool writer)
      : GraphBuilder(context, 128, 64), block(tokens), writes(writer),
        limits(tensor({DataType::Int32, {block}})) {
    if ((block != 1 && block != 4) || cache_capacity % block) {
      throw std::invalid_argument("packed block calibration requires aligned block 1 or 4");
    }
    declared_inputs = {limits};
    if (writes) {
      keys = tensor({f16, {packed_width, block}});
      values = tensor(keys.spec);
      index = tensor({DataType::Int32, {1}});
      declared_inputs.insert(declared_inputs.end(), {keys, values, index});
    }
    std::array<Tensor, 2> batches;
    for (unsigned kind = 0; kind < parents.size(); ++kind) {
      auto backing = bind({storage, parents[kind].id}, parents[kind].spec);
      if (writes) {
        // This fixed OpenVX storage alias is created before either writer is
        // verified. A graph RESHAPE would copy and cannot provide shared writes.
        if (block != 1) {
          backing = storage_reshape(*this, backing, {packed_width * block, cache_capacity / block});
        }
        auto columns = kind ? values : keys;
        if (block != 1) {
          columns = reshape(columns, {packed_width * block, 1});
        }
        backing = indexed_append(*this, columns, index, backing);
      } else {
        declared_inputs.push_back(backing);
      }
      // Use graph edges after append: a second storage-only alias would hide
      // the producer dependency from the reader operators in this graph.
      batches[kind] =
          permute(reshape(backing, {packed_dimension, packed_heads, cache_capacity}), {0, 2, 1});
    }
    auto query = floats(packed_queries(block), {packed_width, block}, f16);
    query = reshape(query, {packed_dimension, packed_heads, block});
    query = permute(query, {0, 2, 1});
    auto result = attention_core(*this, query, batches[0], batches[1], limits, 1).attended;
    // Restore token-major [token, head, dimension] bytes before the comparison.
    result = permute(reshape(result, {packed_dimension, block, packed_heads}), {0, 2, 1});
    attended = reshape(result, {packed_width, block});
    if (writes) {
      reordered = concat(batches[0], batches[1], 2);
    }
  }

  void verify() {
    compile(declared_inputs,
            writes ? std::vector<Tensor>{attended, reordered} : std::vector<Tensor>{attended});
  }

  void run() {
    if (!vxIsGraphVerified(graph.get()->g)) {
      throw std::runtime_error("packed block graph lost its fixed bindings");
    }
    check(sdk_call("vsi_nn_RunGraph", [&] { return vsi_nn_RunGraph(graph.get()); }),
          "run packed block graph");
  }
};

void check_packed_block_cache(Context &context, const std::filesystem::path &output) {
  constexpr unsigned cache_capacity = 16, prefill_block = 4;

  struct Step {
    unsigned position;
    unsigned block;
    unsigned revision;
    bool change_last_only = false;
  };

  // Leave slots 9..11 zero until the last step. Full backing comparisons check
  // that updates preserve both existing history and every unwritten slot.
  const std::array<Step, 8> steps{{{0, 4, 1},
                                   {4, 4, 2},
                                   {8, 1, 3},
                                   {12, 4, 4},
                                   {4, 4, 5},
                                   {4, 4, 6, true},
                                   {7, 1, 7},
                                   {8, 4, 8}}};
  const auto planned_checks = unsigned(steps.size() * 5 + 2);
  PackedCacheReport report(output, planned_checks);
  report.phase("setup.storage");
  const auto before = tensor_transfers();
  Graph storage(context, 2, 0);
  const TensorSpec spec{f16, {packed_width, cache_capacity}};
  std::array<Tensor, 2> parents;
  std::array<std::vector<std::uint16_t>, 2> expected;
  for (unsigned kind = 0; kind < parents.size(); ++kind) {
    expected[kind].resize(spec.elements());
    parents[kind] = {add_tensor(storage, spec, false, as_bytes(expected[kind])), spec};
  }
  PackedBlockGraph decode(context, storage, parents, cache_capacity, 1, true);
  PackedBlockGraph prefill(context, storage, parents, cache_capacity, prefill_block, true);
  PackedBlockGraph decode_reader(context, storage, parents, cache_capacity, 1, false);
  PackedBlockGraph prefill_reader(context, storage, parents, cache_capacity, prefill_block, false);
  report.phase("verify.readers.before_writers");
  decode_reader.verify();
  prefill_reader.verify();
  report.phase("verify.decode");
  decode.verify();
  report.phase("verify.prefill");
  prefill.verify();
  const auto const_peak = context.memory().peak(true);
  const auto nonconst_peak = context.memory().peak(false);
  const auto token_bytes = std::size_t(packed_width) * sizeof(std::uint16_t);
  std::size_t consumed_tokens = 0;
  std::vector<std::uint8_t> before_late_change;

  for (std::size_t step_index = 0; step_index < steps.size(); ++step_index) {
    const auto &step = steps[step_index];
    if (step.position % step.block || step.position + step.block > cache_capacity) {
      throw std::logic_error("invalid packed block calibration step");
    }
    auto &writer = step.block == 1 ? decode : prefill;
    auto &reader = step.block == 1 ? decode_reader : prefill_reader;
    const auto prefix = "step." + std::to_string(step_index);
    report.phase(prefix + ".upload");
    for (unsigned kind = 0; kind < parents.size(); ++kind) {
      const auto first = step.position + (step.change_last_only ? step.block - 1 : 0);
      fill_packed_cache(expected[kind], kind, first, step.position + step.block - first,
                        step.revision);
      const auto begin = expected[kind].begin() + step.position * packed_width;
      const std::vector<std::uint16_t> columns(begin, begin + packed_width * step.block);
      upload_tensor(writer.graph, kind ? writer.values.id : writer.keys.id, as_bytes(columns));
    }
    std::vector<std::int32_t> limits;
    for (unsigned token = 0; token < step.block; ++token) {
      limits.push_back(step.position + token + 1);
    }
    upload_tensor(writer.graph, writer.index.id,
                  as_bytes<std::int32_t>({std::int32_t(step.position / step.block)}));
    upload_tensor(writer.graph, writer.limits.id, as_bytes(limits));
    upload_tensor(reader.graph, reader.limits.id, as_bytes(limits));
    report.phase(prefix + ".run.writer");
    writer.run();
    report.phase(prefix + ".run.preverified_reader");
    reader.run();
    consumed_tokens += step.block;

    // Run both computations before any readback. Cache reads must not provide
    // the synchronization needed by either the writer's or the reader's math.
    report.phase(prefix + ".compare.attention");
    const auto reference =
        packed_attention_reference(expected, packed_queries(step.block), step.position);
    const auto actual = read_tensor(writer.graph, writer.attended.id);
    report.attention(prefix + ".writer_attention", actual, reference);
    report.attention(prefix + ".preverified_reader_attention",
                     read_tensor(reader.graph, reader.attended.id), reference);
    if (step_index == 4) {
      before_late_change = actual;
    } else if (step.change_last_only) {
      const auto early_bytes = (prefill_block - 1) * token_bytes;
      report.exact("causal.earlier_queries_unchanged",
                   {actual.begin(), actual.begin() + early_bytes},
                   {before_late_change.begin(), before_late_change.begin() + early_bytes});
      report.condition("causal.last_query_observes_changed_token",
                       !std::equal(actual.begin() + early_bytes, actual.end(),
                                   before_late_change.begin() + early_bytes));
    }
    report.phase(prefix + ".compare.layout_and_backing");
    report.exact(prefix + ".reordered_keys_values", read_tensor(writer.graph, writer.reordered.id),
                 packed_reordered_reference(expected, cache_capacity));
    for (unsigned kind = 0; kind < parents.size(); ++kind) {
      report.exact(prefix + (kind ? ".values_backing" : ".keys_backing"),
                   read_tensor(storage, parents[kind].id), as_bytes(expected[kind]));
    }
  }
  report.phase("release");
  prefill_reader.graph.close();
  decode_reader.graph.close();
  prefill.graph.close();
  decode.graph.close();
  storage.close();
  const auto expected_reads = steps.size() * 5;
  const auto expected_read_bytes =
      2 * consumed_tokens * token_bytes + 4 * steps.size() * spec.bytes();
  const auto expected_uploads = steps.size() * 5;
  const auto expected_upload_bytes = 2 * consumed_tokens * token_bytes +
                                     (steps.size() + 2 * consumed_tokens) * sizeof(std::int32_t);
  std::ostringstream summary;
  summary << "status: numerical_pass\nmode: packed-block4\nheads: " << packed_heads
          << "\nhead_dimension: " << packed_dimension << "\ncapacity: " << cache_capacity
          << "\nblock: " << prefill_block << "\nchecks: " << planned_checks
          << "\nlaunches: " << steps.size() * 2 << "\nconsumed_tokens: " << consumed_tokens
          << "\nrevalidations: 0\nreader_verified_before_producer: true\n"
          << "computations_before_cache_readback: true\n"
          << "storage_alias_api: vxReshapeTensor\n";
  report.complete(context, summary, before,
                  {expected_uploads, expected_upload_bytes, expected_reads, expected_read_bytes},
                  const_peak, nonconst_peak,
                  "packed block transfer counts differ from the planned scope");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 || (std::string(argv[2]) != "packed" && std::string(argv[2]) != "packed64" &&
                    std::string(argv[2]) != "packed-block4")) {
    std::cerr << "usage: np101_graph_cache_check OUTPUT packed|packed64|packed-block4\n";
    return 2;
  }
  return run_test(argv[1], [&] {
    Context context;
    if (std::string(argv[2]) == "packed-block4") {
      check_packed_block_cache(context, argv[1]);
    } else {
      check_packed_cache(context, argv[1], std::string(argv[2]) == "packed" ? 16 : 64);
    }
    context.close();
  });
}
