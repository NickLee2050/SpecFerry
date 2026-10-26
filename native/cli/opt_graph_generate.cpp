#include "inference/generation_io.hpp"
#include "models/opt/config.hpp"
#include "models/opt/partitioned_graph_model.hpp"
#include "np101/context.hpp"
#include "np101/diagnostics.hpp"
#include "np101/memory_budget.hpp"
#include "np101/tensor.hpp"
#include "np101/weights.hpp"
#include "platform/runtime_profile.hpp"
#include "sys/resource.h"
#include "sys/time.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace specferry;
using Clock = std::chrono::steady_clock;
using PartitionedModel = models::opt::PartitionedGraphModel;

struct ExecutionCounts {
  std::size_t decode_steps = 0;
  std::size_t prefill_steps = 0;
  std::size_t head_runs = 0;
};

struct CpuUsage {
  double process_user_seconds = 0;
  double process_system_seconds = 0;
  double main_thread_user_seconds = 0;
  double main_thread_system_seconds = 0;
};

struct CpuSnapshot {
  rusage process{};
  rusage main_thread{};
};

struct CpuPhaseBoundaries {
  std::optional<CpuSnapshot> prefill, decode;
};

CpuSnapshot read_cpu_snapshot() {
  CpuSnapshot snapshot;
  if (getrusage(RUSAGE_SELF, &snapshot.process) ||
      getrusage(RUSAGE_THREAD, &snapshot.main_thread)) {
    throw std::runtime_error("cannot read host CPU usage");
  }
  return snapshot;
}

double elapsed_cpu_seconds(const timeval &before, const timeval &after) {
  return static_cast<double>(after.tv_sec - before.tv_sec) +
         static_cast<double>(after.tv_usec - before.tv_usec) * 1e-6;
}

CpuUsage cpu_usage_between(const CpuSnapshot &before, const CpuSnapshot &after) {
  return {elapsed_cpu_seconds(before.process.ru_utime, after.process.ru_utime),
          elapsed_cpu_seconds(before.process.ru_stime, after.process.ru_stime),
          elapsed_cpu_seconds(before.main_thread.ru_utime, after.main_thread.ru_utime),
          elapsed_cpu_seconds(before.main_thread.ru_stime, after.main_thread.ru_stime)};
}

void write_cpu_usage(std::ostream &out, CpuUsage usage) {
  out << std::setprecision(17) << "{\"process_user_seconds\":" << usage.process_user_seconds
      << ",\"process_system_seconds\":" << usage.process_system_seconds
      << ",\"main_thread_user_seconds\":" << usage.main_thread_user_seconds
      << ",\"main_thread_system_seconds\":" << usage.main_thread_system_seconds << '}';
}

struct BenchmarkRun {
  std::string name;
  unsigned request_index;
  unsigned iteration;
  unsigned prompt_length;
  bool warmup;
  bool tokens_match;
};

ExecutionCounts execution_counts(const PartitionedModel &model, ExecutionCounts before = {}) {
  return {model.decode_steps() - before.decode_steps, model.prefill_steps() - before.prefill_steps,
          model.head_runs() - before.head_runs};
}

PartitionedModel::Variant parse_variant(const std::string &name) {
  using Variant = PartitionedModel::Variant;
  const std::pair<const char *, Variant> variants[] = {
      {"merged-kv", Variant::MergedKv},
      {"merged-kv-fc1-split2", Variant::MergedKvFc1Split2},
      {"compiled-ffn-finish", Variant::MergedKvCompiledFfn},
      {"compiled-decoder", Variant::CompiledDecoder},
      {"compiled-fullgraph", Variant::CompiledFullModel}};
  for (const auto &[label, variant] : variants) {
    if (name == label) {
      return variant;
    }
  }
  throw std::invalid_argument("unknown or retired partitioned graph variant: " + name);
}

bool same_transfers(np101::TensorTransfers before, np101::TensorTransfers after) {
  return before.uploads == after.uploads && before.upload_bytes == after.upload_bytes &&
         before.reads == after.reads && before.read_bytes == after.read_bytes;
}

template <class Exception>
void check_rejected_step(PartitionedModel &model, std::int32_t token, unsigned &checks,
                         unsigned &rejections, const char *message) {
  const auto transfers = np101::tensor_transfers();
  const auto launches = model.launches();
  ++checks;
  try {
    model.step(token);
  } catch (const Exception &) {
    ++rejections;
  }
  if (checks != rejections || !same_transfers(transfers, np101::tensor_transfers()) ||
      model.launches() != launches) {
    throw std::runtime_error(message);
  }
}

void save_diagnostic(const fs::path &path, const std::vector<std::uint8_t> &bytes,
                     const char *message) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  if (!out) {
    throw std::runtime_error(message);
  }
}

void write_memory(std::ostream &out, const np101::MemoryBudget &budget) {
  out << ",\"application_segment_limit_bytes\":" << np101::segment_payload_limit
      << ",\"const_live_bytes\":" << budget.live(true)
      << ",\"const_peak_bytes\":" << budget.peak(true)
      << ",\"nonconst_live_bytes\":" << budget.live(false)
      << ",\"nonconst_peak_bytes\":" << budget.peak(false);
}

void save_benchmark(const fs::path &directory, unsigned warmups, unsigned repeats,
                    unsigned requests, const std::vector<BenchmarkRun> &runs, bool passed) {
  std::ofstream out(directory / "benchmark.json");
  out << "{\"schema_version\":1,\"warmups\":" << warmups << ",\"repeats\":" << repeats
      << ",\"request_count\":" << requests << ",\"completed\":" << (passed ? "true" : "false")
      << ",\"runs\":[";
  for (std::size_t index = 0; index < runs.size(); ++index) {
    const auto &run = runs[index];
    out << (index ? "," : "") << "{\"name\":\"" << run.name
        << "\",\"request_index\":" << run.request_index << ",\"iteration\":" << run.iteration
        << ",\"prompt_length\":" << run.prompt_length << ",\"kind\":\""
        << (run.warmup ? "warmup" : "measured") << "\",\"result_file\":\"" << run.name
        << ".json\",\"tokens_match\":" << (run.tokens_match ? "true" : "false") << '}';
  }
  out << "]}\n";
  if (!out) {
    throw std::runtime_error("cannot save resident benchmark manifest");
  }
}

std::vector<std::vector<std::int32_t>> read_requests(const fs::path &path,
                                                     const models::opt::ModelConfig &config) {
  std::ifstream input(path);
  std::vector<std::vector<std::int32_t>> requests;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::vector<std::int32_t> tokens;
    std::int32_t token;
    while (stream >> token) {
      config.validate_token(token);
      tokens.push_back(token);
    }
    if (!stream.eof() || tokens.empty() || tokens.size() > config.decoder.capacity) {
      throw std::invalid_argument("invalid experimental request token list");
    }
    requests.push_back(tokens);
  }
  if (!input.eof() || requests.empty() || requests.size() > 64) {
    throw std::invalid_argument("expected one to 64 request lines");
  }
  return requests;
}

void progress(const fs::path &directory, const std::string &phase, const char *status = "running",
              bool released = false) {
  const auto temporary = directory / "execution.json.tmp";
  std::ofstream out(temporary);
  const auto now =
      std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  out << std::setprecision(17) << "{\"status\":\"" << status << "\",\"phase\":\"" << phase
      << "\",\"phase_started_at\":" << now << ",\"released\":" << (released ? "true" : "false")
      << "}\n";
  out.close();
  if (!out) {
    throw std::runtime_error("cannot save experimental progress");
  }
  fs::rename(temporary, directory / "execution.json");
}

void save_generation(const fs::path &path, const inference::Generation &result,
                     std::size_t launches, np101::TensorTransfers before,
                     np101::TensorTransfers after, unsigned prefill_block, ExecutionCounts counts,
                     const CpuSnapshot &cpu_before, const CpuSnapshot &cpu_after,
                     const CpuPhaseBoundaries &boundaries) {
  const auto cpu_usage = cpu_usage_between(cpu_before, cpu_after);
  std::ofstream out(path);
  out << '{';
  inference::write_generation(out, result);
  out << ",\"launches\":" << launches << ",\"uploads\":" << after.uploads - before.uploads
      << ",\"upload_bytes\":" << after.upload_bytes - before.upload_bytes
      << ",\"reads\":" << after.reads - before.reads
      << ",\"read_bytes\":" << after.read_bytes - before.read_bytes
      << ",\"prefill_block\":" << prefill_block << ",\"decode_steps\":" << counts.decode_steps
      << ",\"prefill_steps\":" << counts.prefill_steps << ",\"head_runs\":" << counts.head_runs
      << std::setprecision(17) << ",\"process_user_cpu_seconds\":" << cpu_usage.process_user_seconds
      << ",\"process_system_cpu_seconds\":" << cpu_usage.process_system_seconds
      << ",\"main_thread_user_cpu_seconds\":" << cpu_usage.main_thread_user_seconds
      << ",\"main_thread_system_cpu_seconds\":" << cpu_usage.main_thread_system_seconds
      << ",\"cpu_phases\":{\"reset\":";
  write_cpu_usage(out, cpu_usage_between(cpu_before, boundaries.prefill.value_or(cpu_after)));
  out << ",\"prefill\":";
  write_cpu_usage(out, boundaries.prefill ? cpu_usage_between(*boundaries.prefill,
                                                              boundaries.decode.value_or(cpu_after))
                                          : CpuUsage{});
  out << ",\"decode\":";
  write_cpu_usage(out, boundaries.decode ? cpu_usage_between(*boundaries.decode, cpu_after)
                                         : CpuUsage{});
  out << "}}\n";
  if (!out) {
    throw std::runtime_error("cannot save experimental generation");
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: specferry_opt_graph_generate DEPLOYMENT REQUEST OUTPUT\n";
    return 2;
  }
  const fs::path root(argv[3]);
  auto directory = root;
  try {
    if (std::getenv("SPECFERRY_VERIFY_WEIGHT_PRELOAD")) {
      throw std::invalid_argument("the weight preload diagnostic is retired");
    }
    const auto options = inference::read_run_options(argv[2], true);
    const auto benchmark = options.benchmark, lifecycle_check = options.lifecycle;
    const auto warmups = options.warmups, repeats = options.repeats;
    const auto *guard = std::getenv("SPECFERRY_VERIFY_REQUIRE_OPERATIONS");
    if (lifecycle_check && (!guard || std::string(guard) != "1")) {
      throw std::invalid_argument("lifecycle diagnostics require the pinned compilation guard");
    }
    const auto block = options.block, maximum = options.maximum;
    const auto &readback = options.readback;
    const auto readback_method = options.readback_transport == "mapped"
                                     ? np101::TensorReadback::MappedHost
                                     : np101::TensorReadback::SdkCopy;
    const auto &variant_name = options.variant;
    const auto variant = parse_variant(variant_name);
    const auto &head_policy_name = options.prefill_head;
    const auto head_policy = head_policy_name == "last-token"
                                 ? PartitionedModel::PrefillHeadPolicy::LastToken
                                 : PartitionedModel::PrefillHeadPolicy::EveryToken;
    const auto config = models::opt::read_model_config(argv[2]);
    const auto kv_transfer = options.kv_transfer == "mapped-slot"
                                 ? PartitionedModel::KvTransfer::MappedSlot
                                 : PartitionedModel::KvTransfer::DeviceCopy;
    if (kv_transfer == PartitionedModel::KvTransfer::MappedSlot && config.decoder.capacity != 64 &&
        config.decoder.capacity != 128) {
      throw std::invalid_argument("mapped-slot KV requires the C64/C128 layout");
    }
    const auto cache_first = options.cache_first, cache_last = options.cache_last;
    if (cache_last > config.decoder.layers) {
      throw std::invalid_argument("cache layer range must be within the configured model");
    }
    const auto requests = read_requests(fs::path(argv[2]) / "requests.txt", config);
    const auto expected = read_requests(fs::path(argv[2]) / "expected.txt", config);
    if (expected.size() != requests.size()) {
      throw std::invalid_argument("expected token rows must match request rows");
    }
    for (const auto &tokens : expected) {
      if (tokens.size() > maximum) {
        throw std::invalid_argument("expected tokens exceed generation limit");
      }
    }
    const auto preparation_started = Clock::now();
    np101::WeightStore weights(argv[1]);
    weights.verify();
    std::vector<models::opt::CompiledFfn> compiled_ffn, compiled_prefill_ffn;
    std::vector<models::opt::CompiledPrefillLayer> compiled_prefill_layers;
    models::opt::CompiledHead compiled_head;
    std::vector<models::opt::CompiledProjections> compiled_projections;
    models::opt::CompiledModelPackage compiled_model;
    models::opt::ProjectionPolicy projections;
    if (variant == PartitionedModel::Variant::MergedKvCompiledFfn) {
      std::ifstream manifest(fs::path(argv[2]) / "compiled-ffn-package.txt");
      std::string package, extra;
      if (!(manifest >> std::quoted(package)) || manifest >> extra) {
        throw std::invalid_argument("compiled FFN variant requires one package path");
      }
      compiled_ffn = models::opt::load_compiled_ffn(package, argv[1], config, weights,
                                                    compiled_head, &compiled_projections);
      if (block != 1) {
        compiled_prefill_ffn =
            models::opt::load_compiled_prefill(package, argv[1], config, weights);
        if (compiled_head.graph.path.empty() ||
            (options.projection_backend != "auto" && options.projection_backend != "fcl" &&
             options.projection_backend != "nbg")) {
          throw std::invalid_argument(
              "M16 fast chain requires one NBG head and M1 FCL or NBG projections");
        }
      }
      projections = {np101::ops::LinearBackend::FullyConnected,
                     np101::ops::LinearBackend::FullyConnected};
      const auto backend = options.projection_backend == "auto" && !compiled_projections.empty()
                               ? std::string{"nbg"}
                               : options.projection_backend;
      if (compiled_projections.empty() && backend != "auto" && backend != "fcl") {
        throw std::invalid_argument(
            "NBG projection selection requires packaged projection artifacts");
      }
      projections.compiled_query = backend == "nbg" || backend == "nbg-qkv";
      projections.compiled_output = backend == "nbg" || backend == "nbg-output";
    }
    if (variant == PartitionedModel::Variant::CompiledDecoder ||
        variant == PartitionedModel::Variant::CompiledFullModel) {
      std::ifstream manifest(fs::path(argv[2]) / "compiled-model-package.txt");
      std::string package, extra;
      if (!(manifest >> std::quoted(package)) || manifest >> extra) {
        throw std::invalid_argument("compiled model variant requires one package path");
      }
      const fs::path package_path(package);
      compiled_model = models::opt::load_compiled_model(
          package_path.is_absolute() ? package_path : fs::path(argv[2]) / package_path, argv[1],
          config, weights);
      if ((variant == PartitionedModel::Variant::CompiledDecoder) !=
          !compiled_model.decoders.empty()) {
        throw std::invalid_argument("compiled model scope differs from selected variant");
      }
      if (block == 16) {
        const auto resolved =
            package_path.is_absolute() ? package_path : fs::path(argv[2]) / package_path;
        if (fs::exists(resolved / "compiled-block-prefill.txt")) {
          compiled_prefill_layers =
              models::opt::load_compiled_block_prefill(resolved, argv[1], config, weights);
        } else {
          compiled_prefill_ffn =
              models::opt::load_compiled_prefill(resolved, argv[1], config, weights);
        }
      }
    }
    const auto head_backend = variant == PartitionedModel::Variant::CompiledFullModel
                                  ? "fullgraph-nbg"
                              : !compiled_head.graph.path.empty() ? "nbg"
                              : compiled_ffn.empty()              ? "matmul"
                                                                  : "fcl";
    const std::string projection_backend =
        variant == PartitionedModel::Variant::MergedKvCompiledFfn
            ? (projections.compiled_query    ? (projections.compiled_output ? "nbg" : "nbg-qkv")
               : projections.compiled_output ? "nbg-output"
                                             : "fcl")
        : variant == PartitionedModel::Variant::CompiledDecoder   ? "nbg-decoder"
        : variant == PartitionedModel::Variant::CompiledFullModel ? "nbg-fullgraph"
                                                                  : "matmul";
    const auto preparation =
        std::chrono::duration<double>(Clock::now() - preparation_started).count();
    fs::create_directories(directory);
    platform::write_runtime_profile(directory);
    progress(directory, "initialize");
    np101::SdkTimings timings(directory);
    // A closed SDK Context retains registration slots; each rebuild needs a fresh Context.
    const unsigned cycles = lifecycle_check ? 2 : 1;
    for (unsigned cycle = 0; cycle < cycles; ++cycle) {
      np101::SdkScope initialization_scope("phase", "initialize");
      directory = lifecycle_check ? root / ("cycle." + std::to_string(cycle)) : root;
      fs::create_directories(directory);
      progress(root, lifecycle_check ? "cycle." + std::to_string(cycle) : "initialize");
      const auto start = Clock::now();
      np101::Context context;
      // Preserve the last compilation stage if initialization exceeds the device timeout.
      PartitionedModel model(
          context, weights, config, [&](const std::string &stage) { progress(directory, stage); },
          variant, head_policy, block, compiled_ffn, compiled_model.decoders, compiled_prefill_ffn,
          projections, compiled_head, compiled_projections, compiled_model.full, kv_transfer,
          compiled_prefill_layers);
      const auto initialize = std::chrono::duration<double>(Clock::now() - start).count();
      // Rebuild cycles reuse file validation; report one preparation plus this construction.
      const auto ready = preparation + initialize;
      const auto weight_bytes = model.weight_bytes();
      const auto kv_reset_uploads_per_request = model.cache_reset_uploads_per_request();
      const auto prefill_cache_policy =
          model.shares_prefill_cache() ? "shared-input" : "independent";
      {
        std::ofstream initialized(directory / "initialization.json");
        initialized << std::setprecision(17) << "{\"initialize_seconds\":" << initialize
                    << ",\"preparation_seconds\":" << preparation << ",\"ready_seconds\":" << ready
                    << ",\"weight_payload_bytes\":" << weight_bytes << ",\"variant\":\""
                    << variant_name << "\",\"decoder_group_size\":1"
                    << ",\"prefill_head_policy\":\"" << head_policy_name << "\""
                    << ",\"readback_transport\":\"" << options.readback_transport << "\""
                    << ",\"kv_transfer\":\"" << options.kv_transfer << "\""
                    << ",\"prefill_cache_policy\":\"" << prefill_cache_policy << "\""
                    << ",\"prefill_backend\":\""
                    << (compiled_prefill_layers.empty() ? "handwritten-attention"
                                                        : "compiled-attention")
                    << "\""
                    << ",\"token_readback_transport\":\"mapped\""
                    << ",\"head_backend\":\"" << head_backend << "\""
                    << ",\"attention_projection_backend\":\"" << projection_backend << "\""
                    << ",\"prefill_block\":" << block << ",\"launches\":" << model.launches()
                    << ",\"decode_steps\":" << model.decode_steps()
                    << ",\"prefill_steps\":" << model.prefill_steps()
                    << ",\"head_runs\":" << model.head_runs()
                    << ",\"kv_reset_uploads_per_request\":" << kv_reset_uploads_per_request;
        write_memory(initialized, context.memory());
        initialized << "}\n";
        if (!initialized) {
          throw std::runtime_error("cannot save initialization observation");
        }
      }
      bool tokens_match = true;
      bool cache_suffix_matches = true;
      std::size_t cache_suffix_checked_bytes = 0;
      unsigned capacity_checks = 0, capacity_rejections = 0;
      unsigned closed_checks = 0, closed_rejections = 0;
      np101::TensorTransfers diagnostic_transfers;
      std::vector<BenchmarkRun> benchmark_runs;

      for (unsigned index = 0; index < requests.size(); ++index) {
        for (unsigned repetition = 0; repetition < warmups + repeats; ++repetition) {
          const bool warmup = repetition < warmups;
          const unsigned iteration = warmup ? repetition : repetition - warmups;
          const auto name = (warmup ? "warmup." : "measured.") + std::to_string(index) +
                            (benchmark ? "." + std::to_string(iteration) : "");
          progress(directory, name);

          np101::SdkScope request_scope("request", name);
          const auto before = np101::tensor_transfers();
          const auto launches = model.launches();
          const auto counts_before = execution_counts(model);
          const auto cpu_before = read_cpu_snapshot();
          CpuPhaseBoundaries boundaries;
          const auto observe = [&](auto phase) {
            if (phase == inference::GenerationPhase::Prefill) {
              boundaries.prefill = read_cpu_snapshot();
            } else if (phase == inference::GenerationPhase::Decode) {
              boundaries.decode = read_cpu_snapshot();
            }
          };
          const auto result = options.history_mode == "reuse"
                                  ? model.continue_generation(requests[index], maximum, observe)
                                  : model.generate(requests[index], maximum, observe);
          const auto cpu_after = read_cpu_snapshot();
          const auto completed = model.launches();
          const auto request_counts = execution_counts(model, counts_before);
          save_generation(directory / (name + ".json"), result, completed - launches, before,
                          np101::tensor_transfers(), block, request_counts, cpu_before, cpu_after,
                          boundaries);
          // Diagnostics run after generation, outside its transfer/time contract.
          {
            np101::SdkScope diagnostic_scope("phase", "diagnostic");
            const auto diagnostic_before = np101::tensor_transfers();
            if (readback == "logits") {
              const auto logits = model.read_output("logits", readback_method);
              save_diagnostic(directory / (name + ".logits.bin"), logits,
                              "cannot save final logits diagnostic");
            }
            if (result.consumed == config.decoder.capacity) {
              check_rejected_step<std::out_of_range>(
                  model, requests[index].front(), capacity_checks, capacity_rejections,
                  "capacity rejection performed IO or changed model state");
            }
            if (readback == "kv") {
              for (unsigned layer = cache_first - 1; layer < cache_last; ++layer) {
                for (bool values : {false, true}) {
                  auto bytes = model.read_cache(layer, values, readback_method);
                  const auto head_dimension = config.decoder.kv_spec().head_dim;
                  const auto row_bytes =
                      std::size_t(head_dimension) * np101::dtype_bytes(model.cache_spec().type);
                  const auto head_bytes = row_bytes * config.decoder.capacity;
                  if (bytes.size() != head_bytes * config.decoder.heads) {
                    throw std::runtime_error(
                        "cache readback shape differs from configured capacity");
                  }
                  for (unsigned head = 0; head < config.decoder.heads; ++head) {
                    const auto begin =
                        bytes.begin() + head * head_bytes + result.consumed * row_bytes;
                    const auto end = bytes.begin() + (head + 1) * head_bytes;
                    cache_suffix_matches =
                        std::all_of(begin, end, [](std::uint8_t value) { return value == 0; }) &&
                        cache_suffix_matches;
                    cache_suffix_checked_bytes += end - begin;
                  }
                  auto path = directory / (name + ".layer." + std::to_string(layer) +
                                           (values ? ".values.bin" : ".keys.bin"));
                  save_diagnostic(path, bytes, "cannot save cache diagnostic");
                }
              }
            }
            const auto diagnostic_after = np101::tensor_transfers();
            diagnostic_transfers.reads += diagnostic_after.reads - diagnostic_before.reads;
            diagnostic_transfers.read_bytes +=
                diagnostic_after.read_bytes - diagnostic_before.read_bytes;
          }
          if (benchmark) {
            benchmark_runs.push_back({name, index, iteration,
                                      static_cast<unsigned>(requests[index].size()), warmup,
                                      result.tokens == expected[index]});
          }
          // Keep the first failing request's KV, then release normally. Further
          // requests cannot pass the gate once the independent token oracle fails.
          if (result.tokens != expected[index]) {
            tokens_match = false;
            std::cerr << "Request " << index << " token mismatch; expected ";
            inference::write_array(std::cerr, expected[index]);
            std::cerr << ", got ";
            inference::write_array(std::cerr, result.tokens);
            std::cerr << ". Skipping remaining requests and releasing the model.\n";
            break;
          }
          if (!cache_suffix_matches) {
            std::cerr << "Request " << index << " has nonzero unwritten KV cache slots.\n";
            break;
          }
        }
        if (!tokens_match || !cache_suffix_matches) {
          break;
        }
      }
      const auto launches = model.launches();
      const auto counts = execution_counts(model);
      progress(directory, "release");
      {
        np101::SdkScope release_scope("phase", "release");
        model.close();
        check_rejected_step<std::logic_error>(
            model, requests.front().front(), closed_checks, closed_rejections,
            "closed model rejection performed IO or changed model state");
        if (context.memory().live(true) || context.memory().live(false)) {
          throw std::runtime_error("model close left application tensor storage alive");
        }
        context.close();
      }
      const bool passed = tokens_match && cache_suffix_matches;
      if (benchmark) {
        save_benchmark(directory, warmups, repeats, requests.size(), benchmark_runs, passed);
      }
      rusage usage{};
      if (getrusage(RUSAGE_SELF, &usage)) {
        throw std::runtime_error("cannot read host resource observation");
      }
      std::ofstream out(directory / "run.json");
      out << std::setprecision(17) << "{\"status\":\""
          << (passed ? "executed" : "numerical_mismatch")
          << "\",\"released\":true,\"layers\":" << config.decoder.layers
          << ",\"graph_layout\":\"partitioned\",\"variant\":\"" << variant_name << "\""
          << ",\"decoder_group_size\":1,\"prefill_head_policy\":\"" << head_policy_name << "\""
          << ",\"head_runs\":" << counts.head_runs << ",\"prefill_block\":" << block
          << ",\"capacity\":" << config.decoder.capacity << ",\"launches\":" << launches
          << ",\"history_mode\":\"" << options.history_mode << "\""
          << ",\"kv_transfer\":\"" << options.kv_transfer << "\""
          << ",\"prefill_cache_policy\":\"" << prefill_cache_policy << "\""
          << ",\"prefill_backend\":\""
          << (compiled_prefill_layers.empty() ? "handwritten-attention" : "compiled-attention")
          << "\""
          << ",\"weight_payload_bytes\":" << weight_bytes
          << ",\"kv_reset_uploads_per_request\":" << kv_reset_uploads_per_request
          << ",\"initialize_seconds\":" << initialize << ",\"preparation_seconds\":" << preparation
          << ",\"ready_seconds\":" << ready << ",\"host_peak_rss_kib\":" << usage.ru_maxrss
          << ",\"capacity_checks\":" << capacity_checks
          << ",\"capacity_rejections\":" << capacity_rejections
          << ",\"closed_checks\":" << closed_checks
          << ",\"closed_rejections\":" << closed_rejections
          << ",\"cache_suffix_checked_bytes\":" << cache_suffix_checked_bytes
          << ",\"cache_suffix_matches\":" << (cache_suffix_matches ? "true" : "false")
          << ",\"cache_readback\":\"" << readback << "\""
          << ",\"readback_transport\":\"" << options.readback_transport << "\""
          << ",\"token_readback_transport\":\"mapped\""
          << ",\"head_backend\":\"" << head_backend << "\""
          << ",\"attention_projection_backend\":\"" << projection_backend << "\""
          << ",\"cache_first_layer\":" << cache_first << ",\"cache_last_layer\":" << cache_last
          << ",\"diagnostic_reads\":" << diagnostic_transfers.reads
          << ",\"diagnostic_read_bytes\":" << diagnostic_transfers.read_bytes
          << ",\"decode_steps\":" << counts.decode_steps
          << ",\"prefill_steps\":" << counts.prefill_steps;
      write_memory(out, context.memory());
      out << "}\n";
      out.close();
      if (!out) {
        throw std::runtime_error("cannot save experimental lifecycle");
      }
      progress(directory, "complete", passed ? "executed" : "failed", true);
      if (!passed) {
        timings.flush();
        progress(root, "failed", "failed", true);
        return 1;
      }
    }
    timings.flush();
    if (lifecycle_check) {
      std::ofstream summary(root / "lifecycle.json");
      summary << "{\"cycles\":2,\"reused_context\":false,\"recreated_context\":true,"
                 "\"released\":true}\n";
      if (!summary) {
        throw std::runtime_error("cannot save lifecycle summary");
      }
    }
    progress(root, "complete", "executed", true);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    if (fs::is_directory(directory)) {
      try {
        progress(directory, "failed", "failed");
        if (directory != root) {
          progress(root, "failed", "failed");
        }
      } catch (...) {
        // Preserve the original SDK/IO error.
      }
    }
    return 1;
  }
}
