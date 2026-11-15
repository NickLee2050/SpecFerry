#include "devices/np101/api/context.hpp"
#include "devices/np101/api/diagnostics.hpp"
#include "devices/np101/api/memory_budget.hpp"
#include "devices/np101/api/runtime_profile.hpp"
#include "devices/np101/api/tensor.hpp"
#include "devices/np101/api/weights.hpp"
#include "inference/report.hpp"
#include "inference/request.hpp"
#include "io/files.hpp"
#include "models/opt/execution.hpp"
#include "models/opt/partitioned_graph_model.hpp"
#include "models/opt/request.hpp"
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

using GraphRequest = models::opt::GraphRequest;

struct PreparedModel {
  models::opt::CompiledArtifacts artifacts;
  models::opt::GraphOptions options;
  std::string head_backend, projection_backend, prefill_backend;
  double seconds;
};

struct ModelObservation {
  std::size_t weight_bytes, kv_reset_uploads_per_request, launches;
  std::string cache_dtype, prefill_cache_policy;
  ExecutionCounts counts;
};

struct RequestResults {
  bool tokens_match = true, cache_suffix_matches = true;
  std::size_t cache_suffix_checked_bytes = 0;
  unsigned capacity_checks = 0, capacity_rejections = 0;
  unsigned closed_checks = 0, closed_rejections = 0;
  np101::TensorTransfers diagnostic_transfers;
  std::vector<BenchmarkRun> benchmark_runs;

  bool passed() const { return tokens_match && cache_suffix_matches; }
};

PreparedModel prepare_model(const inference::ExecutionPaths &paths, const GraphRequest &settings,
                            np101::WeightStore &weights, Clock::time_point preparation_started) {
  const auto &options = settings.options;
  const auto &config = settings.config;
  const auto block = options.block;
  const auto variant = settings.model_options.variant;
  std::vector<std::pair<std::string, double>> preparation_steps;
  const auto prepare = [&](const char *name, const auto &action) {
    const auto start = Clock::now();
    action();
    preparation_steps.emplace_back(name,
                                   std::chrono::duration<double>(Clock::now() - start).count());
  };
  if (platform::verify_model_payloads()) {
    prepare("weights", [&] { weights.verify(); });
  }
  std::vector<models::opt::CompiledFfn> compiled_ffn, compiled_prefill_ffn;
  std::vector<models::opt::CompiledPrefillLayer> compiled_prefill_layers;
  models::opt::CompiledHead compiled_head;
  std::vector<models::opt::CompiledProjections> compiled_projections;
  models::opt::CompiledModelPackage compiled_model;
  models::opt::ProjectionPolicy projections;
  if (variant == PartitionedModel::Variant::MergedKvCompiledFfn) {
    std::ifstream manifest(fs::path(paths.request) / "compiled-ffn-package.txt");
    std::string package, extra;
    if (!(manifest >> std::quoted(package)) || manifest >> extra) {
      throw std::invalid_argument("compiled FFN variant requires one package path");
    }
    compiled_ffn = models::opt::load_compiled_ffn(package, paths.deployment, config, weights,
                                                  compiled_head, &compiled_projections);
    if (block != 1) {
      compiled_prefill_ffn =
          models::opt::load_compiled_prefill(package, paths.deployment, config, weights);
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
    std::ifstream manifest(fs::path(paths.request) / "compiled-model-package.txt");
    std::string package, extra;
    if (!(manifest >> std::quoted(package)) || manifest >> extra) {
      throw std::invalid_argument("compiled model variant requires one package path");
    }
    const fs::path package_path(package);
    const auto resolved =
        package_path.is_absolute() ? package_path : fs::path(paths.request) / package_path;
    const auto load_prefill = [&] {
      if (block == 16) {
        prepare("prefill", [&] {
          compiled_prefill_layers =
              models::opt::load_compiled_block_prefill(resolved, paths.deployment, config, weights);
        });
      }
    };
    if (platform::verify_primary_model_last()) {
      load_prefill();
    }
    prepare("model", [&] {
      compiled_model =
          models::opt::load_compiled_model(resolved, paths.deployment, config, weights);
    });
    if ((variant == PartitionedModel::Variant::CompiledDecoder) !=
        !compiled_model.decoders.empty()) {
      throw std::invalid_argument("compiled model scope differs from selected variant");
    }
    if (!platform::verify_primary_model_last()) {
      load_prefill();
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
  const auto prefill_backend = !compiled_prefill_layers.empty() ? "compiled-attention"
                               : block == 1                     ? "disabled"
                                                                : "handwritten-attention";
  const auto preparation =
      std::chrono::duration<double>(Clock::now() - preparation_started).count();
  fs::create_directories(paths.output);
  {
    std::ofstream prepared(paths.output / "preparation.json");
    prepared << std::setprecision(17) << "{\"total_seconds\":" << preparation << ",\"steps\":[";
    for (std::size_t index = 0; index < preparation_steps.size(); ++index) {
      const auto &[name, duration] = preparation_steps[index];
      prepared << (index ? "," : "") << "{\"name\":\"" << name << "\",\"seconds\":" << duration
               << '}';
    }
    prepared << "]}\n";
    if (!prepared) {
      throw std::runtime_error("cannot save preparation timings");
    }
  }

  auto model_options = settings.model_options;
  model_options.projections = projections;
  models::opt::CompiledArtifacts artifacts;
  artifacts.feed_forward = compiled_ffn;
  artifacts.decoders = compiled_model.decoders;
  artifacts.prefill_feed_forward = compiled_prefill_ffn;
  artifacts.head = compiled_head;
  artifacts.projections = compiled_projections;
  artifacts.full_model = compiled_model.full;
  artifacts.prefill_layers = compiled_prefill_layers;
  return {std::move(artifacts), model_options,   head_backend,
          projection_backend,   prefill_backend, preparation};
}

ModelObservation observe_model(const PartitionedModel &model) {
  const auto weight_bytes = model.weight_bytes();
  const auto cache_dtype = np101::dtype_name(model.cache_spec().type);
  const auto kv_reset_uploads_per_request = model.cache_reset_uploads_per_request();
  const auto prefill_cache_policy = model.shares_prefill_cache() ? "shared-input" : "independent";
  return {weight_bytes, kv_reset_uploads_per_request, model.launches(),
          cache_dtype,  prefill_cache_policy,         execution_counts(model)};
}

void save_initialization(const fs::path &directory, const GraphRequest &settings,
                         const PreparedModel &prepared, const ModelObservation &observation,
                         const np101::MemoryBudget &memory, double initialize) {
  const auto &options = settings.options;
  const auto block = options.block;
  const auto &variant_name = options.variant;
  const auto &head_policy_name = options.prefill_head;
  const auto preparation = prepared.seconds;
  const auto ready = preparation + initialize;
  const auto &head_backend = prepared.head_backend;
  const auto &projection_backend = prepared.projection_backend;
  const auto &prefill_backend = prepared.prefill_backend;
  std::ofstream initialized(directory / "initialization.json");
  initialized << std::setprecision(17) << "{\"initialize_seconds\":" << initialize
              << ",\"preparation_seconds\":" << preparation << ",\"ready_seconds\":" << ready
              << ",\"weight_payload_bytes\":" << observation.weight_bytes << ",\"variant\":\""
              << variant_name << "\",\"decoder_group_size\":1"
              << ",\"prefill_head_policy\":\"" << head_policy_name << "\""
              << ",\"readback_transport\":\"" << options.readback_transport << "\""
              << ",\"kv_transfer\":\"" << options.kv_transfer << "\""
              << ",\"prefill_cache_policy\":\"" << observation.prefill_cache_policy << "\""
              << ",\"cache_dtype\":\"" << observation.cache_dtype << "\""
              << ",\"prefill_backend\":\"" << prefill_backend << "\""
              << ",\"token_readback_transport\":\"mapped\""
              << ",\"head_backend\":\"" << head_backend << "\""
              << ",\"attention_projection_backend\":\"" << projection_backend << "\""
              << ",\"prefill_block\":" << block << ",\"launches\":" << observation.launches
              << ",\"decode_steps\":" << observation.counts.decode_steps
              << ",\"prefill_steps\":" << observation.counts.prefill_steps
              << ",\"head_runs\":" << observation.counts.head_runs
              << ",\"kv_reset_uploads_per_request\":" << observation.kv_reset_uploads_per_request;
  write_memory(initialized, memory);
  initialized << "}\n";
  if (!initialized) {
    throw std::runtime_error("cannot save initialization observation");
  }
}

RequestResults run_requests(const fs::path &directory, const GraphRequest &settings,
                            PartitionedModel &model) {
  const auto &options = settings.options;
  const auto &config = settings.config;
  const auto &requests = settings.requests;
  const auto &expected = settings.expected;
  const auto benchmark = options.benchmark;
  const auto warmups = options.warmups, repeats = options.repeats;
  const auto block = options.block, maximum = options.maximum;
  const auto &readback = options.readback;
  const auto readback_method = settings.readback_method;
  const auto cache_first = options.cache_first, cache_last = options.cache_last;
  RequestResults results;
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
          io::write_bytes(directory / (name + ".logits.bin"), logits,
                          "cannot save final logits diagnostic");
        }
        if (result.consumed == config.decoder.capacity) {
          check_rejected_step<std::out_of_range>(
              model, requests[index].front(), results.capacity_checks, results.capacity_rejections,
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
                throw std::runtime_error("cache readback shape differs from configured capacity");
              }
              for (unsigned head = 0; head < config.decoder.heads; ++head) {
                const auto begin = bytes.begin() + head * head_bytes + result.consumed * row_bytes;
                const auto end = bytes.begin() + (head + 1) * head_bytes;
                results.cache_suffix_matches =
                    std::all_of(begin, end, [](std::uint8_t value) { return value == 0; }) &&
                    results.cache_suffix_matches;
                results.cache_suffix_checked_bytes += end - begin;
              }
              auto path = directory / (name + ".layer." + std::to_string(layer) +
                                       (values ? ".values.bin" : ".keys.bin"));
              io::write_bytes(path, bytes, "cannot save cache diagnostic");
            }
          }
        }
        const auto diagnostic_after = np101::tensor_transfers();
        results.diagnostic_transfers.reads += diagnostic_after.reads - diagnostic_before.reads;
        results.diagnostic_transfers.read_bytes +=
            diagnostic_after.read_bytes - diagnostic_before.read_bytes;
      }
      if (benchmark) {
        results.benchmark_runs.push_back({name, index, iteration,
                                          static_cast<unsigned>(requests[index].size()), warmup,
                                          result.tokens == expected[index]});
      }
      // Keep the first failing request's KV, then release normally. Further
      // requests cannot pass the gate once the independent token oracle fails.
      if (result.tokens != expected[index]) {
        results.tokens_match = false;
        std::cerr << "Request " << index << " token mismatch; expected ";
        inference::write_array(std::cerr, expected[index]);
        std::cerr << ", got ";
        inference::write_array(std::cerr, result.tokens);
        std::cerr << ". Skipping remaining requests and releasing the model.\n";
        break;
      }
      if (!results.cache_suffix_matches) {
        std::cerr << "Request " << index << " has nonzero unwritten KV cache slots.\n";
        break;
      }
    }
    if (!results.passed()) {
      break;
    }
  }
  return results;
}

void release_model(const fs::path &directory, const GraphRequest &settings, PartitionedModel &model,
                   np101::Context &context, RequestResults &result) {
  progress(directory, "release");
  {
    np101::SdkScope release_scope("phase", "release");
    model.close();
    check_rejected_step<std::logic_error>(
        model, settings.requests.front().front(), result.closed_checks, result.closed_rejections,
        "closed model rejection performed IO or changed model state");
    if (context.memory().live(true) || context.memory().live(false)) {
      throw std::runtime_error("model close left application tensor storage alive");
    }
    context.close();
  }
}

void save_run(const fs::path &directory, const GraphRequest &settings,
              const PreparedModel &prepared, const ModelObservation &observation,
              const np101::MemoryBudget &memory, double initialize, const RequestResults &result) {
  const auto &options = settings.options;
  const auto &config = settings.config;
  const auto block = options.block;
  const auto &variant_name = options.variant;
  const auto &head_policy_name = options.prefill_head;
  const auto preparation = prepared.seconds;
  const auto ready = preparation + initialize;
  const auto &head_backend = prepared.head_backend;
  const auto &projection_backend = prepared.projection_backend;
  const auto &prefill_backend = prepared.prefill_backend;
  const auto &requests = settings.requests;
  const auto benchmark = options.benchmark;
  const auto warmups = options.warmups, repeats = options.repeats;
  const auto &readback = options.readback;
  const auto cache_first = options.cache_first, cache_last = options.cache_last;
  const bool passed = result.passed();
  if (benchmark) {
    save_benchmark(directory, warmups, repeats, requests.size(), result.benchmark_runs, passed);
  }
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage)) {
    throw std::runtime_error("cannot read host resource observation");
  }
  std::ofstream out(directory / "run.json");
  out << std::setprecision(17) << "{\"status\":\"" << (passed ? "executed" : "numerical_mismatch")
      << "\",\"released\":true,\"layers\":" << config.decoder.layers
      << ",\"graph_layout\":\"partitioned\",\"variant\":\"" << variant_name << "\""
      << ",\"decoder_group_size\":1,\"prefill_head_policy\":\"" << head_policy_name << "\""
      << ",\"head_runs\":" << observation.counts.head_runs << ",\"prefill_block\":" << block
      << ",\"capacity\":" << config.decoder.capacity << ",\"launches\":" << observation.launches
      << ",\"history_mode\":\"" << options.history_mode << "\""
      << ",\"kv_transfer\":\"" << options.kv_transfer << "\""
      << ",\"prefill_cache_policy\":\"" << observation.prefill_cache_policy << "\""
      << ",\"cache_dtype\":\"" << observation.cache_dtype << "\""
      << ",\"prefill_backend\":\"" << prefill_backend << "\""
      << ",\"weight_payload_bytes\":" << observation.weight_bytes
      << ",\"kv_reset_uploads_per_request\":" << observation.kv_reset_uploads_per_request
      << ",\"initialize_seconds\":" << initialize << ",\"preparation_seconds\":" << preparation
      << ",\"ready_seconds\":" << ready << ",\"host_peak_rss_kib\":" << usage.ru_maxrss
      << ",\"capacity_checks\":" << result.capacity_checks
      << ",\"capacity_rejections\":" << result.capacity_rejections
      << ",\"closed_checks\":" << result.closed_checks
      << ",\"closed_rejections\":" << result.closed_rejections
      << ",\"cache_suffix_checked_bytes\":" << result.cache_suffix_checked_bytes
      << ",\"cache_suffix_matches\":" << (result.cache_suffix_matches ? "true" : "false")
      << ",\"cache_readback\":\"" << readback << "\""
      << ",\"readback_transport\":\"" << options.readback_transport << "\""
      << ",\"token_readback_transport\":\"mapped\""
      << ",\"head_backend\":\"" << head_backend << "\""
      << ",\"attention_projection_backend\":\"" << projection_backend << "\""
      << ",\"cache_first_layer\":" << cache_first << ",\"cache_last_layer\":" << cache_last
      << ",\"diagnostic_reads\":" << result.diagnostic_transfers.reads
      << ",\"diagnostic_read_bytes\":" << result.diagnostic_transfers.read_bytes
      << ",\"decode_steps\":" << observation.counts.decode_steps
      << ",\"prefill_steps\":" << observation.counts.prefill_steps;
  write_memory(out, memory);
  out << "}\n";
  out.close();
  if (!out) {
    throw std::runtime_error("cannot save model execution report");
  }
}

} // namespace

namespace specferry::models::opt {
int execute_partitioned(const inference::ExecutionPaths &paths) {
  const auto &directory = paths.output;
  try {
    const auto settings = read_graph_request(paths.request);
    const auto preparation_started = Clock::now();
    np101::WeightStore weights(paths.deployment);
    auto prepared = prepare_model(paths, settings, weights, preparation_started);
    platform::write_runtime_profile(directory);
    progress(directory, "initialize");
    np101::SdkTimings timings(directory);
    {
      np101::SdkScope initialization_scope("phase", "initialize");
      const auto start = Clock::now();
      // Consumers are destroyed before their Context and source weights, including on failure.
      np101::Context context;
      PartitionedModel model(context, weights, settings.config, prepared.options,
                             std::move(prepared.artifacts),
                             [&](const std::string &stage) { progress(directory, stage); });
      const auto initialize = std::chrono::duration<double>(Clock::now() - start).count();
      save_initialization(directory, settings, prepared, observe_model(model), context.memory(),
                          initialize);
      platform::notify_model_ready();
      auto result = run_requests(directory, settings, model);
      const auto observation = observe_model(model);
      release_model(directory, settings, model, context, result);
      save_run(directory, settings, prepared, observation, context.memory(), initialize, result);
      progress(directory, "complete", result.passed() ? "executed" : "failed", true);
      if (!result.passed()) {
        timings.flush();
        progress(directory, "failed", "failed", true);
        return 1;
      }
    }
    timings.flush();
    progress(directory, "complete", "executed", true);
    return 0;
  } catch (...) {
    if (fs::is_directory(directory)) {
      try {
        progress(directory, "failed", "failed");
      } catch (...) {
        // Preserve the original SDK/IO error.
      }
    }
    throw;
  }
}
} // namespace specferry::models::opt
