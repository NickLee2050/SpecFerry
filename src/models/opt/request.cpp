#include "models/opt/request.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace specferry::models::opt {
namespace fs = std::filesystem;

namespace {
using PartitionedModel = PartitionedGraphModel;

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
} // namespace

GraphRequest read_graph_request(const fs::path &request) {
  if (std::getenv("SPECFERRY_VERIFY_WEIGHT_PRELOAD")) {
    throw std::invalid_argument("the weight preload diagnostic is retired");
  }
  const auto options = inference::read_run_options(request, true);
  const auto lifecycle_check = options.lifecycle;
  const auto *guard = std::getenv("SPECFERRY_VERIFY_REQUIRE_OPERATIONS");
  if (lifecycle_check && (!guard || std::string(guard) != "1")) {
    throw std::invalid_argument("lifecycle diagnostics require the pinned compilation guard");
  }
  const auto block = options.block, maximum = options.maximum;
  const auto readback_method = options.readback_transport == "mapped"
                                   ? np101::TensorReadback::MappedHost
                                   : np101::TensorReadback::SdkCopy;
  const auto &variant_name = options.variant;
  const auto variant = parse_variant(variant_name);
  const auto &head_policy_name = options.prefill_head;
  const auto head_policy = head_policy_name == "last-token"
                               ? PartitionedModel::PrefillHeadPolicy::LastToken
                               : PartitionedModel::PrefillHeadPolicy::EveryToken;
  const auto config = models::opt::read_model_config(request);
  const auto kv_transfer = options.kv_transfer == "mapped-slot"
                               ? PartitionedModel::KvTransfer::MappedSlot
                               : PartitionedModel::KvTransfer::DeviceCopy;
  if (kv_transfer == PartitionedModel::KvTransfer::MappedSlot && config.decoder.capacity != 64 &&
      config.decoder.capacity != 128 && config.decoder.capacity != 256 &&
      config.decoder.capacity != 512) {
    throw std::invalid_argument("mapped-slot KV requires the C64/C128/C256/C512 layout");
  }
  const auto cache_last = options.cache_last;
  if (cache_last > config.decoder.layers) {
    throw std::invalid_argument("cache layer range must be within the configured model");
  }
  const auto requests = inference::read_requests(fs::path(request) / "requests.txt",
                                                 config.vocabulary, config.decoder.capacity);
  const auto expected = inference::read_requests(fs::path(request) / "expected.txt",
                                                 config.vocabulary, config.decoder.capacity);
  if (expected.size() != requests.size()) {
    throw std::invalid_argument("expected token rows must match request rows");
  }
  for (const auto &tokens : expected) {
    if (tokens.size() > maximum) {
      throw std::invalid_argument("expected tokens exceed generation limit");
    }
  }
  GraphOptions model_options;
  model_options.variant = variant;
  model_options.prefill_head = head_policy;
  model_options.prefill_block = block;
  model_options.kv_transfer = kv_transfer;
  return {options, config, model_options, readback_method, requests, expected};
}
} // namespace specferry::models::opt
