#pragma once

#include <filesystem>

namespace specferry::platform {
struct RuntimeProfile {
  const char *target;
  const char *architecture;
  const char *sdk_identity;
  const char *sdk_library_directory;
};

// CMake links exactly one platform implementation. Neither call initializes the SDK.
const RuntimeProfile &runtime_profile();
void verify_driver();

// The optional validated cache package selects the larger SoC fast prefix.
// Without it, preserve the accepted bounded-construction baseline; EP stays disabled.
unsigned pointwise_prefill_layers(unsigned verified_cache_layers = 0);

// Only the compiled-attention prefill table uses this platform policy; model/head
// configuration and the 8 MiB tensor/read limit remain independent.
unsigned prefill_embedding_block_rows(unsigned configured_rows);

// SoC deployment verifies model payloads before execution; EP retains startup hashes.
// SDK identity and controlled graph-cache checks are independent of this policy.
bool verify_model_payloads();

// Parse the M1 artifact after prefill preparation on SoC.
bool verify_primary_model_last();

// SoC compiled prefill consumes each layer's KV before the next producer reuses
// fixed output backing. The accepted EP execution policy remains independent.
bool reuse_prefill_cache_outputs();

// Record the linked platform before initialization, outside inference timings.
void write_runtime_profile(const std::filesystem::path &output_directory);

// Notify the owning supervisor once, after initialization and before the first request.
// Without the internal readiness pipe, standalone execution is unchanged.
void notify_model_ready();
} // namespace specferry::platform
