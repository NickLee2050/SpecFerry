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

// Validate the large M1 artifact last on SoC, immediately before construction.
// Every checksum still runs; this only avoids intervening prefill cache reads.
bool verify_primary_model_last();

// SoC compiled prefill consumes each layer's KV before the next producer reuses
// fixed output backing. The accepted EP execution policy remains independent.
bool reuse_prefill_cache_outputs();

// Record the linked platform before initialization, outside inference timings.
void write_runtime_profile(const std::filesystem::path &output_directory);
} // namespace specferry::platform
