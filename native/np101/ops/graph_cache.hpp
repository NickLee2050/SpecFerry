#pragma once

#include "np101/ops/graph_builder.hpp"

#include <cstddef>
#include <filesystem>
#include <string>

namespace specferry::np101::ops {
// An existing SDK-generated cache, validated by the model package before Context creation.
// Its opaque allocations are SDK copies/workspaces, not new application const tensors.
struct VerifiedGraphCache {
  std::filesystem::path path;
  std::string sha256;
  std::size_t bytes = 0;
};

void validate_graph_cache(const VerifiedGraphCache &cache);

// Prepare before cached Verify. The SDK retains a Context flag beyond the environment
// scope; a prepared ordinary graph verified with caching disabled refreshes that flag
// before execution/release of the mixed graph set. Never write private Context fields.
class GraphCacheSession {
  GraphBuilder boundary_;
  bool closed_ = false;

public:
  explicit GraphCacheSession(Context &context);
  ~GraphCacheSession();
  GraphCacheSession(const GraphCacheSession &) = delete;
  GraphCacheSession &operator=(const GraphCacheSession &) = delete;

  // Construct/bind and compile all noncached graphs first. A miss is an error,
  // never a cold fallback. SDK initialization remains serial with fixed bindings.
  void verify(GraphBuilder &builder, const VerifiedGraphCache &cache);
  void close();
};
} // namespace specferry::np101::ops
