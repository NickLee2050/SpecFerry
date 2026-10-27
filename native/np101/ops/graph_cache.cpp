#include "np101/ops/graph_cache.hpp"

#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace specferry::np101::ops {
namespace {
class CacheEnvironment {
  const std::array<const char *, 4> names_{
      "VIV_VX_ENABLE_CACHE_GRAPH_BINARY", "VIV_VX_CACHE_BINARY_GRAPH_DIR",
      "VIV_VX_CACHE_BINARY_GRAPH_MAX_SIZEKB", "VIV_VX_CACHE_BINARY_GRAPH_MAX_COUNT"};
  std::array<std::optional<std::string>, 4> previous_;

public:
  explicit CacheEnvironment(const std::filesystem::path &directory, bool enabled = true) {
    for (std::size_t index = 0; index < names_.size(); ++index) {
      if (const auto *value = std::getenv(names_[index])) {
        previous_[index] = value;
      }
    }
    const std::array<std::string, 4> values{enabled ? "1" : "0", directory.string(), "65536", "1"};
    for (std::size_t index = 0; index < names_.size(); ++index) {
      if (setenv(names_[index], values[index].c_str(), 1) != 0) {
        restore();
        throw std::runtime_error("cannot configure verified SDK graph cache");
      }
    }
  }

  ~CacheEnvironment() { restore(); }

private:
  void restore() noexcept {
    for (std::size_t index = 0; index < names_.size(); ++index) {
      if (previous_[index]) {
        setenv(names_[index], previous_[index]->c_str(), 1);
      } else {
        unsetenv(names_[index]);
      }
    }
  }
};
} // namespace

void validate_graph_cache(const VerifiedGraphCache &cache) {
  if (cache.path.empty() || std::filesystem::is_symlink(cache.path) ||
      !std::filesystem::is_regular_file(cache.path) || !cache.bytes ||
      cache.bytes > 64 * 1024 * 1024 || std::filesystem::file_size(cache.path) != cache.bytes ||
      cache.sha256.size() != 64) {
    throw std::invalid_argument("verified graph cache file/size differs");
  }
  unsigned count = 0;
  for (const auto &entry : std::filesystem::directory_iterator(cache.path.parent_path())) {
    if (!entry.is_regular_file() || entry.is_symlink() || entry.path() != cache.path) {
      throw std::invalid_argument("verified graph cache directory contains unexpected entries");
    }
    ++count;
  }
  if (count != 1) {
    throw std::invalid_argument("verified graph cache directory is incomplete");
  }
}

GraphCacheSession::GraphCacheSession(Context &context) : boundary_(context, 4, 1) {
  // A verified graph returns before refreshing the SDK's Context flag. Prepare a
  // tiny ordinary graph now and defer its first Verify until cached imports finish.
  const auto input = boundary_.tensor({DataType::Float16, {1}});
  const auto output = boundary_.convert(input, DataType::Float32);
  boundary_.setup({input}, {output});
}

GraphCacheSession::~GraphCacheSession() {
  try {
    close();
  } catch (...) {
    // Explicit close reports failures; do not replace an initialization exception.
  }
}

void GraphCacheSession::verify(GraphBuilder &builder, const VerifiedGraphCache &cache) {
  if (closed_) {
    throw std::logic_error("verified graph cache session is closed");
  }
  validate_graph_cache(cache);
  {
    CacheEnvironment environment(cache.path.parent_path());
    builder.verify();
  }
  vx_uint32 nodes = 0;
  check(vxQueryGraph(builder.graph.get()->g, VX_GRAPH_NUMNODES, &nodes, sizeof(nodes)),
        "query cached graph node count");
  // The pinned SDK replaces all original nodes with one imported binary node on a hit.
  // The package has already checked the file hash; recheck after SDK access to reject
  // an implicit regeneration as well as directory changes. Access-time changes are allowed.
  validate_graph_cache(cache);
  if (nodes != 1 || file_sha256(cache.path) != cache.sha256) {
    throw std::runtime_error("verified SDK graph cache missed or changed; refusing cold fallback");
  }
}

void GraphCacheSession::close() {
  if (!closed_) {
    CacheEnvironment environment({}, false);
    boundary_.verify();
    boundary_.graph.close();
    closed_ = true;
  }
}
} // namespace specferry::np101::ops
