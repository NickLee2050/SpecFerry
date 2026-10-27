#include "np101/ops/graph_cache.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
using specferry::np101::ops::validate_graph_cache;

template <class Action> void rejects(Action action) {
  try {
    action();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("unsafe graph cache layout was accepted");
}

int main() {
  const auto root = fs::temp_directory_path() /
                    ("specferry-graph-cache-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directory(root);
  try {
    const auto file = root / "graph.nb";
    std::ofstream(file) << "cache";
    const specferry::np101::ops::VerifiedGraphCache cache{file, std::string(64, 'a'), 5};
    validate_graph_cache(cache);
    auto truncated = cache;
    truncated.bytes = 6;
    rejects([&] { validate_graph_cache(truncated); });
    std::ofstream(root / "unexpected.nb") << "other graph";
    rejects([&] { validate_graph_cache(cache); });
    fs::remove(root / "unexpected.nb");
    fs::rename(file, root / "original");
    fs::create_symlink(root / "original", file);
    rejects([&] { validate_graph_cache(cache); });
    fs::remove_all(root);
    std::cout << "Verified cache layout contract passed without creating an SDK Context\n";
    return 0;
  } catch (const std::exception &error) {
    fs::remove_all(root);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
