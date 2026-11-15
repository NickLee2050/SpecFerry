#include "cli/command.hpp"
#include "supervision/process.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {
double seconds(const char *value) {
  std::size_t consumed = 0;
  const std::string text(value);
  const double result = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(result)) {
    throw std::invalid_argument("invalid supervisor budget");
  }
  return result;
}
} // namespace

int main(int argc, char **argv) {
  // This is an internal adapter protocol; deployment and model commands stay unchanged.
  if (argc < 8) {
    return specferry::cli::command(
        0, 1, "usage: specferry_supervise OUTPUT PREPARATION EXECUTION LIMIT CWD -- COMMAND...",
        [] { return 2; });
  }
  return specferry::cli::command(argc, argc, "", [&] {
    specferry::supervision::ProcessOptions options;
    options.output = argv[1];
    const double preparation = seconds(argv[2]);
    if (preparation != -1) {
      options.preparation_seconds = preparation;
    }
    options.execution_seconds = seconds(argv[3]);
    const double limit = seconds(argv[4]);
    if (limit != 300 && limit != 600) {
      throw std::invalid_argument("unsupported execution limit");
    }
    options.execution_limit = static_cast<unsigned>(limit);
    options.working_directory = argv[5];
    if (std::string(argv[6]) != "--") {
      throw std::invalid_argument("missing supervisor command separator");
    }
    options.command.assign(argv + 7, argv + argc);
    return specferry::supervision::supervise(options);
  });
}
