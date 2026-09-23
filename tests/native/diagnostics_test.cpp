#include "np101/diagnostics.hpp"
#include "unistd.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
using namespace specferry::np101;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::string read(const std::filesystem::path &path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void contracts(const std::filesystem::path &directory) {
  setenv("SPECFERRY_SDK_TIMING", "off", 1);
  unsigned calls = 0;
  {
    SdkTimings timing(directory);
    require(sdk_call("off", [&] { return ++calls; }) == 1, "disabled observer changed result");
    timing.save();
  }
  require(!std::filesystem::exists(directory / "sdk-timing.tsv"), "disabled mode wrote timings");

  setenv("SPECFERRY_SDK_TIMING", "calls", 1);
  {
    SdkTimings timing(directory);
    {
      TimingLabel request(TimingField::Request, "warmup.0");
      sdk_call("fake_status", [] { return -5; });
    }
    {
      TimingLabel request(TimingField::Request, "measured.0");
      TimingLabel phase(TimingField::Phase, "prefill");
      require(sdk_call("fake_status", [] { return 42; }) == 42, "observer changed status");
      sdk_call("fake_void", [&] { ++calls; });
      const unsigned char binary[] = {255, 254, 0};
      require(sdk_call("fake_pointer", [&] { return binary; }) == binary,
              "observer changed pointer");
      bool rejected = false;
      try {
        TimingLabel inner(TimingField::Component, "exception_scope");
        sdk_call("fake_throw", []() -> int { throw std::runtime_error("fixture"); });
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "observer swallowed exception");
      sdk_call("after_exception", [] { return 0; });
    }
    timing.save();
  }
  require(calls == 2 && !SdkTimings::enabled(), "observer changed call count or leaked session");
  const auto summary = read(directory / "sdk-timing.tsv");
  require(summary.find("warmup.0\tinitialize\tmodel\tfake_status\t1\t") != std::string::npos,
          "warmup attribution lost");
  require(summary.find("measured.0\tprefill\tmodel\tafter_exception\t1\t") != std::string::npos,
          "exception did not restore labels");
  require(summary.find("exception_scope\tfake_throw\t1\t") != std::string::npos,
          "throwing call was not counted");
  const auto trace = read(directory / "sdk-calls.tsv");
  require(trace.find("\t-5\n") != std::string::npos, "SDK status missing from trace");
  require(trace.find("\texception\n") != std::string::npos, "exception missing from trace");
  require(trace.find("\t0x") != std::string::npos, "pointer result was not formatted as address");
}
} // namespace

int main() {
  const auto directory =
      std::filesystem::temp_directory_path() / ("specferry-timing-" + std::to_string(getpid()));
  try {
    std::filesystem::create_directory(directory);
    contracts(directory);
    std::filesystem::remove_all(directory);
    std::cout << "SDK timing preserves calls, results, exceptions and request/phase labels.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    std::filesystem::remove_all(directory);
    return 1;
  }
}
