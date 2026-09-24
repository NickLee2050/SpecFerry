#include "np101/diagnostics.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace specferry::np101 {
thread_local SdkTimings *SdkTimings::active_ = nullptr;

SdkTimings::SdkTimings(const std::filesystem::path &directory) {
  const auto *value = std::getenv("SPECFERRY_SDK_TIMING");
  if (!value || std::string(value) == "off") {
    return;
  }
  if (std::string(value) != "calls" || active_) {
    throw std::invalid_argument("SDK timing requires one calls session per thread");
  }
  output_.open(directory / "sdk-calls.tsv");
  if (!output_) {
    throw std::runtime_error("cannot open SDK timing output");
  }
  output_ << "event\tapi\tseconds\n";
  active_ = this;
}

SdkTimings::~SdkTimings() {
  if (active_ == this) {
    active_ = nullptr;
  }
}

bool SdkTimings::enabled() { return active_ != nullptr; }

SdkTimer::SdkTimer(const char *api) : owner_(SdkTimings::active_), api_(api) {
  owner_->output_ << "begin\t" << api_ << "\t\n" << std::flush;
  start_ = std::chrono::steady_clock::now();
}

SdkTimer::~SdkTimer() {
  // End records mean the call returned or unwound; errors remain in stderr.
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  owner_->output_ << "end\t" << api_ << '\t' << seconds << '\n' << std::flush;
}
} // namespace specferry::np101
