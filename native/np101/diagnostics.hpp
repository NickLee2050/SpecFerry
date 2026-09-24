#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>

namespace specferry::np101 {
// Optional public API wall times. No tensor reads, summary parser or device profiler.
class SdkTimings {
public:
  explicit SdkTimings(const std::filesystem::path &directory);
  ~SdkTimings();
  SdkTimings(const SdkTimings &) = delete;
  SdkTimings &operator=(const SdkTimings &) = delete;
  static bool enabled();

private:
  friend class SdkTimer;
  static thread_local SdkTimings *active_;
  std::ofstream output_;
};

class SdkTimer {
public:
  explicit SdkTimer(const char *api);
  ~SdkTimer();

private:
  SdkTimings *owner_;
  const char *api_;
  std::chrono::steady_clock::time_point start_;
};

template <class Action> auto sdk_call(const char *api, Action action) {
  if (!SdkTimings::enabled()) {
    return action();
  }
  SdkTimer timer(api);
  return action();
}
} // namespace specferry::np101
