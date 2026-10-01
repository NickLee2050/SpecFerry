#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

namespace specferry::np101 {
// Optional public API timings. Durations use a steady clock; Unix timestamps correlate
// calls with kernel logs and may jump if the system clock changes. Summary mode
// accumulates in memory and writes only on flush. No tensor reads or SDK dependency.
class SdkTimings {
public:
  explicit SdkTimings(const std::filesystem::path &directory);
  ~SdkTimings() noexcept;
  SdkTimings(const SdkTimings &) = delete;
  SdkTimings &operator=(const SdkTimings &) = delete;
  static bool enabled();
  // Explicitly report output errors; destruction makes a best-effort flush.
  void flush();

private:
  struct Statistics {
    std::uint64_t count = 0, completed = 0, exceptions = 0, nested_count = 0;
    double total_seconds = 0, min_seconds = 0, max_seconds = 0;
  };
  enum class Mode { Off, Calls, Summary };

  friend class SdkScope;
  friend class SdkTimer;
  static thread_local SdkTimings *active_;
  Mode mode_ = Mode::Off;
  std::filesystem::path directory_;
  std::ofstream output_;
  std::array<std::string, 3> scope_;
  std::map<std::array<std::string, 4>, Statistics> statistics_;
  unsigned timer_depth_ = 0;
  bool dirty_ = false;
};

// Labels apply to timers started inside the scope. Empty values retain an outer
// label; nesting restores the previous value, including during exception unwind.
class SdkScope {
public:
  SdkScope(const char *field, const std::string &value);
  ~SdkScope() noexcept;
  SdkScope(const SdkScope &) = delete;
  SdkScope &operator=(const SdkScope &) = delete;

private:
  std::string *target_ = nullptr;
  std::string previous_;
};

// Durations are inclusive API spans. nested_count in the summary identifies calls
// nested inside another timer; totals from enclosing and enclosed APIs overlap.
class SdkTimer {
public:
  explicit SdkTimer(const char *api);
  ~SdkTimer() noexcept;
  SdkTimer(const SdkTimer &) = delete;
  SdkTimer &operator=(const SdkTimer &) = delete;

private:
  SdkTimings *owner_;
  const char *api_;
  SdkTimings::Statistics *statistics_ = nullptr;
  std::chrono::steady_clock::time_point start_;
  int uncaught_exceptions_ = 0;
  bool nested_ = false;
};

template <class Action> auto sdk_call(const char *api, Action action) {
  if (!SdkTimings::enabled()) {
    return action();
  }
  SdkTimer timer(api);
  return action();
}
} // namespace specferry::np101
