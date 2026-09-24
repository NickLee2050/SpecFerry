#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>

namespace specferry::np101 {
// Linux process RSS; call outside measured request intervals. No device IO.
void sample_host_memory(const std::filesystem::path &directory, const std::string &phase);

enum class TimingField { Request, Phase, Component };

// One optional observer on the submitting thread. It measures public API wall
// time, not device kernel time. No SDK calls or tensor readbacks are added.
class SdkTimings {
public:
  explicit SdkTimings(const std::filesystem::path &directory);
  ~SdkTimings();
  SdkTimings(const SdkTimings &) = delete;
  SdkTimings &operator=(const SdkTimings &) = delete;

  void save() const;
  static bool enabled();

private:
  friend class TimingLabel;
  friend class SdkTimer;

  struct Measurement {
    std::size_t calls = 0, exceptions = 0;
    double seconds = 0, maximum = 0;
  };

  using Key = std::array<std::string, 4>;
  static thread_local SdkTimings *active_;
  std::array<std::string, 3> labels_{"setup", "initialize", "model"};
  std::map<Key, Measurement> measurements_;
  std::filesystem::path directory_;
  std::ofstream trace_;
  std::chrono::steady_clock::time_point origin_;
  std::size_t next_id_ = 0;
  bool enabled_ = false;
};

// Labels restore their previous value on exit, including exception paths.
class TimingLabel {
public:
  TimingLabel(TimingField field, const std::string &value);
  ~TimingLabel();
  TimingLabel(const TimingLabel &) = delete;
  TimingLabel &operator=(const TimingLabel &) = delete;

private:
  SdkTimings *owner_;
  std::size_t field_;
  std::string previous_;
};

class SdkTimer {
public:
  explicit SdkTimer(const char *api);
  bool trace_results() const;
  void finish(const std::string &result = {}, bool exception = false);

private:
  SdkTimings *owner_;
  SdkTimings::Measurement *measurement_;
  std::chrono::steady_clock::time_point start_;
  std::size_t id_;
};

template <class Action> auto sdk_call(const char *api, Action action) {
  if (!SdkTimings::enabled()) {
    return action();
  }
  SdkTimer timer(api);
  try {
    if constexpr (std::is_void_v<decltype(action())>) {
      action();
      timer.finish("void");
    } else {
      auto result = action();
      std::string description;
      if (timer.trace_results()) {
        std::ostringstream text;
        if constexpr (std::is_pointer_v<decltype(result)>) {
          text << static_cast<const void *>(result);
        } else {
          text << result;
        }
        description = text.str();
      }
      timer.finish(description);
      return result;
    }
  } catch (...) {
    timer.finish("exception", true);
    throw;
  }
}
} // namespace specferry::np101
