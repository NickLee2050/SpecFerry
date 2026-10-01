#include "np101/diagnostics.hpp"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <stdexcept>
#include <string_view>

namespace specferry::np101 {
namespace {
double unix_seconds() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void validate_label(std::string_view value) {
  if (value.find_first_of("\t\r\n") != std::string_view::npos) {
    throw std::invalid_argument("SDK timing labels cannot contain tabs or newlines");
  }
}
} // namespace

thread_local SdkTimings *SdkTimings::active_ = nullptr;

SdkTimings::SdkTimings(const std::filesystem::path &directory) : directory_(directory) {
  const auto *value = std::getenv("SPECFERRY_SDK_TIMING");
  if (!value || std::string(value) == "off") {
    return;
  }
  if (active_) {
    throw std::invalid_argument("SDK timing requires one session per thread");
  }
  if (std::string_view(value) == "calls") {
    mode_ = Mode::Calls;
    output_.open(directory / "sdk-calls.tsv");
    if (!output_) {
      throw std::runtime_error("cannot open SDK timing output");
    }
    output_ << "event\tapi\tseconds\tunix_seconds\toutcome\n";
  } else if (std::string_view(value) == "summary") {
    mode_ = Mode::Summary;
    dirty_ = true;
  } else {
    throw std::invalid_argument("SDK timing must be off, calls or summary");
  }
  active_ = this;
}

SdkTimings::~SdkTimings() noexcept {
  try {
    flush();
  } catch (...) {
    // Callers requiring output must explicitly flush while errors can be handled.
  }
  if (active_ == this) {
    active_ = nullptr;
  }
}

bool SdkTimings::enabled() { return active_ != nullptr; }

void SdkTimings::flush() {
  if (mode_ == Mode::Off) {
    return;
  }
  if (timer_depth_) {
    throw std::logic_error("cannot flush SDK timing while a call is active");
  }
  if (mode_ == Mode::Calls) {
    output_.flush();
    if (!output_) {
      throw std::runtime_error("cannot write SDK call timings");
    }
    return;
  }
  if (!dirty_) {
    return;
  }
  std::ofstream summary(directory_ / "sdk-summary.tsv");
  summary << "request\tphase\trole\tapi\tcount\ttotal_seconds\tmin_seconds\tmax_seconds"
             "\tcompleted\texceptions\tnested_count\n"
          << std::setprecision(17);
  for (const auto &[key, statistics] : statistics_) {
    summary << key[0] << '\t' << key[1] << '\t' << key[2] << '\t' << key[3] << '\t'
            << statistics.count << '\t' << statistics.total_seconds << '\t'
            << statistics.min_seconds << '\t' << statistics.max_seconds << '\t'
            << statistics.completed << '\t' << statistics.exceptions << '\t'
            << statistics.nested_count << '\n';
  }
  summary.close();
  if (!summary) {
    throw std::runtime_error("cannot write SDK timing summary");
  }
  dirty_ = false;
}

SdkScope::SdkScope(const char *field, const std::string &value) {
  const std::string_view name = field ? field : "";
  unsigned index;
  if (name == "request") {
    index = 0;
  } else if (name == "phase") {
    index = 1;
  } else if (name == "role") {
    index = 2;
  } else {
    throw std::invalid_argument("SDK timing scope must be request, phase or role");
  }
  if (!SdkTimings::active_ || value.empty()) {
    return;
  }
  validate_label(value);
  // Copy before changing the active label so allocation failure preserves scope.
  auto replacement = value;
  target_ = &SdkTimings::active_->scope_[index];
  previous_.swap(*target_);
  target_->swap(replacement);
}

SdkScope::~SdkScope() noexcept {
  if (target_) {
    target_->swap(previous_);
  }
}

SdkTimer::SdkTimer(const char *api) : owner_(SdkTimings::active_), api_(api) {
  if (!owner_) {
    return;
  }
  if (!api_) {
    throw std::invalid_argument("SDK timing API label cannot be null");
  }
  validate_label(api_);
  if (owner_->mode_ == SdkTimings::Mode::Summary) {
    const auto &scope = owner_->scope_;
    statistics_ = &owner_->statistics_[{scope[0], scope[1], scope[2], api_}];
  } else {
    const auto timestamp = unix_seconds();
    owner_->output_ << "begin\t" << api_ << "\t\t" << std::fixed << std::setprecision(6)
                    << timestamp << std::defaultfloat << "\t\n"
                    << std::flush;
  }
  uncaught_exceptions_ = std::uncaught_exceptions();
  nested_ = owner_->timer_depth_++ != 0;
  start_ = std::chrono::steady_clock::now();
}

SdkTimer::~SdkTimer() noexcept {
  if (!owner_) {
    return;
  }
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  --owner_->timer_depth_;
  const bool exception = std::uncaught_exceptions() > uncaught_exceptions_;
  if (statistics_) {
    auto &statistics = *statistics_;
    if (!statistics.count || seconds < statistics.min_seconds) {
      statistics.min_seconds = seconds;
    }
    if (seconds > statistics.max_seconds) {
      statistics.max_seconds = seconds;
    }
    ++statistics.count;
    statistics.total_seconds += seconds;
    statistics.completed += !exception;
    statistics.exceptions += exception;
    statistics.nested_count += nested_;
    owner_->dirty_ = true;
    return;
  }
  try {
    const auto timestamp = unix_seconds();
    owner_->output_ << "end\t" << api_ << '\t' << seconds << '\t' << std::fixed
                    << std::setprecision(6) << timestamp << std::defaultfloat << '\t'
                    << (exception ? "exception" : "returned") << '\n'
                    << std::flush;
  } catch (...) {
    // An output failure must not replace an exception from the timed call.
    owner_->output_.setstate(std::ios::badbit);
  }
}
} // namespace specferry::np101
