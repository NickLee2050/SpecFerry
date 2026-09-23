#include "np101/diagnostics.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>

namespace specferry::np101 {
thread_local SdkTimings *SdkTimings::active_ = nullptr;

SdkTimings::SdkTimings(const std::filesystem::path &directory)
    : directory_(directory), origin_(std::chrono::steady_clock::now()) {
  const auto *setting = std::getenv("SPECFERRY_SDK_TIMING");
  const std::string mode = setting ? setting : "off";
  if (mode != "off" && mode != "summary" && mode != "calls") {
    throw std::invalid_argument("SPECFERRY_SDK_TIMING must be off, summary or calls");
  }
  if (active_) {
    throw std::logic_error("SDK timing sessions cannot overlap on one thread");
  }
  enabled_ = mode != "off";
  if (mode == "calls") {
    trace_.open(directory / "sdk-calls.tsv");
    trace_ << "event\tid\telapsed_seconds\trequest\tphase\tcomponent\tapi_or_result\n";
    if (!trace_) {
      throw std::runtime_error("cannot open SDK call trace");
    }
  }
  if (enabled_) {
    active_ = this;
  }
}

SdkTimings::~SdkTimings() {
  if (active_ == this) {
    active_ = nullptr;
  }
}

bool SdkTimings::enabled() { return active_ != nullptr; }

void SdkTimings::save() const {
  if (!enabled_) {
    return;
  }
  if (trace_.is_open() && !trace_) {
    throw std::runtime_error("SDK call trace could not be written completely");
  }
  // Cumulative snapshots are written only outside timed requests.
  const auto temporary = directory_ / "sdk-timing.tsv.tmp";
  std::ofstream out(temporary);
  out << std::setprecision(17)
      << "request\tphase\tcomponent\tapi\tcalls\ttotal_seconds\tmax_seconds\texceptions\n";
  for (const auto &[key, value] : measurements_) {
    for (const auto &label : key) {
      out << label << '\t';
    }
    out << value.calls << '\t' << value.seconds << '\t' << value.maximum << '\t' << value.exceptions
        << '\n';
  }
  out.close();
  if (!out) {
    throw std::runtime_error("cannot save SDK timing summary");
  }
  std::filesystem::rename(temporary, directory_ / "sdk-timing.tsv");
}

TimingLabel::TimingLabel(TimingField field, const std::string &value)
    : owner_(SdkTimings::active_), field_(static_cast<std::size_t>(field)) {
  if (owner_) {
    previous_ = owner_->labels_[field_];
    owner_->labels_[field_] = value;
  }
}

TimingLabel::~TimingLabel() {
  if (owner_) {
    owner_->labels_[field_].swap(previous_);
  }
}

SdkTimer::SdkTimer(const char *api) : owner_(SdkTimings::active_) {
  auto &labels = owner_->labels_;
  measurement_ = &owner_->measurements_[{labels[0], labels[1], labels[2], api}];
  id_ = ++owner_->next_id_;
  if (owner_->trace_.is_open()) {
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - owner_->origin_).count();
    owner_->trace_ << std::setprecision(17) << "begin\t" << id_ << '\t' << elapsed << '\t'
                   << labels[0] << '\t' << labels[1] << '\t' << labels[2] << '\t' << api
                   << std::endl;
  }
  start_ = std::chrono::steady_clock::now();
}

bool SdkTimer::trace_results() const { return owner_->trace_.is_open(); }

void SdkTimer::finish(const std::string &result, bool exception) {
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  ++measurement_->calls;
  measurement_->seconds += seconds;
  measurement_->maximum = std::max(measurement_->maximum, seconds);
  measurement_->exceptions += exception;
  if (owner_->trace_.is_open()) {
    owner_->trace_ << "end\t" << id_ << '\t' << seconds << "\t\t\t\t" << result << std::endl;
  }
}
} // namespace specferry::np101
