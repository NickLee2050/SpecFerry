#include "inference/report.hpp"

#include <cmath>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace specferry::inference {
namespace {
void write_rate(std::ostream &out, const std::optional<double> &rate) {
  if (rate) {
    out << *rate;
  } else {
    out << "null";
  }
}
} // namespace

GenerationMetrics generation_metrics(const Generation &result) {
  const auto expected = result.tokens.empty() ? 0 : result.tokens.size() - 1;
  const auto valid_time = [](double value) { return std::isfinite(value) && value >= 0; };
  const double decode_seconds =
      std::accumulate(result.token_seconds.begin(), result.token_seconds.end(), 0.0);
  if (result.token_seconds.size() != expected || !valid_time(result.first_token_seconds) ||
      !valid_time(result.total_seconds) || !valid_time(result.prefill_seconds) ||
      result.prefill_seconds > result.first_token_seconds ||
      result.first_token_seconds > result.total_seconds ||
      result.first_token_seconds + decode_seconds > result.total_seconds + 1e-6) {
    throw std::invalid_argument("inconsistent generation timings");
  }
  for (const auto seconds : result.token_seconds) {
    if (!std::isfinite(seconds) || seconds <= 0) {
      throw std::invalid_argument("inconsistent generation timings");
    }
  }
  GenerationMetrics metrics;
  metrics.generated_tokens = result.tokens.size();
  metrics.decode_intervals = expected;
  if (result.total_seconds > 0) {
    metrics.output_tokens_per_second = metrics.generated_tokens / result.total_seconds;
  }
  if (expected) {
    metrics.decode_tokens_per_second = expected / decode_seconds;
  }
  return metrics;
}

void write_generation(std::ostream &out, const Generation &result) {
  const auto metrics = generation_metrics(result);
  out << std::setprecision(17) << "\"tokens\":";
  write_array(out, result.tokens);
  out << ",\"consumed\":" << result.consumed << ",\"stop_reason\":\"" << result.stop_reason
      << "\",\"state_preparation_seconds\":" << result.state_preparation_seconds
      << ",\"replayed_prompt_tokens\":" << result.replayed_prompt_tokens
      << ",\"prefill_seconds\":" << result.prefill_seconds
      << ",\"first_token_seconds\":" << result.first_token_seconds
      << ",\"total_seconds\":" << result.total_seconds << ",\"token_seconds\":";
  write_array(out, result.token_seconds);
  out << ",\"generated_tokens\":" << metrics.generated_tokens
      << ",\"decode_intervals\":" << metrics.decode_intervals << ",\"output_tokens_per_second\":";
  write_rate(out, metrics.output_tokens_per_second);
  out << ",\"decode_tokens_per_second\":";
  write_rate(out, metrics.decode_tokens_per_second);
}

} // namespace specferry::inference
