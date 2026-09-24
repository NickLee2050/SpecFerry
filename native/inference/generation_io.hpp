#pragma once

#include "inference/generation.hpp"

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <vector>

namespace specferry::inference {
template <class T> void write_array(std::ostream &out, const std::vector<T> &values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    out << (i ? "," : "") << values[i];
  }
  out << ']';
}

inline void write_generation(std::ostream &out, const Generation &result) {
  out << std::setprecision(17) << "\"tokens\":";
  write_array(out, result.tokens);
  out << ",\"consumed\":" << result.consumed << ",\"stop_reason\":\"" << result.stop_reason
      << "\",\"prefill_seconds\":" << result.prefill_seconds
      << ",\"first_token_seconds\":" << result.first_token_seconds
      << ",\"total_seconds\":" << result.total_seconds << ",\"token_seconds\":";
  write_array(out, result.token_seconds);
}
} // namespace specferry::inference
