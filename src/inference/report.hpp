#pragma once

#include "inference/generation.hpp"

#include <cstddef>
#include <optional>
#include <ostream>
#include <vector>

namespace specferry::inference {
struct GenerationMetrics {
  std::size_t generated_tokens = 0;
  std::size_t decode_intervals = 0;
  std::optional<double> output_tokens_per_second;
  std::optional<double> decode_tokens_per_second;
};

// Preparation and readback are excluded by Generation's existing timing contract.
GenerationMetrics generation_metrics(const Generation &result);
void write_generation(std::ostream &out, const Generation &result);

template <class T> void write_array(std::ostream &out, const std::vector<T> &values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    out << (i ? "," : "") << values[i];
  }
  out << ']';
}

} // namespace specferry::inference
