#include "np101/memory_budget.hpp"

#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace specferry::np101;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_rejected(MemoryBudget &budget, bool constant, std::size_t bytes) {
  try {
    budget.reserve(constant, bytes);
  } catch (const std::length_error &) {
    return;
  }
  throw std::runtime_error("oversized allocation was accepted");
}

void independent_segments_and_boundaries() {
  MemoryBudget budget;
  auto weights = budget.reserve(true, segment_payload_limit);
  auto state = budget.reserve(false, segment_payload_limit);
  expect_rejected(budget, true, 1);
  expect_rejected(budget, false, 1);
  expect_rejected(budget, false, std::numeric_limits<std::size_t>::max());
  require(budget.live(true) == segment_payload_limit && budget.live(false) == segment_payload_limit,
          "segments overlap or rejection changes accounting");
  weights.reset();
  state.reset();
  require(!budget.live(true) && !budget.live(false), "release did not return the budget");
  require(budget.peak(true) == segment_payload_limit && budget.peak(false) == segment_payload_limit,
          "release lost the high-water marks");
}

void aliases_and_temporary_backing() {
  MemoryBudget budget(64);
  auto owner = budget.reserve(false, 40);
  auto reader = owner;
  owner.reset();
  require(budget.live(false) == 40, "an alias lost the retained allocation");
  expect_rejected(budget, false, 25);
  auto wrapper = budget.reserve(false, 24);
  require(budget.peak(false) == 64, "temporary backing was not charged");
  wrapper = reader;
  require(budget.live(false) == 40, "alias replacement did not release temporary backing");
  reader.reset();
  require(budget.live(false) == 40, "shared storage was released before its final reader");
  wrapper.reset();
  require(budget.live(false) == 0, "last alias did not release storage");
}
} // namespace

int main() {
  try {
    independent_segments_and_boundaries();
    aliases_and_temporary_backing();
    std::cout << "Segment limits, shared storage and temporary allocation accounting passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
