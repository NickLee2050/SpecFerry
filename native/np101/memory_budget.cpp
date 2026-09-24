#include "np101/memory_budget.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace specferry::np101 {
struct MemoryBudget::State {
  std::size_t limit;
  std::array<std::size_t, 2> live{}, peak{};
};

MemoryBudget::MemoryBudget(std::size_t limit) : state_(std::make_shared<State>()) {
  if (!limit || limit > segment_payload_limit) {
    throw std::invalid_argument("segment payload limit must be in 1..1 GiB");
  }
  state_->limit = limit;
}

MemoryBudget::Allocation::Allocation(std::shared_ptr<State> state, bool constant, std::size_t bytes)
    : state_(std::move(state)), constant_(constant), bytes_(bytes) {}

MemoryBudget::Allocation::~Allocation() { state_->live[constant_] -= bytes_; }

MemoryBudget::Lease MemoryBudget::reserve(bool constant, std::size_t bytes) {
  if (bytes > state_->limit - state_->live[constant]) {
    throw std::length_error(
        std::string("application payload budget exceeded: ") + (constant ? "const" : "non-const") +
        " live=" + std::to_string(state_->live[constant]) + " requested=" + std::to_string(bytes) +
        " limit=" + std::to_string(state_->limit) + "; SDK allocation was not attempted");
  }
  // Construct the control block before charging: allocation failure must not
  // consume the budget, including when shared_ptr itself cannot allocate.
  auto lease = Lease(new Allocation(state_, constant, 0));
  lease->bytes_ = bytes;
  state_->live[constant] += bytes;
  state_->peak[constant] = std::max(state_->peak[constant], state_->live[constant]);
  return lease;
}

std::size_t MemoryBudget::live(bool constant) const { return state_->live[constant]; }

std::size_t MemoryBudget::peak(bool constant) const { return state_->peak[constant]; }

} // namespace specferry::np101
