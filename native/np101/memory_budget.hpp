#pragma once

#include <cstddef>
#include <memory>

namespace specferry::np101 {
// Diagnostic compatibility policy, not a measurement of the current driver.
inline constexpr std::size_t segment_payload_limit = std::size_t{1024} * 1024 * 1024;

class MemoryBudget {
  struct State;

public:
  class Allocation {
  public:
    ~Allocation();
    Allocation(const Allocation &) = delete;
    Allocation &operator=(const Allocation &) = delete;

  private:
    friend class MemoryBudget;
    Allocation(std::shared_ptr<State> state, bool constant, std::size_t bytes);
    std::shared_ptr<State> state_;
    bool constant_;
    std::size_t bytes_;
  };

  using Lease = std::shared_ptr<Allocation>;
  explicit MemoryBudget(std::size_t limit = segment_payload_limit);
  Lease reserve(bool constant, std::size_t bytes);
  std::size_t live(bool constant) const;
  std::size_t peak(bool constant) const;

private:
  std::shared_ptr<State> state_;
};
} // namespace specferry::np101
