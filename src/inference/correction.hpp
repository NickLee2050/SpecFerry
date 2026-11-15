#pragma once

#include "inference/generation.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace specferry::inference {
struct CorrectionDecision {
  std::uint64_t revision;
  unsigned accepted_count;
  // Replacement after rejection, or a bonus token after full acceptance.
  std::optional<std::int32_t> token = {};
};

struct CorrectionProposal {
  std::uint64_t revision;
  std::vector<std::int32_t> tokens;
};

struct CorrectionPlan {
  CorrectionDecision decision;
  std::vector<std::int32_t> full_prefix;
  unsigned retained_consumed;
  bool reset_required;
  bool finished;
};

// One pending branch per endpoint. Accepted tokens are logical history; the pending
// GenerationHistory describes the draft state, whose last prediction is unconsumed.
// Planning never changes either history or executes a model. Execute a plan first,
// then commit it; invalidate the session if execution fails after modifying state.
class CorrectionSession {
public:
  explicit CorrectionSession(GenerationPolicy policy, const std::vector<std::int32_t> &prompt = {});

  void reset(const std::vector<std::int32_t> &prompt = {});
  CorrectionProposal propose(const GenerationHistory &generated);
  CorrectionPlan plan(const CorrectionDecision &decision) const;
  void commit(const CorrectionPlan &plan);
  void invalidate();

  const std::vector<std::int32_t> &accepted_tokens() const { return accepted_; }

  std::uint64_t revision() const { return revision_; }

  bool finished() const { return finished_; }

  bool valid() const { return valid_; }

  bool pending_proposal() const { return pending_.has_value(); }

private:
  void validate_token(std::int32_t token) const;
  void require_revision_space() const;

  GenerationPolicy policy_;
  std::vector<std::int32_t> accepted_;
  std::optional<GenerationHistory> pending_;
  std::uint64_t revision_ = 0;
  bool finished_ = false;
  bool valid_ = true;
};
} // namespace specferry::inference
