#include "inference/correction.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace specferry::inference {
CorrectionSession::CorrectionSession(GenerationPolicy policy,
                                     const std::vector<std::int32_t> &prompt)
    : policy_(policy) {
  if (!policy.capacity || !policy.vocabulary ||
      policy.vocabulary > std::numeric_limits<std::int32_t>::max() ||
      policy.bos >= policy.vocabulary || policy.eos >= policy.vocabulary) {
    throw std::invalid_argument("invalid correction generation policy");
  }
  reset(prompt);
}

void CorrectionSession::validate_token(std::int32_t token) const {
  if (token < 0 || unsigned(token) >= policy_.vocabulary) {
    throw std::invalid_argument("correction token exceeds vocabulary");
  }
}

void CorrectionSession::require_revision_space() const {
  if (revision_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("correction session revision exhausted");
  }
}

void CorrectionSession::reset(const std::vector<std::int32_t> &prompt) {
  auto tokens = prompt.empty() ? std::vector<std::int32_t>{std::int32_t(policy_.bos)} : prompt;
  if (tokens.size() > policy_.capacity) {
    throw std::out_of_range("correction prompt exceeds capacity");
  }
  for (const auto token : tokens) {
    validate_token(token);
  }
  require_revision_space();
  accepted_ = std::move(tokens);
  pending_.reset();
  ++revision_;
  finished_ = false;
  valid_ = true;
}

CorrectionProposal CorrectionSession::propose(const GenerationHistory &generated) {
  if (!valid_ || finished_ || pending_) {
    throw std::logic_error("correction session cannot accept a new proposal");
  }
  if (!generated.valid || generated.tokens.size() <= accepted_.size() ||
      generated.consumed > policy_.capacity ||
      std::uint64_t(generated.consumed) + 1 != generated.tokens.size() ||
      !std::equal(accepted_.begin(), accepted_.end(), generated.tokens.begin())) {
    throw std::invalid_argument("draft history does not extend the accepted correction prefix");
  }
  for (std::size_t index = accepted_.size(); index < generated.tokens.size(); ++index) {
    validate_token(generated.tokens[index]);
    if (unsigned(generated.tokens[index]) == policy_.eos && index + 1 != generated.tokens.size()) {
      throw std::invalid_argument("draft proposal contains tokens after EOS");
    }
  }
  require_revision_space();
  CorrectionProposal proposal{
      revision_ + 1, {generated.tokens.begin() + accepted_.size(), generated.tokens.end()}};
  pending_ = generated;
  ++revision_;
  return proposal;
}

CorrectionPlan CorrectionSession::plan(const CorrectionDecision &decision) const {
  if (!valid_ || !pending_ || decision.revision != revision_) {
    throw std::invalid_argument("correction decision has no matching pending revision");
  }
  const auto proposal_size = pending_->tokens.size() - accepted_.size();
  if (decision.accepted_count > proposal_size) {
    throw std::out_of_range("correction accepted count exceeds the pending proposal");
  }
  if (decision.token) {
    validate_token(*decision.token);
  }
  CorrectionPlan result{decision,
                        {pending_->tokens.begin(),
                         pending_->tokens.begin() + accepted_.size() + decision.accepted_count},
                        0,
                        true,
                        false};
  const bool accepted_eos =
      decision.accepted_count && unsigned(result.full_prefix.back()) == policy_.eos;
  if (decision.token) {
    if (accepted_eos) {
      throw std::invalid_argument("correction cannot append a token after accepted EOS");
    }
    result.full_prefix.push_back(*decision.token);
  }
  // Capacity counts consumed tokens; one final prediction may remain unconsumed.
  if (result.full_prefix.size() - 1 > policy_.capacity) {
    throw std::out_of_range("corrected prefix exceeds generation capacity");
  }
  result.finished = accepted_eos || (decision.token && unsigned(*decision.token) == policy_.eos) ||
                    result.full_prefix.size() > policy_.capacity;
  // Existing model adapters retain only an exact extension of their complete draft
  // history, including its unconsumed last prediction. Rejection uses reset/replay.
  result.reset_required =
      result.full_prefix.size() < pending_->tokens.size() ||
      !std::equal(pending_->tokens.begin(), pending_->tokens.end(), result.full_prefix.begin());
  if (!result.reset_required) {
    result.retained_consumed = pending_->consumed;
  }
  return result;
}

void CorrectionSession::commit(const CorrectionPlan &candidate) {
  const auto canonical = plan(candidate.decision);
  if (candidate.full_prefix != canonical.full_prefix ||
      candidate.retained_consumed != canonical.retained_consumed ||
      candidate.reset_required != canonical.reset_required ||
      candidate.finished != canonical.finished) {
    throw std::invalid_argument("correction plan differs from its validated decision");
  }
  require_revision_space();
  accepted_ = canonical.full_prefix;
  pending_.reset();
  finished_ = canonical.finished;
  ++revision_;
}

void CorrectionSession::invalidate() {
  valid_ = false;
  pending_.reset();
}
} // namespace specferry::inference
