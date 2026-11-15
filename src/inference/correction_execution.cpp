#include "inference/correction_execution.hpp"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace specferry::inference {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace

CorrectionLoop::CorrectionLoop(GenerationPolicy policy, CorrectionGenerate generate,
                               bool supports_retention)
    : policy_(policy), session_(policy), generate_(std::move(generate)),
      supports_retention_(supports_retention) {
  if (!generate_) {
    throw std::invalid_argument("correction loop requires a generation callback");
  }
}

GenerationHistory CorrectionLoop::history(const std::vector<std::int32_t> &prefix,
                                          const Generation &generation, unsigned maximum) const {
  if (generation.tokens.empty() || generation.tokens.size() > maximum ||
      generation.consumed > policy_.capacity ||
      std::uint64_t(generation.consumed) + 1 != prefix.size() + generation.tokens.size()) {
    throw std::runtime_error("correction executor returned an incomplete generation");
  }
  GenerationHistory result{prefix, generation.consumed, true};
  result.tokens.insert(result.tokens.end(), generation.tokens.begin(), generation.tokens.end());
  return result;
}

CorrectionStep CorrectionLoop::start(const std::vector<std::int32_t> &prompt, unsigned maximum) {
  if (!maximum) {
    throw std::invalid_argument("correction draft length must be positive");
  }
  session_.reset(prompt);
  CorrectionStep step;
  step.reset = true;
  const auto start = Clock::now();
  try {
    step.generation = generate_(session_.accepted_tokens(), maximum, false);
    step.proposal =
        session_.propose(history(session_.accepted_tokens(), *step.generation, maximum));
  } catch (...) {
    session_.invalidate();
    throw;
  }
  step.response_seconds = elapsed(start);
  return step;
}

CorrectionStep CorrectionLoop::apply(const CorrectionDecision &decision, unsigned maximum) {
  // Invalid decisions and zero draft length fail before touching the device or
  // accepting any tokens. EOS/capacity completion requires no further generation.
  const auto plan = session_.plan(decision);
  if (!plan.finished && !maximum) {
    throw std::invalid_argument("correction draft length must be positive");
  }
  CorrectionStep step;
  step.finished = plan.finished;
  const auto start = Clock::now();
  if (plan.finished) {
    session_.commit(plan);
  } else {
    step.reset = plan.reset_required || !supports_retention_;
    step.retained_consumed = step.reset ? 0 : plan.retained_consumed;
    step.replayed_prefix_tokens = step.reset ? unsigned(plan.full_prefix.size()) : 0;
    try {
      step.generation = generate_(plan.full_prefix, maximum, !step.reset);
      const auto generated = history(plan.full_prefix, *step.generation, maximum);
      session_.commit(plan);
      step.proposal = session_.propose(generated);
    } catch (...) {
      // A failed or malformed callback may have partially changed device state.
      // Recovery is a new start with a usable executor, never a second apply.
      session_.invalidate();
      throw;
    }
  }
  step.response_seconds = elapsed(start);
  return step;
}
} // namespace specferry::inference
