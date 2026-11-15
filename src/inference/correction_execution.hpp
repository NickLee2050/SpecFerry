#pragma once

#include "inference/correction.hpp"

namespace specferry::inference {
// The callback owns one model's state and must not be called independently while
// a loop is active. false resets all model state; true extends its exact history.
using CorrectionGenerate =
    std::function<Generation(const std::vector<std::int32_t> &, unsigned, bool)>;

struct CorrectionStep {
  std::optional<CorrectionProposal> proposal;
  std::optional<Generation> generation;
  bool finished = false;
  bool reset = false;
  unsigned retained_consumed = 0;
  unsigned replayed_prefix_tokens = 0;
  // Includes state reset and replay, unlike the legacy generation timer.
  double response_seconds = 0;
};

// Local, single-branch correction. Decisions are supplied by the caller; there is
// no target model, acceptance heuristic or communication layer here. Rejected
// branches use bounded reset/replay, preserving recurrent as well as KV semantics.
class CorrectionLoop {
public:
  CorrectionLoop(GenerationPolicy policy, CorrectionGenerate generate,
                 bool supports_retention = true);
  CorrectionLoop(const CorrectionLoop &) = delete;
  CorrectionLoop &operator=(const CorrectionLoop &) = delete;
  CorrectionStep start(const std::vector<std::int32_t> &prompt, unsigned maximum);
  CorrectionStep apply(const CorrectionDecision &decision, unsigned maximum);

  const CorrectionSession &session() const { return session_; }

private:
  GenerationHistory history(const std::vector<std::int32_t> &prefix, const Generation &generation,
                            unsigned maximum) const;
  GenerationPolicy policy_;
  CorrectionSession session_;
  CorrectionGenerate generate_;
  bool supports_retention_;
};
} // namespace specferry::inference
