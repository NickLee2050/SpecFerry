#include "inference/correction.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
using namespace specferry::inference;

void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("endpoint correction contract failed");
  }
}

template <class Action> void rejects(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected);
}

constexpr GenerationPolicy policy{100, 32, 2, 99};

void acceptance_contracts() {
  for (unsigned count : {0, 1, 3}) {
    for (const auto replacement :
         {std::optional<std::int32_t>{}, std::optional<std::int32_t>{20}}) {
      CorrectionSession session(policy, {2, 5});
      const auto proposal = session.propose({{2, 5, 6, 7, 8}, 4, true});
      require(proposal.tokens == std::vector<std::int32_t>({6, 7, 8}));
      require(session.accepted_tokens() == std::vector<std::int32_t>({2, 5}));
      const auto plan = session.plan({proposal.revision, count, replacement});
      std::vector<std::int32_t> expected{2, 5};
      expected.insert(expected.end(), proposal.tokens.begin(), proposal.tokens.begin() + count);
      if (replacement) {
        expected.push_back(*replacement);
      }
      require(plan.full_prefix == expected && !plan.finished);
      require(plan.reset_required == (count != 3));
      require(plan.retained_consumed == (count == 3 ? 4 : 0));
      // Planning itself does not publish draft tokens as accepted history.
      require(session.accepted_tokens() == std::vector<std::int32_t>({2, 5}));
      session.commit(plan);
      require(session.accepted_tokens() == expected && !session.pending_proposal());
      rejects([&] { session.commit(plan); });
    }
  }
  CorrectionSession session(policy, {2});
  const auto proposal = session.propose({{2, 3, 4}, 2, true});
  // Replacing only the unconsumed last prediction still resets: the adapter's
  // logical history includes it, and the current adapter has no truncate API.
  require(session.plan({proposal.revision, 1, 5}).reset_required);
  const auto same_token = session.plan({proposal.revision, 1, 4});
  require(!same_token.reset_required && same_token.retained_consumed == 2);
}

void boundary_contracts() {
  CorrectionSession session({100, 3, 2, 99});
  require(session.accepted_tokens() == std::vector<std::int32_t>{2});
  const auto capacity = session.propose({{2, 3, 4, 5}, 3, true});
  const auto full = session.plan({capacity.revision, 3});
  require(full.finished && !full.reset_required && full.retained_consumed == 3);
  rejects([&] { session.plan({capacity.revision, 3, 6}); });
  const auto rejected_tail = session.plan({capacity.revision, 2});
  require(!rejected_tail.finished && rejected_tail.reset_required);
  const auto replaced_tail = session.plan({capacity.revision, 2, 6});
  require(replaced_tail.finished && replaced_tail.reset_required);
  session.commit(full);
  rejects([&] { session.propose({{2, 3, 4, 5, 6}, 4, true}); });

  session.reset({99});
  require(!session.finished()); // A prompt may itself end in EOS.
  const auto eos = session.propose({{99, 3, 99}, 2, true});
  require(session.plan({eos.revision, 2}).finished);
  require(!session.plan({eos.revision, 1, 4}).finished);
  require(session.plan({eos.revision, 0, 99}).finished);
  rejects([&] { session.plan({eos.revision, 2, 4}); });
  session.commit(session.plan({eos.revision, 0, 99}));
  require(session.finished() && session.accepted_tokens() == std::vector<std::int32_t>({99, 99}));

  CorrectionSession single({100, 1, 2, 99});
  const auto one = single.propose({{2, 99}, 1, true});
  require(single.plan({one.revision, 1}).finished);
  require(!single.plan({one.revision, 0}).finished);
}

void invalid_input_contracts() {
  rejects([] { CorrectionSession session({0, 1, 0, 0}); });
  rejects([] { CorrectionSession session({100, 0, 2, 99}); });
  rejects([] { CorrectionSession session({100, 1, 100, 99}); });
  rejects([] { CorrectionSession session({100, 1, 2, 100}); });
  CorrectionSession session(policy, {2, 5});
  const auto initial_revision = session.revision();
  rejects([&] { session.plan({initial_revision, 0}); });
  rejects([&] { session.reset({-1}); });
  rejects([&] { session.reset(std::vector<std::int32_t>(33, 2)); });
  for (const auto &history :
       std::vector<GenerationHistory>{{{}, 0, true},
                                      {{2, 5}, 1, true},
                                      {{2, 5, 6}, 2, false},
                                      {{2, 5, 6}, 1, true},
                                      {{2, 5, 6}, 3, true},
                                      {{2, 4, 6}, 2, true},
                                      {{2, 5, -1}, 2, true},
                                      {{2, 5, 100}, 2, true},
                                      {{2, 5, 99, 6}, 3, true},
                                      {std::vector<std::int32_t>(34, 2), 33, true}}) {
    rejects([&] { session.propose(history); });
    require(session.revision() == initial_revision && !session.pending_proposal());
  }
  const auto proposal = session.propose({{2, 5, 6, 7}, 3, true});
  rejects([&] { session.propose({{2, 5, 6}, 2, true}); });
  rejects([&] { session.plan({proposal.revision - 1, 1}); });
  rejects([&] { session.plan({proposal.revision, 3}); });
  rejects([&] { session.plan({proposal.revision, 0, -1}); });
  rejects([&] { session.plan({proposal.revision, 0, 100}); });
  const auto canonical = session.plan({proposal.revision, 1, 9});
  for (unsigned field = 0; field < 4; ++field) {
    auto forged = canonical;
    if (field == 0) {
      forged.full_prefix.back() = 8;
    } else if (field == 1) {
      ++forged.retained_consumed;
    } else if (field == 2) {
      forged.reset_required = !forged.reset_required;
    } else {
      forged.finished = !forged.finished;
    }
    rejects([&] { session.commit(forged); });
    require(session.revision() == proposal.revision && session.pending_proposal());
  }
  session.reset({2, 5});
  rejects([&] { session.commit(canonical); });
  const auto next = session.propose({{2, 5, 6, 7}, 3, true});
  require(next.revision > proposal.revision);
  rejects([&] { session.plan(canonical.decision); });
  session.invalidate();
  require(!session.valid() && !session.pending_proposal());
  rejects([&] { session.plan({next.revision, 2}); });
  rejects([&] { session.propose({{2, 5, 6}, 2, true}); });
  session.reset();
  require(session.valid() && !session.finished());
}

struct FakeModel {
  std::vector<std::int32_t> consumed;
  GenerationHistory history;
  unsigned resets = 0;

  Generation generate(const std::vector<std::int32_t> &prefix, bool retain) {
    if (!retain) {
      history = {};
    }
    const GenerationExecutor executor{[&] {
                                        consumed.clear();
                                        ++resets;
                                      },
                                      [&](std::int32_t token) { consumed.push_back(token); },
                                      [&] {
                                        unsigned state = 7;
                                        for (const auto token : consumed) {
                                          state = (state * 3 + unsigned(token)) % 89;
                                        }
                                        return std::int32_t(state + 3);
                                      }};
    return generate_tokens(policy, executor, prefix, 3, {}, {}, &history);
  }
};

void replay_contracts() {
  CorrectionSession session(policy, {2, 5});
  FakeModel model;
  model.generate(session.accepted_tokens(), false);
  auto proposal = session.propose(model.history);
  for (const auto accepted : {3U, 1U, 0U, 2U, 3U}) {
    const auto plan = session.plan({proposal.revision, accepted, 20});
    const auto resets_before = model.resets;
    const auto continued = model.generate(plan.full_prefix, !plan.reset_required);
    FakeModel reference;
    const auto replayed = reference.generate(plan.full_prefix, false);
    require(continued.tokens == replayed.tokens && model.consumed == reference.consumed);
    require(model.resets == resets_before + unsigned(plan.reset_required));
    require(model.history.consumed + 1 == model.history.tokens.size());
    require(std::equal(model.consumed.begin(), model.consumed.end(), model.history.tokens.begin()));
    session.commit(plan);
    require(session.accepted_tokens() == plan.full_prefix);
    proposal = session.propose(model.history);
  }
}
} // namespace

int main() {
  try {
    acceptance_contracts();
    boundary_contracts();
    invalid_input_contracts();
    replay_contracts();
    std::cout << "Endpoint acceptance, correction, stale revision and replay contracts passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
