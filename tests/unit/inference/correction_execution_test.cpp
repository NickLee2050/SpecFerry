#include "inference/correction_execution.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace specferry::inference;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Action> void rejects(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  require(rejected, "invalid correction execution was accepted");
}

constexpr GenerationPolicy policy{100, 64, 2, 99};
const std::vector<std::int32_t> prompt{2, 5};

struct ModelState {
  std::vector<std::int32_t> consumed;
  std::vector<std::int32_t> convolution;
  std::uint64_t recurrent = 7;
};

struct Invocation {
  std::vector<std::int32_t> prefix;
  unsigned maximum;
  bool retain;
};

// State deliberately depends on every consumed token and the recent convolution
// window. Truncating a token vector without rebuilding both states cannot pass.
struct FakeModel {
  GenerationPolicy selection;
  ModelState state;
  GenerationHistory history;
  std::vector<Invocation> calls;
  unsigned resets = 0;
  bool fail_next_consume = false;
  bool predict_eos = false;
  bool return_empty_generation = false;

  explicit FakeModel(GenerationPolicy settings = policy) : selection(settings) {}

  Generation generate(const std::vector<std::int32_t> &prefix, unsigned maximum, bool retain) {
    calls.push_back({prefix, maximum, retain});
    if (!retain) {
      history = {};
    }
    const GenerationExecutor executor{
        [&] {
          state = {};
          ++resets;
        },
        [&](std::int32_t token) {
          state.consumed.push_back(token);
          state.convolution.push_back(token);
          if (state.convolution.size() > 3) {
            state.convolution.erase(state.convolution.begin());
          }
          state.recurrent =
              (state.recurrent * 31 + unsigned(token) + unsigned(state.convolution.front()) * 7) %
              1000003;
          if (fail_next_consume) {
            fail_next_consume = false;
            throw std::runtime_error("injected failure after partial state mutation");
          }
        },
        [&] {
          if (predict_eos) {
            return std::int32_t(selection.eos);
          }
          // Reserve 95 for replacements and 99 for explicitly requested EOS.
          return std::int32_t(3 + (state.recurrent + unsigned(state.convolution.back())) % 87);
        }};
    auto result = generate_tokens(selection, executor, prefix, maximum, {}, {}, &history);
    if (return_empty_generation) {
      result.tokens.clear();
    }
    return result;
  }

  CorrectionGenerate callback() {
    return [&](const auto &prefix, unsigned maximum, bool retain) {
      return generate(prefix, maximum, retain);
    };
  }
};

void equivalent_state(const FakeModel &actual, const FakeModel &expected) {
  require(actual.state.consumed == expected.state.consumed, "consumed state differs from replay");
  require(actual.state.convolution == expected.state.convolution,
          "convolution state differs from replay");
  require(actual.state.recurrent == expected.state.recurrent,
          "recurrent state differs from replay");
  require(actual.history.valid && actual.history.tokens == expected.history.tokens &&
              actual.history.consumed == expected.history.consumed,
          "model-owned history differs from replay");
  require(actual.history.consumed + 1 == actual.history.tokens.size(),
          "final prediction must remain unconsumed");
  require(std::equal(actual.state.consumed.begin(), actual.state.consumed.end(),
                     actual.history.tokens.begin()),
          "consumed tokens do not match model-owned history");
}

void equivalent_generation(const Generation &actual, const Generation &expected) {
  require(actual.tokens == expected.tokens && actual.consumed == expected.consumed &&
              actual.stop_reason == expected.stop_reason,
          "corrected generation differs from fresh generation");
}

void acceptance_and_replay() {
  for (const auto accepted : {0U, 1U, 2U, 3U}) {
    for (const auto replacement :
         {std::optional<std::int32_t>{}, std::optional<std::int32_t>{95}}) {
      FakeModel model;
      CorrectionLoop loop(policy, model.callback());
      const auto initial = loop.start(prompt, 3);
      require(initial.reset && !initial.finished && initial.proposal && initial.generation,
              "start did not create a reset proposal");
      require(model.resets == 1 && !model.calls.back().retain, "start did not reset the model");

      auto prefix = prompt;
      prefix.insert(prefix.end(), initial.proposal->tokens.begin(),
                    initial.proposal->tokens.begin() + accepted);
      if (replacement) {
        prefix.push_back(*replacement);
      }
      const auto corrected = loop.apply({initial.proposal->revision, accepted, replacement}, 2);
      const bool reset = accepted != 3;
      require(corrected.proposal && corrected.generation && !corrected.finished,
              "correction did not produce the next draft");
      require(corrected.reset == reset && model.resets == 1 + unsigned(reset),
              "correction selected the wrong reset policy");
      require(model.calls.size() == 2 && model.calls.back().prefix == prefix &&
                  model.calls.back().maximum == 2 && model.calls.back().retain == !reset,
              "correction called the model with the wrong prefix or retention policy");
      require(corrected.retained_consumed == (reset ? 0 : initial.generation->consumed) &&
                  corrected.replayed_prefix_tokens == (reset ? prefix.size() : 0),
              "correction state preparation counts are incorrect");
      require(loop.session().accepted_tokens() == prefix && loop.session().pending_proposal(),
              "correction failed to publish the accepted prefix and next proposal");

      FakeModel reference;
      equivalent_generation(*corrected.generation, reference.generate(prefix, 2, false));
      equivalent_state(model, reference);

      // The repaired state must itself support the next retained generation.
      const auto next_prefix = model.history.tokens;
      const auto next = loop.apply({corrected.proposal->revision, 2}, 2);
      require(!next.reset && model.calls.back().retain,
              "repaired correction state could not be retained");
      FakeModel next_reference;
      equivalent_generation(*next.generation, next_reference.generate(next_prefix, 2, false));
      equivalent_state(model, next_reference);
    }
  }
}

void retention_fallback() {
  FakeModel model;
  CorrectionLoop loop(policy, model.callback(), false);
  const auto initial = loop.start(prompt, 3);
  auto prefix = model.history.tokens;
  prefix.push_back(95);
  const auto next = loop.apply({initial.proposal->revision, 3, 95}, 2);
  require(next.reset && next.retained_consumed == 0 &&
              next.replayed_prefix_tokens == prefix.size() && model.resets == 2 &&
              !model.calls.back().retain,
          "unsupported retention did not fall back to reset and replay");
  FakeModel reference;
  equivalent_generation(*next.generation, reference.generate(prefix, 2, false));
  equivalent_state(model, reference);
}

void terminal_decisions() {
  for (const bool generated_eos : {false, true}) {
    FakeModel model;
    model.predict_eos = generated_eos;
    CorrectionLoop loop(policy, model.callback());
    const auto initial = loop.start(prompt, 3);
    const CorrectionDecision decision = generated_eos
                                            ? CorrectionDecision{initial.proposal->revision, 1}
                                            : CorrectionDecision{initial.proposal->revision, 0, 99};
    const auto terminal = loop.apply(decision, 0);
    require(terminal.finished && !terminal.generation && !terminal.proposal &&
                model.calls.size() == 1 && loop.session().finished() &&
                !loop.session().pending_proposal(),
            "EOS completion invoked generation or left a pending proposal");
    require(loop.session().accepted_tokens().back() == 99, "EOS was not committed");
    rejects([&] { loop.apply(decision, 1); });
    require(model.calls.size() == 1, "terminal decision was replayed");
  }

  for (const bool replacement : {false, true}) {
    const GenerationPolicy bounded{100, 4, 2, 99};
    FakeModel model(bounded);
    CorrectionLoop loop(bounded, model.callback());
    const auto initial = loop.start(prompt, 4);
    require(initial.generation->stop_reason == "capacity" && initial.proposal->tokens.size() == 3,
            "capacity fixture did not reach its final pending prediction");
    const CorrectionDecision decision = replacement
                                            ? CorrectionDecision{initial.proposal->revision, 2, 95}
                                            : CorrectionDecision{initial.proposal->revision, 3};
    const auto terminal = loop.apply(decision, 0);
    require(terminal.finished && !terminal.generation && !terminal.proposal &&
                model.calls.size() == 1 && loop.session().accepted_tokens().size() == 5,
            "capacity completion invoked generation or lost the pending token");
  }
}

void invalid_decisions_preserve_state() {
  FakeModel model;
  CorrectionLoop loop(policy, model.callback());
  const auto initial = loop.start(prompt, 3);
  const auto revision = initial.proposal->revision;
  const auto original_state = model.state;
  const auto original_history = model.history;
  for (const auto decision : std::vector<CorrectionDecision>{
           {revision - 1, 1}, {revision, 4}, {revision, 0, -1}, {revision, 0, 100}}) {
    rejects([&] { loop.apply(decision, 2); });
  }
  rejects([&] { loop.apply({revision, 1, 95}, 0); });
  rejects([&] { loop.start({-1}, 2); });
  rejects([&] { loop.start(prompt, 0); });
  require(model.calls.size() == 1 && model.state.consumed == original_state.consumed &&
              model.state.convolution == original_state.convolution &&
              model.state.recurrent == original_state.recurrent &&
              model.history.tokens == original_history.tokens && model.history.valid &&
              loop.session().revision() == revision && loop.session().pending_proposal(),
          "invalid request touched the callback or discarded usable state");
  const auto next = loop.apply({revision, 1, 95}, 2);
  rejects([&] { loop.apply({revision, 1, 95}, 2); });
  require(model.calls.size() == 2 && loop.session().revision() == next.proposal->revision,
          "stale decision touched the callback");

  const GenerationPolicy bounded{100, 4, 2, 99};
  FakeModel capacity_model(bounded);
  CorrectionLoop capacity_loop(bounded, capacity_model.callback());
  const auto full = capacity_loop.start(prompt, 4);
  rejects([&] { capacity_loop.apply({full.proposal->revision, 3, 95}, 2); });
  require(capacity_model.calls.size() == 1 && capacity_loop.session().pending_proposal(),
          "over-capacity correction touched the callback");
}

void failure_and_restart() {
  for (const bool malformed : {false, true}) {
    FakeModel model;
    CorrectionLoop loop(policy, model.callback());
    const auto initial = loop.start(prompt, 3);
    const CorrectionDecision decision{initial.proposal->revision, 3, 95};
    model.fail_next_consume = !malformed;
    model.return_empty_generation = malformed;
    rejects([&] { loop.apply(decision, 2); });
    require(!loop.session().valid() && !loop.session().pending_proposal(),
            "failed or malformed execution left a usable correction session");
    rejects([&] { loop.apply(decision, 2); });
    require(model.calls.size() == 2, "invalidated session called the executor again");

    model.return_empty_generation = false;
    const auto restarted = loop.start({2, 8}, 2);
    require(restarted.reset && loop.session().valid() && loop.session().pending_proposal() &&
                model.resets == 2 && !model.calls.back().retain &&
                restarted.proposal->revision > initial.proposal->revision,
            "new start did not reset and recover the failed session");
    FakeModel reference;
    equivalent_generation(*restarted.generation, reference.generate({2, 8}, 2, false));
    equivalent_state(model, reference);
  }

  FakeModel model;
  CorrectionLoop loop(policy, model.callback());
  model.fail_next_consume = true;
  rejects([&] { loop.start(prompt, 3); });
  require(!loop.session().valid(), "failed initial generation left a valid session");
  const auto restarted = loop.start(prompt, 3);
  require(restarted.proposal && model.resets == 2, "initial failure could not be reset");
}
} // namespace

int main() {
  try {
    acceptance_and_replay();
    retention_fallback();
    terminal_decisions();
    invalid_decisions_preserve_state();
    failure_and_restart();
    std::cout << "Correction execution, state replay, retention and failure contracts passed.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
