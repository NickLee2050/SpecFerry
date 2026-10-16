import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from specferry.verification import GreedyVerifier, decide_greedy


class ToyBackend:
    """History-dependent oracle detects stale rejected cache contents."""
    @staticmethod
    def next_token(history):
        return (sum((i + 1) * t for i, t in enumerate(history)) % 29) + 3

    def prefill(self, tokens):
        return list(tokens)

    def forward(self, tokens, cache):
        predictions = []
        for token in tokens:
            cache.append(token)
            predictions.append(self.next_token(cache))
        return predictions, cache, {}

    def crop(self, cache, length):
        del cache[length:]

    def cache_length(self, cache):
        return len(cache)


def oracle(prefix, count):
    history = list(prefix)
    for _ in range(count):
        history.append(ToyBackend.next_token(history))
    return history[len(prefix):]


class GreedyVerificationTests(unittest.TestCase):
    def test_all_accepted_adds_bonus(self):
        d = decide_greedy([4, 5, 6], [4, 5, 6, 7])
        self.assertEqual(d.output_tokens, [4, 5, 6, 7])
        self.assertEqual(d.accepted_drafts, 3)

    def test_every_rejection_position_discards_suffix(self):
        for index in range(4):
            drafts = [4, 5, 6, 7]
            drafts[index] = 100
            d = decide_greedy(drafts, [4, 5, 6, 7, 8])
            self.assertEqual(d.output_tokens, [4, 5, 6, 7][:index + 1])
            self.assertEqual(d.first_rejected_index, index)
            self.assertEqual(d.accepted_drafts, index)

    def test_eos_on_accept_correction_and_bonus(self):
        for drafts, predictions, accepted in [([4, 2, 6], [4, 2, 6, 7], 2),
                                              ([4, 9], [4, 2, 8], 1),
                                              ([4], [4, 2], 1)]:
            d = decide_greedy(drafts, predictions, eos_token_id=2)
            self.assertEqual(d.output_tokens, [4, 2])
            self.assertEqual(d.accepted_drafts, accepted)
            self.assertTrue(d.stopped_on_eos)

    def test_output_budget_limits_committed_cache(self):
        prefix = [2, 8, 10]
        v = GreedyVerifier(ToyBackend(), prefix)
        expected = oracle(prefix, 8)
        result = v.step(expected[:5], max_output=2)
        self.assertEqual(result['output_tokens'], expected[:2])
        self.assertEqual(v.cache, v.tokens[:-1])
        self.assertEqual(v.step([])['output_tokens'], expected[2:3])

    def test_multiple_rounds_match_autoregression_after_rollbacks(self):
        prefix = [2, 8, 10]
        expected = oracle(prefix, 48)
        v = GreedyVerifier(ToyBackend(), prefix)
        emitted = []
        round_id = 0
        while len(emitted) < len(expected):
            k = min(5, len(expected) - len(emitted) - 1)
            draft = expected[len(emitted):len(emitted) + k].copy()
            if draft and round_id % 3:
                draft[round_id % len(draft)] = 100
            result = v.step(draft, max_output=len(expected) - len(emitted))
            emitted.extend(result['output_tokens'])
            self.assertEqual(emitted, expected[:len(emitted)])
            self.assertEqual(v.cache, v.tokens[:-1])
            round_id += 1
        self.assertGreater(round_id, 8)

    def test_single_token_prompt_and_no_draft(self):
        v = GreedyVerifier(ToyBackend(), [2])
        self.assertEqual(v.step([])['output_tokens'], oracle([2], 1))

    def test_invalid_budget_and_context_fail_before_mutation(self):
        v = GreedyVerifier(ToyBackend(), [2, 3], max_model_len=3)
        for draft, limit in [([4, 5], None), ([], 0)]:
            with self.assertRaises(ValueError):
                v.step(draft, max_output=limit)
            self.assertEqual(v.tokens, [2, 3])
            self.assertEqual(v.cache, [2])

    def test_finished_session_rejects_further_generation(self):
        next_id = oracle([2], 1)[0]
        v = GreedyVerifier(ToyBackend(), [2], eos_token_id=next_id)
        v.step([])
        with self.assertRaises(ValueError):
            v.step([])

    def test_prediction_shape_checked(self):
        with self.assertRaises(ValueError):
            decide_greedy([3, 4], [3, 4])


if __name__ == '__main__':
    unittest.main()
