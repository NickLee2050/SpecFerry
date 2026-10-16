"""Legacy test/reference verifier, not used by the production vLLM adapter.

The Transformers experiment scripts use this independent debugging reference.
The cache contains committed tokens except the final token. A verification
forward consumes that final token followed by K candidates and predicts K+1
positions. This module does not implement probabilistic rejection sampling.
"""
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Decision:
    output_tokens: list[int]
    accepted_drafts: int
    first_rejected_index: int | None
    stopped_on_eos: bool


def decide_greedy(drafts, predictions, *, eos_token_id=None, max_output=None):
    if len(predictions) != len(drafts) + 1:
        raise ValueError("Need one prediction per draft plus one bonus prediction")
    if max_output is not None and max_output < 1:
        raise ValueError("max_output must be positive")
    emitted = []
    accepted = 0
    rejected = None
    for index, prediction in enumerate(predictions):
        matching = index < len(drafts) and drafts[index] == prediction
        if matching:
            accepted += 1
        elif index < len(drafts):
            rejected = index
        emitted.append(int(prediction))
        eos = eos_token_id is not None and prediction == eos_token_id
        if eos or not matching or (max_output is not None and len(emitted) >= max_output):
            return Decision(emitted, accepted, rejected, eos)
    raise AssertionError("Bonus prediction must terminate the round")


class GreedyVerifier:
    """Backend supplies prefill, forward, crop and cache_length operations."""

    def __init__(self, backend, prompt, *, eos_token_id=None, max_model_len=2048):
        if not prompt or len(prompt) > max_model_len:
            raise ValueError("Prompt must be nonempty and fit the context window")
        self.backend = backend
        self.tokens = list(prompt)
        self.eos_token_id = eos_token_id
        self.max_model_len = max_model_len
        self.cache = backend.prefill(self.tokens[:-1])
        self.finished = False
        self._check_cache()

    def _check_cache(self):
        if self.backend.cache_length(self.cache) != len(self.tokens) - 1:
            raise RuntimeError("KV cache must cover all committed tokens except the final token")

    def step(self, drafts, *, max_output=None):
        if self.finished:
            raise ValueError("Cannot verify after EOS")
        if max_output is not None and max_output < 1:
            raise ValueError("max_output must be positive")
        if len(self.tokens) + len(drafts) > self.max_model_len:
            raise ValueError("Candidate block exceeds the context window")
        self._check_cache()
        before = len(self.tokens)
        started = time.perf_counter()
        predictions, self.cache, stages = self.backend.forward(
            [self.tokens[-1], *drafts], self.cache)
        decision_start = time.perf_counter()
        decision = decide_greedy(drafts, predictions, eos_token_id=self.eos_token_id,
                                 max_output=max_output)
        decision_end = time.perf_counter()
        self.tokens.extend(decision.output_tokens)
        crop_start = time.perf_counter()
        self.backend.crop(self.cache, len(self.tokens) - 1)
        crop_end = time.perf_counter()
        self.finished = decision.stopped_on_eos
        self._check_cache()
        return {
            "draft_tokens": len(drafts), "prediction_positions": len(predictions),
            "output_tokens": decision.output_tokens, "accepted_drafts": decision.accepted_drafts,
            "first_rejected_index": decision.first_rejected_index,
            "stopped_on_eos": decision.stopped_on_eos,
            "context_before": before, "context_after": len(self.tokens),
            "cache_after": self.backend.cache_length(self.cache),
            "decision_seconds": decision_end - decision_start,
            "rollback_seconds": crop_end - crop_start,
            "total_seconds": time.perf_counter() - started,
            **stages,
        }

    def rewind_for_benchmark(self, prompt):
        """Reuse an unchanged prefix between trials; never reuse candidate KV."""
        if self.tokens[:len(prompt)] != list(prompt):
            raise ValueError("Can only rewind to an unchanged committed prefix")
        self.backend.crop(self.cache, len(prompt) - 1)
        self.tokens = list(prompt)
        self.finished = False
        self._check_cache()
