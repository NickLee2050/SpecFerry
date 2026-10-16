"""Numerical regressions for the compact reporting and graph-cache checks."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.validation.arrays import compare_arrays
from specferry.validation.generation_metrics import summarize


class ValidationTests(unittest.TestCase):
    def test_throughput_counts_only_intervals_after_first_prediction(self):
        run = {
            "tokens": [2, 3, 4],
            "prefill_seconds": 10,
            "first_token_seconds": 12,
            "token_seconds": [0.25, 0.75],
            "total_seconds": 13,
        }
        self.assertEqual(summarize([run])["decode_tokens_per_second"], 2)
        self.assertEqual(summarize([run])["ttft_seconds"], 12)
        self.assertIsNone(
            summarize([run | {"tokens": [2], "token_seconds": []}])["decode_tokens_per_second"]
        )
        for times in ([float("nan"), 1], [-1, 1], [100, 100], []):
            with self.assertRaises(ValueError):
                summarize([run | {"token_seconds": times}])

    def test_nonfinite_and_wrong_shape_cannot_pass(self):
        reference = np.ones(4)
        for actual in (np.zeros(4), np.ones(3), np.full(4, np.nan)):
            self.assertFalse(compare_arrays(actual, reference, 0.08, 0.02)["passed"])
