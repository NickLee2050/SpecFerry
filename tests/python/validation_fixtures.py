"""Numerical regressions for the compact reporting and graph-cache checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def write_generation_cache(fixture, device, metadata, expected_values):
    """Write independent CPU prefixes and padded device backings for validator tests."""
    for index, request in enumerate(metadata["requests"]):
        for layer in range(metadata["layers"]):
            expected = expected_values(layer, request["consumed"])
            actual = np.zeros(
                (metadata["heads"], metadata["capacity"], metadata["width"]), dtype="<f2"
            )
            actual[:, : request["consumed"]] = expected
            for name in ("keys", "values"):
                expected.tofile(fixture / f"expected.{index}.{layer}.{name}.bin")
                actual.tofile(device / f"measured.{index}.layer.{layer}.{name}.bin")


class GraphEvidenceFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture, self.device = self.root / "fixture", self.root / "device"
        self.fixture.mkdir()
        self.device.mkdir()
        self.results = []

    def save_metadata(self):
        (self.device / "initialization.json").write_text(json.dumps(self.initialized))
        (self.device / "run.json").write_text(json.dumps(self.native_run))
        for index, result in enumerate(self.results):
            (self.device / f"measured.{index}.json").write_text(json.dumps(result))
