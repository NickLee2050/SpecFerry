"""CPU regression checks for checkpoint validation and numerical comparison (no model download)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from specferry.reference.checkpoint import inventory, local_path

try:
    import torch

    from specferry.reference.trace import compare
except ImportError:
    torch = None


class CheckpointBoundaryTests(unittest.TestCase):
    def test_incomplete_or_wrong_revision_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for manifest in ({"status": "running"}, {"status": "complete", "repo_id": "other"}):
                (root / "download-manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    inventory(root)

    def test_paths_cannot_escape_model(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../weight", "/tmp/weight", "a\\..\\weight"):
                with self.assertRaises(ValueError):
                    local_path(Path(directory), name)

    @unittest.skipIf(torch is None, "activate the SpecFerry Conda environment for numeric checks")
    def test_nonfinite_never_passes_comparison(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                compare(torch.tensor([value]), torch.zeros(1), 1e-2, 2e-2)


if __name__ == "__main__":
    unittest.main()
