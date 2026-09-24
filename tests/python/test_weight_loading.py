"""Generic weight checks preserve provenance and make state allocation explicit."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from scripts import check_np101_allocation

from specferry.export.weights import write_native_index, write_weight_pack
from specferry.validation.allocation import (
    SEGMENT_LIMIT_BYTES,
    check_segment_budget,
    prepare_weight_check,
    state_payload_bytes,
    weights_complete,
)
from tests.python.test_capacity import clean_evidence, weights_report
from tests.python.weight_fixtures import complete_manifest, source_tensor


def tiny_pack(root):
    root.mkdir()
    entries = [
        source_tensor(root, "encoder.projection", torch.arange(8).half(), "F16"),
        source_tensor(root, "state.scale", torch.ones(4), "F32"),
    ]
    records = write_weight_pack(root, entries, root)
    write_native_index(root, records)
    manifest = complete_manifest(root, records, aliases={"shared.projection": "encoder.projection"})
    manifest.update(repo_id="fixture/custom-architecture", revision="fixture-revision")
    (root / "deployment-manifest.json").write_text(json.dumps(manifest))


class WeightLoadingTests(unittest.TestCase):
    def test_const_and_nonconst_budgets_are_separate_but_mutable_weights_share_state_pool(self):
        limit = SEGMENT_LIMIT_BYTES
        self.assertEqual(check_segment_budget(limit, limit, "constant")["const_bytes"], limit)
        self.assertEqual(check_segment_budget(limit - 1, 1, "mutable")["nonconst_bytes"], limit)
        for weights, states, storage in (
            (limit + 1, 0, "constant"),
            (1, limit + 1, "constant"),
            (limit, 1, "mutable"),
        ):
            with self.subTest(storage=storage), self.assertRaises(ValueError):
                check_segment_budget(weights, states, storage)

    def test_oversized_weight_pack_is_rejected_before_device_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "check_np101_allocation",
                        "--deployment",
                        "unused",
                        "--output",
                        str(Path(directory) / "run"),
                    ],
                ),
                patch.object(
                    check_np101_allocation,
                    "prepare_weight_check",
                    return_value={
                        "expected": {
                            "expected_weight_bytes": SEGMENT_LIMIT_BYTES + 1,
                            "expected_state_bytes": 0,
                        },
                    },
                ),
                patch.object(check_np101_allocation, "run_allocation") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(check_np101_allocation.main(), 1)
                run.assert_not_called()

    def test_historical_model_state_fixture_preserves_its_payload(self):
        path = Path(__file__).resolve().parents[1] / "fixtures/qwen3_5_allocation_states.txt"
        self.assertEqual(state_payload_bytes(path), 46_071_808)

    def test_prepare_only_accepts_arbitrary_identity_without_states_or_device_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_pack(root / "pack")
            output = root / "prepared"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "check_np101_allocation",
                        "--deployment",
                        str(root / "pack"),
                        "--prepare-only",
                        "--output",
                        str(output),
                    ],
                ),
                patch.object(check_np101_allocation, "run_allocation") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(check_np101_allocation.main(), 0)
            run.assert_not_called()
            inputs = json.loads((output / "inputs.json").read_text())
            self.assertEqual(inputs["repo_id"], "fixture/custom-architecture")
            self.assertEqual(
                inputs["expected"],
                {"expected_weights": 2, "expected_weight_bytes": 32, "expected_state_bytes": 0},
            )
            self.assertFalse(inputs["architecture_validated"])
            self.assertFalse((output / "allocation-states.txt").exists())

    def test_explicit_states_are_snapshotted_and_invalid_records_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_pack(root / "pack")
            output = root / "prepared"
            output.mkdir()
            states = root / "states.txt"
            states.write_text("specferry-allocation-states 1\nF32 4,8 2\nF16 8 1\n")
            inputs = prepare_weight_check(root / "pack", states, output)
            self.assertEqual(inputs["expected"]["expected_state_bytes"], 272)
            saved = (output / "allocation-states.txt").read_bytes()
            for record in (
                "",
                "F64 8 1",
                "F16 0 1",
                "F16 8 -1",
                "F16 8 1025",
                "F32 4096,4096 1",
                "F16 8, 1",
                "F16 8 1 extra",
            ):
                states.write_text("specferry-allocation-states 1\n" + record + "\n")
                with self.subTest(record=record), self.assertRaises(ValueError):
                    state_payload_bytes(states)
            self.assertEqual((output / "allocation-states.txt").read_bytes(), saved)

    def test_reports_must_match_requested_weights_and_optional_state_budget(self):
        report = weights_report() | {
            "expected_state_bytes": 0,
            "uploaded_state_bytes": 0,
            "verified_state_bytes": 0,
            "payload_bytes": 16,
        }
        expected = {"expected_weights": 2, "expected_weight_bytes": 16, "expected_state_bytes": 0}
        self.assertTrue(weights_complete(report, clean_evidence(), "constant", expected))
        for key in expected:
            with self.subTest(key=key):
                self.assertFalse(
                    weights_complete(
                        report, clean_evidence(), "constant", expected | {key: expected[key] + 1}
                    )
                )
        self.assertFalse(
            weights_complete(report | {"released": False}, clean_evidence(), "constant", expected)
        )

    def test_corrupt_pack_is_rejected_before_device_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_pack(root / "pack")
            (root / "pack/weights.bin").write_bytes(b"corrupt")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "check_np101_allocation",
                        "--deployment",
                        str(root / "pack"),
                        "--output",
                        str(root / "run"),
                    ],
                ),
                patch.object(check_np101_allocation, "run_allocation") as run,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(check_np101_allocation.main(), 1)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
