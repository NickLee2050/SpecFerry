"""Allocation acceptance requires intact weights/states, coherent reports and cleanup."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from scripts import check_np101_capacity

from specferry.validation import allocation
from specferry.validation.allocation import BLOCK_BYTES, MIB, capacity_complete, weights_complete


def clean_evidence():
    return {
        "returncode": 0,
        "timeout": False,
        "process_group_exited": True,
        "remaining_processes": [],
        "device_recovery_required": False,
    }


def clean_report():
    return {
        "report_version": allocation.REPORT_VERSION,
        "phase": "complete",
        "released": True,
        "failure_phase": "",
        "error": "",
        "release_error": "",
        "readback_error": "",
        "first_mismatch": None,
    }


def capacity_report(storage="constant", dtype="F16", target_mib=64):
    return clean_report() | {
        "status": "target_reached",
        "storage": storage,
        "dtype": dtype,
        "pattern": allocation.PATTERN,
        "target_bytes": target_mib * MIB,
        "block_bytes": BLOCK_BYTES,
        "uploaded_bytes": target_mib * MIB,
        "verified_bytes": target_mib * MIB,
        "retained_tensors": (target_mib + 7) // 8,
        "rejected_block_bytes": 0,
        "allocation_error": "",
    }


def weights_report(storage="constant"):
    return clean_report() | {
        "status": "allocation_pass",
        "weight_storage": storage,
        "is_const": storage == "constant",
        "expected_weights": 2,
        "completed_weights": 2,
        "expected_weight_bytes": 16,
        "uploaded_weight_bytes": 16,
        "verified_weight_bytes": 16,
        "expected_state_bytes": 8,
        "uploaded_state_bytes": 8,
        "verified_state_bytes": 8,
        "state_allocation_complete": True,
        "payload_bytes": 24,
    }


class AllocationReportTests(unittest.TestCase):
    def test_complete_reports_require_the_requested_dtype_storage_and_size(self):
        for storage in allocation.STORAGE_MODES:
            self.assertTrue(weights_complete(weights_report(storage), clean_evidence(), storage))
            for dtype in allocation.FLOAT_DTYPES:
                with self.subTest(storage=storage, dtype=dtype):
                    report = capacity_report(storage, dtype)
                    self.assertTrue(capacity_complete(report, clean_evidence(), storage, dtype, 64))
                    self.assertFalse(
                        capacity_complete(report, clean_evidence(), storage, dtype, 65)
                    )
        self.assertFalse(
            capacity_complete(capacity_report(), clean_evidence(), "constant", "F32", 64)
        )
        self.assertFalse(weights_complete(weights_report(), clean_evidence(), "mutable"))

    def test_sdk_rejection_requires_intact_retained_blocks(self):
        report = capacity_report(target_mib=4096) | {
            "status": "allocation_limit",
            "uploaded_bytes": 3840 * MIB,
            "verified_bytes": 3840 * MIB,
            "retained_tensors": 480,
            "rejected_block_bytes": BLOCK_BYTES,
            "allocation_error": "AddTensor failed",
        }
        self.assertTrue(capacity_complete(report, clean_evidence(), "constant", "F16", 4096))
        for change in (
            {"verified_bytes": 0},
            {"rejected_block_bytes": 0},
            {"rejected_block_bytes": BLOCK_BYTES + 1},
            {"allocation_error": ""},
            {"error": "cannot save capacity progress"},
            {"uploaded_bytes": 0, "verified_bytes": 0, "retained_tensors": 0},
        ):
            with self.subTest(change=change):
                self.assertFalse(
                    capacity_complete(report | change, clean_evidence(), "constant", "F16", 4096)
                )

    def test_incomplete_or_contradictory_reports_never_pass(self):
        for change in (
            {"report_version": 1},
            {"status": "running"},
            {"phase": "release"},
            {"released": False},
            {"failure_phase": "readback"},
            {"error": "report IO failed"},
            {"release_error": "graph release failed"},
            {"readback_error": "byte_mismatch"},
            {"first_mismatch": {"byte_offset": 17}},
        ):
            with self.subTest(change=change):
                self.assertFalse(
                    capacity_complete(
                        capacity_report() | change, clean_evidence(), "constant", "F16", 64
                    )
                )
                self.assertFalse(
                    weights_complete(weights_report() | change, clean_evidence(), "constant")
                )
        for change in (
            {"allocation_error": "rejected"},
            {"pattern": "old-periodic-pattern"},
            {"retained_tensors": 1},
            {"uploaded_bytes": -1, "verified_bytes": -1},
        ):
            with self.subTest(change=change):
                self.assertFalse(
                    capacity_complete(
                        capacity_report() | change, clean_evidence(), "constant", "F16", 64
                    )
                )
        self.assertFalse(capacity_complete({}, clean_evidence(), "constant", "F16", 64))
        self.assertFalse(weights_complete({}, clean_evidence(), "constant"))

    def test_signal_timeout_or_residual_process_invalidates_matching_data(self):
        for change in (
            {"returncode": -8},
            {"timeout": True},
            {"process_group_exited": False},
            {"remaining_processes": [{"pid": 123}]},
            {"device_recovery_required": True},
        ):
            with self.subTest(change=change):
                evidence = clean_evidence() | change
                self.assertFalse(
                    capacity_complete(capacity_report(), evidence, "constant", "F16", 64)
                )
                self.assertFalse(weights_complete(weights_report(), evidence, "constant"))

    def test_weights_require_state_readback_and_complete_uploads(self):
        for change in (
            {"is_const": False},
            {"completed_weights": 1},
            {"uploaded_weight_bytes": 8},
            {"verified_weight_bytes": 8},
            {"uploaded_state_bytes": 0},
            {"verified_state_bytes": 0},
            {"state_allocation_complete": False},
            {"payload_bytes": 16},
        ):
            with self.subTest(change=change):
                self.assertFalse(
                    weights_complete(weights_report() | change, clean_evidence(), "constant")
                )


class AllocationRunnerTests(unittest.TestCase):
    def test_custom_binary_is_snapshotted_and_custom_sdk_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "custom-probe"
            binary.write_bytes(b"test executable")
            output = root / "run"
            output.mkdir()
            sdk = root / "custom-sdk"
            with (
                patch.object(allocation, "device_lock", return_value=contextlib.nullcontext()),
                patch.object(allocation.os, "access", return_value=True),
                patch.object(allocation, "run_device", return_value=clean_evidence()) as run,
            ):
                allocation.run_allocation(binary, ["arg"], output, sdk, 42)
            self.assertEqual((output / binary.name).read_bytes(), binary.read_bytes())
            run.assert_called_once_with(output / binary.name, ["arg"], output, sdk, 42)
            self.assertFalse((output / "binary.json").exists())

    def test_capacity_cli_passes_dtype_and_paths_without_opening_the_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            binary = root / "probe"
            sdk = root / "sdk"

            def execute(actual_binary, arguments, actual_output, actual_sdk, timeout):
                self.assertEqual(
                    (actual_binary, actual_output, actual_sdk, timeout), (binary, output, sdk, 42)
                )
                self.assertEqual(arguments, [str(output / "capacity.json"), "mutable", "F32", "64"])
                (output / "capacity.json").write_text(json.dumps(capacity_report("mutable", "F32")))
                return clean_evidence()

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "check_np101_capacity.py",
                        "--output",
                        str(output),
                        "--binary",
                        str(binary),
                        "--sdk-lib",
                        str(sdk),
                        "--timeout",
                        "42",
                        "--dtype",
                        "F32",
                        "--storage",
                        "mutable",
                        "--target-mib",
                        "64",
                    ],
                ),
                patch.object(check_np101_capacity, "run_allocation", side_effect=execute),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(check_np101_capacity.main(), 0)
            self.assertTrue(
                json.loads((output / "summary.json").read_text())["readback_and_release_passed"]
            )


if __name__ == "__main__":
    unittest.main()
