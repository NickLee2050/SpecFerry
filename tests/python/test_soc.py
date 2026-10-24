"""SoC packaging and fail-closed supervision, without SDK/device execution."""

import ast
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.platforms.soc import board_runner as board
from specferry.platforms.soc import remote


class SocTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_board_script_uses_python39_and_only_standard_library(self):
        tree = ast.parse(Path(board.__file__).read_text(), feature_version=(3, 9))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in sys.stdlib_module_names for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, sys.stdlib_module_names)

    def test_diagnostic_only_acknowledges_exact_old_fault_occurrences(self):
        old = "[57512.756734] [galcore]: NP[0] hang, automatic recovery."
        new = old.replace("57512", "60000")
        pattern = re.compile("hang")
        self.assertEqual(board.new_faults(old, pattern, [old]), [])
        self.assertEqual(board.new_faults(old + "\n" + new, pattern, [old]), [new])
        self.assertEqual(board.new_faults(old + "\n" + old, pattern, [old]), [old])
        self.assertEqual(board.new_faults(old, pattern), [old])

    def test_bundle_rejects_corruption_wrong_architecture_and_missing_execution_bit(self):
        binary = self.root / "bin/specferry_opt_graph_generate"
        binary.parent.mkdir()
        header = bytearray(20)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, 243)
        binary.write_bytes(header)
        binary.chmod(0o755)
        record = {"bytes": 20, "sha256": board.sha256(binary), "executable": True}
        manifest = {
            "schema_version": 1,
            "target": "np101-soc-riscv64",
            "files": {"bin/specferry_opt_graph_generate": record},
        }
        board.save(self.root / "bundle.json", manifest)
        board.verify_bundle(self.root, check_sdk=False)
        binary.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "permission"):
            board.verify_bundle(self.root, check_sdk=False)
        binary.chmod(0o755)
        header[18] = 62  # x86-64 must not enter a board bundle.
        binary.write_bytes(header)
        with self.assertRaisesRegex(ValueError, "identity"):
            board.verify_bundle(self.root, check_sdk=False)
        record["sha256"] = board.sha256(binary)
        board.save(self.root / "bundle.json", manifest)
        with self.assertRaisesRegex(ValueError, "RISC-V"):
            board.verify_bundle(self.root, check_sdk=False)

    def test_bundle_and_remote_paths_cannot_escape_their_roots(self):
        for name in ("../weights.bin", "/tmp/weights.bin"):
            with self.assertRaises(ValueError):
                board.bundle_file(self.root, name)
        (self.root / "outside").symlink_to("/tmp")
        with self.assertRaises(ValueError):
            board.bundle_file(self.root, "outside/file")
        for name in ("/root", "/root/specferry-soc/../other", "relative"):
            with self.assertRaises(ValueError):
                remote.remote_path(name)

    def supervise(self, source, timeout=1, monitor=lambda: None):
        return board.supervise(
            [sys.executable, "-c", source],
            self.root,
            os.environ.copy(),
            self.root,
            timeout,
            monitor,
        )

    def test_deadline_stops_child_even_when_health_sampling_blocks(self):
        result = self.supervise(
            "import time; time.sleep(60)", timeout=0.15, monitor=lambda: time.sleep(0.4)
        )
        self.assertLess(result["exit"], 0)
        self.assertEqual(result["stop_reason"], "child deadline expired")
        self.assertEqual(result["remaining_processes"], [])
        fired = board.read_json(self.root / "watchdog.json")["expired_at"]
        self.assertLess(fired - result["child_start_monotonic_seconds"], 0.35)

    def test_kernel_fault_stops_child_without_waiting_for_deadline(self):
        result = self.supervise("import time; time.sleep(60)", monitor=lambda: "kernel fault")
        self.assertEqual(result["stop_reason"], "kernel fault")
        self.assertLess(result["exit"], 0)
        self.assertEqual(result["remaining_processes"], [])

    def test_monitor_error_still_terminates_and_records_owned_child(self):
        def broken():
            raise OSError("lost kernel records")

        with self.assertRaises(OSError):
            self.supervise("import time; time.sleep(60)", monitor=broken)
        result = board.read_json(self.root / "child.json")
        self.assertLess(result["exit"], 0)
        self.assertEqual(result["remaining_processes"], [])

    def test_clean_exit_and_deadline_limit(self):
        self.assertEqual(self.supervise("print('finished')")["exit"], 0)
        for limit in (0, 301):
            with self.assertRaises(ValueError):
                self.supervise("pass", timeout=limit)

    def test_sdk_success_does_not_replace_release_and_budget_checks(self):
        native = self.root / "native"
        native.mkdir()
        record = {
            "status": "executed",
            "released": True,
            "const_live_bytes": 0,
            "nonconst_live_bytes": 0,
            "const_peak_bytes": 100,
            "nonconst_peak_bytes": 100,
        }
        board.save(native / "execution.json", {"released": True})
        board.save(native / "run.json", record)
        self.assertTrue(board.native_complete(self.root))
        for change in (
            {"nonconst_live_bytes": 1},
            {"const_peak_bytes": 2**30 + 1},
            {"released": False},
        ):
            board.save(native / "run.json", record | change)
            self.assertFalse(board.native_complete(self.root))

    def test_existing_gate_prevents_any_ssh_launch(self):
        gate = self.root / "np101-recovery-required.json"
        board.save(gate, {"members": [], "pid_namespace": os.readlink("/proc/self/ns/pid")})
        original = gate.read_bytes()
        with (
            patch.object(remote, "RECOVERY_ROOT", self.root),
            patch.object(remote, "verify_bundle"),
            patch.object(remote, "ssh") as ssh,
        ):
            with self.assertRaises(RuntimeError):
                remote.run(self.root, "np101", "/root/specferry-soc/bundle", self.root / "run", 180)
            ssh.assert_not_called()
        self.assertEqual(gate.read_bytes(), original)

    def test_collection_transport_failure_keeps_pending_marker(self):
        output = self.root / "run"
        output.mkdir()
        gate = self.root / "np101-recovery-required.json"
        board.save(gate, {"evidence": str(output)})
        (self.root / "bundle.json").write_text("{}")
        board.save(
            output / "session.json",
            {
                "host": "np101",
                "remote_output": "/root/specferry-soc/runs/test",
                "bundle_sha256": board.sha256(self.root / "bundle.json"),
            },
        )
        original = gate.read_bytes()
        with (
            patch.object(remote, "RECOVERY_ROOT", self.root),
            patch.object(remote, "verify_bundle"),
            patch.object(remote, "ssh", side_effect=subprocess.CalledProcessError(255, "ssh")),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                remote.collect(self.root, output)
        self.assertEqual(gate.read_bytes(), original)

    def test_successful_diagnostic_restores_old_gate_byte_for_byte(self):
        output = self.root / "run"
        output.mkdir()
        (output / "board").mkdir()
        (self.root / "bundle.json").write_text("{}")
        digest = board.sha256(self.root / "bundle.json")
        retained = output / "retained-prior-host-marker.json"
        retained.write_bytes(b'{"reason": "old fault"}\n')
        gate = self.root / "np101-recovery-required.json"
        board.save(gate, {"evidence": str(output)})
        board.save(
            output / "session.json",
            {
                "host": "np101",
                "remote_output": "/root/specferry-soc/runs/test",
                "remote_bundle": "/root/specferry-soc/bundles/test",
                "bundle_sha256": digest,
                "diagnostic": {"host_marker_sha256": board.sha256(retained)},
            },
        )
        board.save(output / "board/result.json", {"clean": True, "bundle_sha256": digest})
        with (
            patch.object(remote, "RECOVERY_ROOT", self.root),
            patch.object(remote, "verify_bundle"),
            patch.object(remote, "host_health"),
            patch.object(remote, "numerical_acceptance", return_value={"passed": True}),
            patch.object(remote, "ssh"),
        ):
            result = remote.collect(self.root, output)
        self.assertTrue(result["passed"])
        self.assertFalse(result["recovery_cleared"])
        self.assertEqual(gate.read_bytes(), retained.read_bytes())

    def test_status_connection_loss_never_resubmits_child(self):
        (self.root / "bundle.json").write_text("{}")
        (self.root / "board_runner.py").write_text("# fixture")
        responses = [
            subprocess.CompletedProcess([], 0, stdout="123\n"),
            subprocess.CalledProcessError(255, "ssh"),
            subprocess.CompletedProcess([], 0, stdout="1\n"),
        ]
        with (
            patch.object(remote, "RECOVERY_ROOT", self.root),
            patch.object(remote, "verify_bundle"),
            patch.object(remote, "host_health"),
            patch.object(remote, "collect", return_value={"passed": True}),
            patch.object(remote.time, "sleep"),
            patch.object(remote, "ssh", side_effect=responses) as transport,
        ):
            result = remote.run(
                self.root, "np101", "/root/specferry-soc/bundles/test", self.root / "run", 180
            )
        self.assertTrue(result["passed"])
        commands = [call.args[1] for call in transport.call_args_list]
        self.assertEqual(sum("subprocess.Popen" in command for command in commands), 1)
        self.assertEqual(sum("result.json" in command for command in commands), 2)
