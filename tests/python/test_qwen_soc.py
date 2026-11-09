"""CPU-only Qwen SoC request, routing and state acceptance contracts."""

import copy
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.models.qwen3_5.components import Components
from specferry.models.qwen3_5.validation import evaluate_generation
from specferry.platforms.soc import board_runner as board
from specferry.platforms.soc import remote
from specferry.platforms.soc.qwen_deployment import validate_request


class QwenSocTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.request = self.root / "request"
        self.request.mkdir()
        self.config = {
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_value_heads": 16,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
        }
        self.config["layer_types"] *= 6
        Components.from_text_config(self.config, capacity=64).write(self.request)
        self.metadata = {
            "tokens": [7, 8],
            "capacity": 64,
            "maximum": 2,
            "repeats": 1,
            "deployment_sha256": "a" * 64,
            "reference": {"sha256": "b" * 64},
            "selection": {
                "effective": {
                    "eos_token_id": 248044,
                    "do_sample": False,
                    "num_beams": 1,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 2,
                }
            },
        }
        self.reference = {
            "tokens": [9, 10],
            "consumed": 3,
            "stop_reason": "max_new_tokens",
            "prompt_length": 2,
            "reference": {"sha256": "b" * 64},
            "snapshots": {
                "hidden": {"dtype": "<f2", "shape": [1, 1, 1024]},
                "recurrent": {"dtype": "<f4", "shape": [1, 16, 128, 128]},
                **{name: {"dtype": "<f2", "shape": [1, 2, 3, 256]} for name in ("keys", "values")},
            },
        }
        board.save(self.request / "request.json", self.metadata)
        board.save(self.request / "reference.json", self.reference)
        (self.request / "tokens.txt").write_text("7 8\n")
        (self.request / "policy.txt").write_text("248320 64 0 248044 2 1\n")
        for name, value in {
            "embedding-lookup": "gather-nd",
            "matmul-feed-forward-layer": "9",
            "token-readback": "mapped",
            "head-backend": "shared-fcl",
            "state-commit": "combined",
        }.items():
            (self.request / (name + ".txt")).write_text(value + "\n")
        for name, contract in self.reference["snapshots"].items():
            np.zeros(contract["shape"], dtype=contract["dtype"]).tofile(
                self.request / f"expected.{name}.bin"
            )

    def validate(self):
        return validate_request(self.request, {"text_config": self.config}, "a" * 64)

    def test_split_shared_head_keeps_request_and_reference_policy(self):
        (self.request / "head-backend.txt").write_text("shared-fcl-split\n")
        self.assertEqual(self.validate()["head_backend"], "shared-fcl-split")
        self.assertEqual(self.validate()["maximum"], 2)

    def test_request_rejects_mismatched_tokens_precision_and_unsupported_eos(self):
        self.assertEqual(self.validate()["capacity"], 64)
        (self.request / "tokens.txt").write_text("7 11\n")
        with self.assertRaisesRegex(ValueError, "policy differs"):
            self.validate()
        (self.request / "tokens.txt").write_text("7 8\n")
        bad = copy.deepcopy(self.metadata)
        bad["selection"]["effective"]["eos_token_id"] = [248044, 248046]
        board.save(self.request / "request.json", bad)
        with self.assertRaisesRegex(ValueError, "policy differs"):
            self.validate()
        board.save(self.request / "request.json", self.metadata)
        bad = copy.deepcopy(self.reference)
        bad["snapshots"]["recurrent"]["dtype"] = "<f2"
        board.save(self.request / "reference.json", bad)
        with self.assertRaisesRegex(ValueError, "precision/size differs"):
            self.validate()

    def test_model_routing_is_explicit_and_executable_must_be_verified(self):
        manifest = {"model_family": "qwen3_5"}
        command = board.native_command(self.root, manifest, self.root / "run")
        self.assertEqual(
            command[1:4], [str(self.root / n) for n in ("weights", "model", "request")]
        )
        binary = Path(command[0])
        binary.parent.mkdir()
        header = bytearray(20)
        header[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<H", header, 18, 243)
        binary.write_bytes(header)
        binary.chmod(0o755)
        manifest.update(
            schema_version=1,
            target="np101-soc-riscv64",
            files={
                "bin/specferry_qwen_generate": {
                    "bytes": 20,
                    "sha256": board.sha256(binary),
                    "executable": True,
                },
            },
        )
        board.save(self.root / "bundle.json", manifest)
        board.verify_bundle(self.root, check_sdk=False)
        manifest["files"]["bin/specferry_qwen_generate"]["executable"] = False
        board.save(self.root / "bundle.json", manifest)
        with self.assertRaisesRegex(ValueError, "verified file inventory"):
            board.verify_bundle(self.root, check_sdk=False)
        with self.assertRaisesRegex(ValueError, "model family"):
            board.native_command(self.root, {"model_family": "../../bin/sh"}, self.root)

    def prepare_device(self):
        device = self.root / "board/native"
        device.mkdir(parents=True)
        for name in self.reference["snapshots"]:
            if name in ("keys", "values"):
                np.zeros((1, 2, 64, 256), dtype="<f2").tofile(device / f"measured.0.{name}.bin")
            else:
                shutil.copyfile(
                    self.request / f"expected.{name}.bin", device / f"measured.0.{name}.bin"
                )
        run = {
            "tokens": [9, 10],
            "consumed": 3,
            "stop_reason": "max_new_tokens",
            "total_seconds": 1.0,
            "first_token_seconds": 0.5,
            "prefill_seconds": 0.4,
            "token_seconds": [0.5],
        }
        board.save(device / "measured.0.json", run)
        board.save(
            device / "execution.json",
            {
                "released": True,
                "const_live_bytes": 0,
                "nonconst_live_bytes": 0,
                "const_peak_bytes": 100,
                "nonconst_peak_bytes": 100,
            },
        )
        return device

    def test_numerical_acceptance_rejects_bad_state_and_unwritten_kv(self):
        device = self.prepare_device()
        self.assertTrue(evaluate_generation(self.request, device)["passed"])
        path = device / "measured.0.keys.bin"
        values = np.fromfile(path, dtype="<f2").reshape(1, 2, 64, 256)
        values[0, 0, 3, 0] = 1
        values.tofile(path)
        result = evaluate_generation(self.request, device)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["performance"])
        values.fill(0)
        values.tofile(path)
        state = device / "measured.0.recurrent.bin"
        values = np.fromfile(state, dtype="<f4")
        values[0] = np.nan
        values.tofile(state)
        self.assertFalse(evaluate_generation(self.request, device)["passed"])

    def test_soc_acceptance_requires_soc_identity_and_selected_composition(self):
        device = self.prepare_device()
        manifest = {"model_family": "qwen3_5", "request_options": self.validate()}
        board.save(self.root / "bundle.json", manifest)
        profile = {"target": "soc", "architecture": "riscv64", "sdk_identity": "np101-soc-riscv64"}
        board.save(device / "platform.json", profile)
        board.save(
            device / "initialization.json",
            {
                "compiled_mixers": 18,
                "compiled_feed_forward": 23,
                "compiled_input_norm_layers": 18,
                "embedding_lookup": "gather-nd",
                "head_backend": "shared-fcl",
                "state_commit": "combined",
                "matmul_feed_forward_layer": 9,
                "token_readback": "mapped",
            },
        )
        self.assertTrue(remote.numerical_acceptance(self.root, device.parent)["passed"])
        manifest["request_options"]["projections"] = "shared-fcl"
        board.save(self.root / "bundle.json", manifest)
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        initialized = board.read_json(device / "initialization.json")
        board.save(device / "initialization.json", initialized | {"projections": "matmul"})
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        board.save(device / "initialization.json", initialized | {"projections": "shared-fcl"})
        self.assertTrue(remote.numerical_acceptance(self.root, device.parent)["passed"])
        manifest["request_options"]["kv_write_mode"] = "mapped"
        board.save(self.root / "bundle.json", manifest)
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        initialized = board.read_json(device / "initialization.json")
        board.save(device / "initialization.json", initialized | {"kv_write_mode": "mapped"})
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        execution = board.read_json(device / "execution.json")
        board.save(device / "execution.json", execution | {"kv_revalidations": 1})
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        board.save(device / "execution.json", execution | {"kv_revalidations": 0})
        self.assertTrue(remote.numerical_acceptance(self.root, device.parent)["passed"])
        board.save(device / "platform.json", profile | {"target": "ep"})
        result = remote.numerical_acceptance(self.root, device.parent)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["performance"])


if __name__ == "__main__":
    unittest.main()
