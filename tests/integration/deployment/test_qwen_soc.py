"""CPU-only Qwen SoC request, routing and state acceptance contracts."""

import copy
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.deploy.execute.np101.soc import board_runner as board
from tools.deploy.execute.np101.soc import remote
from tools.deploy.export.qwen3_5.components import Components
from tools.deploy.pack.np101.qwen import validate_request
from tools.validate.qwen3_5.generation import evaluate_generation


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

    def test_batched_ffn_requires_complete_mapped_composition(self):
        self.assertEqual(self.validate()["ffn_batch_tokens"], 1)
        self.assertEqual(self.validate()["decoder_layout"], "separate")
        commit_file = self.request / "state-commit.txt"
        commit_file.write_text("ffn\n")
        with self.assertRaisesRegex(ValueError, "FFN state commit requires batched"):
            self.validate()
        commit_file.write_text("combined\n")
        layout_file = self.request / "decoder-layout.txt"
        layout_file.write_text("fused-attention-norm\n")
        self.metadata["decoder_layout"] = "fused-attention-norm"
        board.save(self.request / "request.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "requires batched FFN"):
            self.validate()
        layout_file.unlink()
        del self.metadata["decoder_layout"]
        self.metadata["capacity"] = 128
        board.save(self.request / "request.json", self.metadata)
        Components.from_text_config(self.config, capacity=128).write(self.request)
        (self.request / "policy.txt").write_text("248320 128 0 248044 2 1\n")
        for name, value in {
            "head-backend": "shared-fcl-split",
            "projections": "shared-fcl",
            "kv-write-mode": "mapped",
            "matmul-feed-forward-layer": "-1",
        }.items():
            (self.request / (name + ".txt")).write_text(value + "\n")
        for width in (8, 16):
            (self.request / "ffn-batch-tokens.txt").write_text(str(width) + "\n")
            self.assertEqual(self.validate()["ffn_batch_tokens"], width)
            self.assertEqual(self.validate()["matmul_feed_forward_layer"], -1)
            for layout in ("separate", "fused-attention-norm"):
                layout_file.write_text(layout + "\n")
                self.metadata["decoder_layout"] = layout
                board.save(self.request / "request.json", self.metadata)
                self.assertEqual(self.validate()["decoder_layout"], layout)
            commit_file.write_text("ffn\n")
            with self.assertRaisesRegex(ValueError, "state commit metadata/option"):
                self.validate()
            self.metadata["state_commit"] = "ffn"
            board.save(self.request / "request.json", self.metadata)
            self.assertEqual(self.validate()["state_commit"], "ffn")
            commit_file.write_text("unknown\n")
            with self.assertRaisesRegex(ValueError, "unsupported Qwen state commit"):
                self.validate()
            commit_file.write_text("combined\n")
            del self.metadata["state_commit"]
            board.save(self.request / "request.json", self.metadata)
        for capacity in (512, 128):
            self.metadata["capacity"] = capacity
            board.save(self.request / "request.json", self.metadata)
            Components.from_text_config(self.config, capacity=capacity).write(self.request)
            (self.request / "policy.txt").write_text(f"248320 {capacity} 0 248044 2 1\n")
            self.assertEqual(self.validate()["capacity"], capacity)
        for retired in ("shared-residual", "unknown"):
            layout_file.write_text(retired + "\n")
            with self.assertRaisesRegex(ValueError, "layout metadata/option"):
                self.validate()
        layout_file.write_text("separate\n")
        with self.assertRaisesRegex(ValueError, "layout metadata/option"):
            self.validate()
        layout_file.write_text("fused-attention-norm\n")
        (self.request / "matmul-feed-forward-layer.txt").write_text("9\n")
        with self.assertRaisesRegex(ValueError, "execution option differs"):
            self.validate()
        (self.request / "matmul-feed-forward-layer.txt").write_text("-1\n")
        (self.request / "kv-write-mode.txt").write_text("copy-graph\n")
        with self.assertRaisesRegex(ValueError, "C128/C512 mapped"):
            self.validate()
        (self.request / "ffn-batch-tokens.txt").write_text("4\n")
        with self.assertRaisesRegex(ValueError, "batch width"):
            self.validate()

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
        options = manifest["request_options"]
        options.update(
            ffn_batch_tokens=8,
            matmul_feed_forward_layer=-1,
            capacity=128,
            head_backend="shared-fcl-split",
        )
        board.save(self.root / "bundle.json", manifest)
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        initialized = board.read_json(device / "initialization.json")
        initialized.update(
            ffn_batch_tokens=8,
            matmul_feed_forward_layer=-1,
            compiled_feed_forward=24,
            capacity=128,
            head_backend="shared-fcl-split",
        )
        board.save(device / "initialization.json", initialized)
        self.assertTrue(remote.numerical_acceptance(self.root, device.parent)["passed"])
        initialized["ffn_batch_tokens"] = 16
        board.save(device / "initialization.json", initialized)
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        initialized["ffn_batch_tokens"] = 8
        board.save(device / "initialization.json", initialized)
        options["state_commit"] = "ffn"
        board.save(self.root / "bundle.json", manifest)
        self.metadata["state_commit"] = "ffn"
        board.save(self.request / "request.json", self.metadata)
        (self.request / "state-commit.txt").write_text("ffn\n")
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        initialized["state_commit"] = "ffn"
        board.save(device / "initialization.json", initialized)
        self.assertTrue(remote.numerical_acceptance(self.root, device.parent)["passed"])
        (self.request / "state-commit.txt").write_text("combined\n")
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        options["state_commit"] = initialized["state_commit"] = "combined"
        del self.metadata["state_commit"]
        board.save(self.request / "request.json", self.metadata)
        board.save(self.root / "bundle.json", manifest)
        board.save(device / "initialization.json", initialized)
        layout_file = self.request / "decoder-layout.txt"
        for layout in ("fused-attention-norm",):
            options["decoder_layout"] = layout
            board.save(self.root / "bundle.json", manifest)
            self.metadata["decoder_layout"] = layout
            board.save(self.request / "request.json", self.metadata)
            layout_file.write_text(layout + "\n")
            self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
            initialized["decoder_layout"] = layout
            board.save(device / "initialization.json", initialized)
            self.assertTrue(remote.numerical_acceptance(self.root, device.parent)["passed"])
            layout_file.write_text("separate\n")
            self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
            layout_file.write_text(layout + "\n")
        options["decoder_layout"] = "unknown"
        board.save(self.root / "bundle.json", manifest)
        initialized["decoder_layout"] = "unknown"
        board.save(device / "initialization.json", initialized)
        self.metadata["decoder_layout"] = "unknown"
        board.save(self.request / "request.json", self.metadata)
        layout_file.write_text("unknown\n")
        self.assertFalse(remote.numerical_acceptance(self.root, device.parent)["passed"])
        options["decoder_layout"] = initialized["decoder_layout"] = "fused-attention-norm"
        self.metadata["decoder_layout"] = "fused-attention-norm"
        board.save(self.root / "bundle.json", manifest)
        board.save(device / "initialization.json", initialized)
        board.save(self.request / "request.json", self.metadata)
        layout_file.write_text("fused-attention-norm\n")
        board.save(device / "platform.json", profile | {"target": "ep"})
        result = remote.numerical_acceptance(self.root, device.parent)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["performance"])


if __name__ == "__main__":
    unittest.main()
