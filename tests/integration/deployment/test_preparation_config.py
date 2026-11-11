"""Offline equivalence and rejection checks for JSON preparation inputs."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.deploy.execute.np101.soc.cli import main
from tools.deploy.pack.np101.config import QwenExecutionOptions, read_preparation_config


class PreparationConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / "prepare.json"
        self.sources = {name: self.root / name for name in ("package", "request", "sdk")}
        self.options = {
            "projections": "shared-fcl",
            "kv_write_mode": "mapped",
            "ffn_batch_tokens": 8,
            "decoder_layout": "fused-attention-norm",
            "state_commit": "ffn",
        }

    def write_config(self, family="qwen3_5", **fields):
        config = {
            "schema_version": 1,
            "model_family": family,
            **{key: path.name for key, path in self.sources.items()},
            **fields,
        }
        self.config.write_text(json.dumps(config))

    def invoke(self, args):
        with (
            patch.object(sys, "argv", ["soc", "prepare", *args]),
            patch("tools.deploy.pack.np101.qwen.prepare_bundle") as qwen,
            patch("tools.deploy.pack.np101.soc.prepare_bundle") as opt,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for prepare in (qwen, opt):
                prepare.return_value = {"payload_bytes": 0, "files": {}}
            result = main()
            self.assertEqual(result, 0)
            return qwen.call_args, opt.call_args

    def source_args(self):
        return [item for key, path in self.sources.items() for item in ("--" + key, str(path))]

    def test_qwen_json_matches_existing_explicit_cli(self):
        output = ["--output", str(self.root / "out")]
        explicit, unused = self.invoke(
            [
                "--model",
                "qwen",
                *self.source_args(),
                *output,
                "--qwen-kv-write-mode",
                "mapped",
                "--qwen-ffn-batch",
                "8",
                "--qwen-decoder-layout",
                "fused-attention-norm",
                "--qwen-state-commit",
                "ffn",
            ]
        )
        self.assertIsNone(unused)
        self.write_config(request_options=self.options)
        configured, unused = self.invoke(["--config", str(self.config), *output])
        self.assertEqual(configured, explicit)
        self.assertEqual(configured.kwargs, self.options)
        self.assertIsNone(unused)

    def test_cli_override_and_qwen_defaults(self):
        output = ["--output", str(self.root / "out")]
        original, _ = self.invoke(["--model", "qwen", *self.source_args(), *output])
        self.assertEqual(original.kwargs, QwenExecutionOptions().as_mapping())
        self.write_config(request_options=self.options)
        replacement = self.root / "other-package"
        configured, _ = self.invoke(
            [
                "--config",
                str(self.config),
                *output,
                "--package",
                str(replacement),
                "--qwen-ffn-batch",
                "16",
            ]
        )
        self.assertEqual(configured.args[2], replacement)
        self.assertEqual(configured.kwargs, self.options | {"ffn_batch_tokens": 16})

    def test_opt_json_preserves_request_owned_options_and_prefill_path(self):
        output = ["--output", str(self.root / "out")]
        prefill = self.root / "prefill"
        unused, explicit = self.invoke(
            [*self.source_args(), *output, "--prefill-package", str(prefill)]
        )
        self.assertIsNone(unused)
        self.write_config("opt", prefill_package="prefill")
        unused, configured = self.invoke(["--config", str(self.config), *output])
        self.assertIsNone(unused)
        self.assertEqual(configured, explicit)

    def test_reject_invalid_config_before_packaging(self):
        for fields in (
            {"schema_version": 2},
            {"schema_version": True},
            {"driver_parameters": {}},
            {"request_options": {"ffn_batch_tokens": True}},
            {"request_options": {"ffn_batch_tokens": "8"}},
            {"request_options": {"projections": None}},
            {"request_options": {"unknown": 1}},
            {"request_options": {"ffn_batch_tokens": 8}},
            {"sdk": ""},
            {"model_family": "unknown"},
        ):
            with self.subTest(fields=fields):
                self.write_config(**fields)
                with self.assertRaises(ValueError):
                    read_preparation_config(self.config)
        self.write_config("opt", request_options=self.options)
        with self.assertRaisesRegex(ValueError, "OPT execution options"):
            read_preparation_config(self.config)

    def test_duplicate_keys_are_not_silently_overwritten(self):
        self.config.write_text(
            '{"schema_version":1,"model_family":"qwen3_5",'
            '"request_options":{"ffn_batch_tokens":1,"ffn_batch_tokens":8}}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate configuration key"):
            read_preparation_config(self.config)

    def test_wrong_model_missing_input_and_invalid_override_do_not_package(self):
        self.write_config(request_options=self.options)
        output = ["--output", str(self.root / "out")]
        for arguments in (
            ["--config", str(self.config), "--model", "opt", *output],
            ["--config", str(self.config), "--qwen-ffn-batch", "1", *output],
            ["--model", "opt", "--qwen-state-commit", "ffn", *self.source_args(), *output],
            output,
        ):
            with (
                self.subTest(arguments=arguments),
                patch.object(sys, "argv", ["soc", "prepare", *arguments]),
                patch("tools.deploy.pack.np101.soc.prepare_bundle") as opt,
                patch("tools.deploy.pack.np101.qwen.prepare_bundle") as qwen,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                opt.assert_not_called()
                qwen.assert_not_called()
