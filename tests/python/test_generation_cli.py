"""Numerical regressions for the compact reporting and graph-cache checks."""

import contextlib
import io
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


from tests.python.validation_fixtures import GraphEvidenceFixture


class GenerationCommandTests(GraphEvidenceFixture):
    def test_mapped_slot_is_explicit_and_limited_to_fullgraph_c64(self):
        path = Path(__file__).resolve().parents[2] / "scripts/generate_opt.py"
        main = runpy.run_path(str(path))["main"]

        def prepare_request(*arguments, **kwargs):
            arguments[2].mkdir()
            return dict(
                capacity=64, selection={"mode": "greedy"}, tokens=[2], prompt="test"
            ), Mock()

        for kind, capacity, accepted in (
            ("fullgraph", 64, True),
            ("decoders", 64, False),
            ("fullgraph", 128, False),
        ):
            output = self.root / f"{kind}.{capacity}"
            arguments = [
                str(path),
                "--backend",
                "graph",
                "--compiled-model",
                str(self.root / "package"),
                "--capacity",
                str(capacity),
                "--kv-transfer",
                "mapped-slot",
                "--prepare-only",
                "--output",
                str(output),
            ]
            with (
                self.subTest(kind=kind, capacity=capacity),
                patch.object(sys, "argv", arguments),
                patch.dict(
                    main.__globals__, prepare_request=prepare_request, run_device=Mock()
                ) as globals_,
                patch(
                    "specferry.models.opt.compiled_model.load_model_package",
                    return_value={"kind": kind},
                ),
                patch("specferry.models.opt.graph_validation.prepare", return_value={}),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                if accepted:
                    self.assertEqual(main(), 0)
                    self.assertIn(
                        "kv_transfer mapped-slot\n",
                        (output / "request/run-options.txt").read_text(),
                    )
                else:
                    with self.assertRaises(SystemExit) as raised:
                        main()
                    self.assertEqual(raised.exception.code, 2)
                globals_["run_device"].assert_not_called()

    def test_compiled_prefill_defaults_to_packaged_nbg_projections_without_device_access(self):
        self.check_compiled_prefill_preparation(64)

    def test_compiled_prefill_accepts_declared_c256_without_device_access(self):
        self.check_compiled_prefill_preparation(256)

    def check_compiled_prefill_preparation(self, capacity):
        path = Path(__file__).resolve().parents[2] / "scripts/generate_opt.py"
        main = runpy.run_path(str(path))["main"]
        output, package = self.root / "prefill", self.root / "package"
        request = dict(
            capacity=capacity, selection={"mode": "greedy"}, tokens=[2] * 17, prompt="test"
        )

        def prepare_request(*arguments, **kwargs):
            arguments[2].mkdir()
            return request, Mock()

        arguments = [
            str(path),
            "--backend",
            "graph",
            "--compiled-ffn",
            str(package),
            "--capacity",
            str(capacity),
            "--block",
            "16",
            "--prefill-head",
            "last-token",
            "--prepare-only",
            "--output",
            str(output),
        ]
        with (
            patch.object(sys, "argv", arguments),
            patch.dict(main.__globals__, prepare_request=prepare_request),
            patch(
                "specferry.models.opt.compiled.load_ffn_package",
                return_value={
                    "prefill": {"tokens": 16, "supported_capacities": [64, 256]},
                    "projections": [{}] * 24,
                    "head": {"path": "head.nb"},
                },
            ),
            patch("specferry.models.opt.graph_validation.prepare", return_value={}),
            patch.dict(main.__globals__, run_device=Mock()) as globals_,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)
            globals_["run_device"].assert_not_called()
        options = dict(
            line.split() for line in (output / "request/run-options.txt").read_text().splitlines()
        )
        self.assertEqual(options["projection_backend"], "nbg")
        self.assertEqual(options["block"], "16")

    def test_prepare_and_benchmark_share_options_and_three_path_contract(self):
        path = Path(__file__).resolve().parents[2] / "scripts/generate_opt.py"
        main = runpy.run_path(str(path))["main"]
        request = dict(capacity=16, selection={"mode": "greedy"}, tokens=[2] * 5, prompt="test")
        metadata = {
            "layers": 2,
            "requests": [{"prompt_length": 5, "tokens": [1, 2, 3], "consumed": 7}],
        }
        generation = metadata["requests"][0] | dict(
            request_index=0,
            iteration=0,
            prefill_seconds=1,
            first_token_seconds=1,
            total_seconds=3,
            token_seconds=[1, 1],
        )
        tokenizer = Mock()
        tokenizer.decode.return_value = "test"

        def prepare_request(*arguments, **kwargs):
            arguments[2].mkdir()
            return request, tokenizer

        for prepare_only in (True, False):
            output = self.root / f"benchmark.{prepare_only}"
            arguments = [
                str(path),
                "--backend",
                "graph",
                "--block",
                "4",
                "--benchmark",
                "--warmups",
                "1",
                "--repeats",
                "3",
                "--max-new-tokens",
                "3",
                "--output",
                str(output),
            ]
            if prepare_only:
                arguments.append("--prepare-only")
            runner = Mock(return_value={})
            with (
                self.subTest(prepare_only=prepare_only),
                patch.object(sys, "argv", arguments),
                patch.dict(
                    main.__globals__,
                    prepare_request=prepare_request,
                    device_lock=Mock(return_value=contextlib.nullcontext()),
                    run_device=runner,
                    lifecycle=Mock(return_value=True),
                ),
                patch("specferry.models.opt.graph_validation.prepare", return_value=metadata),
                patch(
                    "specferry.models.opt.graph_validation.evaluate_benchmark",
                    return_value=({"checked": True}, [generation]),
                ) as evaluator,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(), 0)
            options = dict(
                line.split()
                for line in (output / "request/run-options.txt").read_text().splitlines()
            )
            expected = dict(block="4", benchmark="1", max_new_tokens="3", warmups="1", repeats="3")
            self.assertEqual({name: options[name] for name in expected}, expected)
            self.assertEqual(options["readback"], "none")
            if prepare_only:
                runner.assert_not_called()
                evaluator.assert_not_called()
            else:
                self.assertEqual(
                    runner.call_args.args[1],
                    [
                        Path(main.__globals__["ROOT"]) / ".cache/np101/opt-350m",
                        output / "request",
                        output / "device",
                    ],
                )
                self.assertEqual(evaluator.call_args.kwargs["repeats"], 3)
                self.assertTrue((output / "result.json").is_file())
