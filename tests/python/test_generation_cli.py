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
