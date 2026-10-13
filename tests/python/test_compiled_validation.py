"""Compiled acceptance must reject changed artifacts, bad state and inflated throughput."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from specferry.data.checkpoint import sha256
from specferry.validation import compiled


class CompiledValidationTests(unittest.TestCase):
    def test_plan_pins_independent_reference_and_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reference.json").write_text(json.dumps({"requests": [{"tokens": [1]}]}))
            np.array([1], dtype="<f2").tofile(root / "expected.0.logits.bin")
            artifact = root / "network.nb"
            artifact.write_bytes(b"independent artifact")
            files = {str(path): sha256(path) for path in root.iterdir()}
            plan = dict(
                schema_version=1,
                sdk={},
                file_sets={"artifacts": files},
                cases={
                    "model": dict(
                        kind="fullgraph",
                        binary=sys.executable,
                        request=str(root),
                        repeats=1,
                        stop_reason="max_new_tokens",
                        payload_bound_bytes=dict(const=100, nonconst=100),
                        file_sets=["artifacts"],
                    )
                },
            )
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with (
                patch.object(compiled, "snapshot") as kernel,
                patch.object(compiled, "run_device") as device,
            ):
                compiled.load_case(path, "model")
                kernel.assert_not_called()
                device.assert_not_called()
            artifact.write_bytes(b"changed artifact")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                compiled.load_case(path, "model")
            artifact.write_bytes(b"independent artifact")
            del files[str(root / "reference.json")]
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "references require pinned"):
                compiled.load_case(path, "model")

    def test_generation_rate_excludes_warmup_and_keeps_actual_output_counts(self):
        result = compiled.summarize(
            [
                dict(
                    tokens=[1] * 100, total_seconds=100, first_token_seconds=99, token_seconds=[1]
                ),
                dict(
                    tokens=[1, 2, 3],
                    total_seconds=1.2,
                    first_token_seconds=1,
                    token_seconds=[0.1, 0.1],
                ),
                dict(tokens=[1, 2], total_seconds=1.1, first_token_seconds=1, token_seconds=[0.1]),
            ]
        )
        self.assertEqual(result["generated_tokens"], 5)
        self.assertAlmostEqual(result["output_tokens_per_second"], 5 / 2.3)
        self.assertAlmostEqual(result["decode_tokens_per_second"], 10)
        self.assertAlmostEqual(result["decode_ms"], 100)

    def test_tokens_alone_cannot_accept_bad_warmup_cache_or_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device = root / "device"
            device.mkdir()
            generation = dict(
                tokens=[1, 2],
                consumed=2,
                stop_reason="max_new_tokens",
                total_seconds=2,
                first_token_seconds=1,
                token_seconds=[1],
            )
            case = dict(
                kind="fullgraph",
                repeats=1,
                request=root,
                reference=generation,
                stop_reason="max_new_tokens",
                cache=dict(
                    dtype="F32",
                    shape=[1, 4, 2],
                    keys=str(root / "keys.bin"),
                    values=str(root / "values.bin"),
                ),
            )
            for name in ("warmup", "measured"):
                (device / (name + ".json")).write_text(json.dumps(generation))
            logits = np.array([1, 2], dtype="<f2")
            logits.tofile(root / "expected.0.logits.bin")
            logits.tofile(device / "logits.bin")
            cache = np.zeros((1, 4, 2), dtype="<f4")
            cache[:, :2] = 1
            for name in ("key", "value"):
                cache.tofile(root / ("keys.bin" if name == "key" else "values.bin"))
                cache.tofile(device / f"layer23.present_{name}.bin")
            native = dict(
                released=True,
                const_live_bytes=0,
                nonconst_live_bytes=0,
                const_peak_bytes=100,
                nonconst_peak_bytes=100,
            )
            (device / "execution.json").write_text(json.dumps(native))
            self.assertTrue(compiled.evaluate(case, device)["passed"])
            cache[:, 3] = -0.0
            cache.tofile(device / "layer23.present_key.bin")
            self.assertFalse(compiled.evaluate(case, device)["passed"])
            cache[:, 3] = 0.0
            cache.tofile(device / "layer23.present_key.bin")
            (device / "warmup.json").write_text(json.dumps(generation | {"tokens": [9]}))
            self.assertIsNone(compiled.evaluate(case, device)["performance"])
            (device / "warmup.json").write_text(json.dumps(generation))
            (device / "execution.json").write_text(json.dumps(native | {"nonconst_live_bytes": 1}))
            self.assertFalse(compiled.evaluate(case, device)["passed"])
