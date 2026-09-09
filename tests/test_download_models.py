import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import download_models as downloader


class DownloadModelsTests(unittest.TestCase):
    def test_dry_run_selects_only_dlm_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "not-created"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = downloader.main(
                    ["--dry-run", "--output", str(destination), "--parallel-models", "8"]
                )
            plan = json.loads(stream.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual([model["repo_id"] for model in plan["models"]], ["Qwen/Qwen3.5-0.8B"])
            self.assertEqual(plan["parallel_models"], 1)
            self.assertFalse(destination.exists())

    def test_excluded_and_unapproved_models_fail_before_network(self):
        for name in ("meta-llama/Llama-3.2-1B", "someone/Distill-LLaMA", "facebook/opt-125m"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                downloader.validate_models((downloader.ModelSpec(name),))

    def test_interrupted_download_resumes_the_same_commit(self):
        contents = {
            "config.json": b'{"model_type":"qwen3_5"}',
            "tokenizer.json": b'{"version":"1.0"}',
            "weights.safetensors": b"test weight bytes",
            "model.safetensors.index.json": b'{"weight_map":{"w":"weights.safetensors"}}',
        }
        sha = "a" * 40
        metadata_requests = []
        download_requests = []

        def model_info(repo_id, revision, files_metadata):
            metadata_requests.append(revision)
            return SimpleNamespace(
                sha=sha,
                siblings=[
                    SimpleNamespace(
                        rfilename=name,
                        size=len(content),
                        lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()),
                    )
                    for name, content in contents.items()
                ],
            )

        def snapshot_download(**kwargs):
            download_requests.append(kwargs["revision"])
            destination = kwargs["local_dir"]
            if len(download_requests) == 1:
                (destination / "weights.safetensors.incomplete").write_bytes(b"partial")
                raise OSError("simulated interrupted transfer")
            for name, content in contents.items():
                (destination / name).write_bytes(content)

        fake_hub = SimpleNamespace(
            HfApi=lambda: SimpleNamespace(model_info=model_info),
            snapshot_download=snapshot_download,
        )
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict("sys.modules", {"huggingface_hub": fake_hub}),
            patch.object(downloader.importlib.metadata, "version", return_value="test"),
        ):
            model = downloader.ModelSpec("Qwen/Qwen3.5-0.8B")
            destination = Path(root) / model.repo_id
            with self.assertRaises(OSError):
                downloader.download_one(model, Path(root), 2)
            self.assertTrue((destination / "download-lock.json").is_file())
            self.assertFalse((destination / "download-manifest.json").exists())
            downloader.download_one(model, Path(root), 2)
            self.assertEqual(metadata_requests, ["main", sha])
            self.assertEqual(download_requests, [sha, sha])
            manifest = json.loads((destination / "download-manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["resolved_revision"], sha)

    def test_same_size_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root)
            (destination / "weights.safetensors").write_bytes(b"bad!")
            expected = [
                {
                    "path": "weights.safetensors",
                    "size": 4,
                    "sha256": hashlib.sha256(b"good").hexdigest(),
                }
            ]
            with self.assertRaisesRegex(ValueError, "SHA256"):
                downloader.verify_files(destination, expected)


if __name__ == "__main__":
    unittest.main()
