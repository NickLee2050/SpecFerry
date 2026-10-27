import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_models as downloader


class DownloadModelsTests(unittest.TestCase):
    def test_dry_run_selects_only_opt_tlm_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "not-created"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = downloader.main(["--dry-run", "--output", str(destination),
                                          "--parallel-models", "8"])
            plan = json.loads(stream.getvalue())
            self.assertEqual(result, 0)
            self.assertEqual([model["repo_id"] for model in plan["models"]], ["facebook/opt-6.7b"])
            self.assertEqual(plan["parallel_models"], 1)
            self.assertFalse(destination.exists())

    def test_excluded_and_unapproved_models_fail_before_network(self):
        for name in ("meta-llama/Llama-3.2-1B", "someone/Distill-LLaMA", "facebook/opt-125m"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                downloader.validate_models((downloader.ModelSpec(name),))

    def test_optional_validation_requires_explicit_selection_and_creates_no_files(self):
        for model in ('Qwen/Qwen3.5-27B', 'facebook/opt-30b'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as root:
                destination = Path(root) / 'not-created'
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    result = downloader.main(['--dry-run', '--model', model,
                                              '--output', str(destination)])
                self.assertEqual(result, 0)
                self.assertEqual([m['repo_id'] for m in json.loads(stream.getvalue())['models']],
                                 [model])
                self.assertFalse(destination.exists())

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
            return SimpleNamespace(sha=sha, siblings=[
                SimpleNamespace(rfilename=name, size=len(content),
                                lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()))
                for name, content in contents.items()
            ])

        def snapshot_download(**kwargs):
            download_requests.append(kwargs["revision"])
            destination = kwargs["local_dir"]
            if len(download_requests) == 1:
                (destination / "weights.safetensors.incomplete").write_bytes(b"partial")
                raise OSError("simulated interrupted transfer")
            for name, content in contents.items():
                (destination / name).write_bytes(content)

        fake_hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=model_info),
                                   snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {"huggingface_hub": fake_hub}), \
                patch.object(downloader.importlib.metadata, "version", return_value="test"):
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
            expected = [{"path": "weights.safetensors", "size": 4,
                         "sha256": hashlib.sha256(b"good").hexdigest()}]
            with self.assertRaisesRegex(ValueError, "SHA256"):
                downloader.verify_files(destination, expected)

    def test_parallel_shard_failure_cannot_publish_a_complete_manifest(self):
        contents = {
            'config.json': b'{"model_type":"qwen3_5"}',
            'tokenizer.json': b'{"version":"1.0"}',
            'part-1.safetensors': b'first weight',
            'part-2.safetensors': b'second weight',
        }
        info = SimpleNamespace(sha='b' * 40, siblings=[
            SimpleNamespace(rfilename=name, size=len(data),
                            lfs={'sha256': hashlib.sha256(data).hexdigest()})
            for name, data in contents.items()
        ])
        fail = True
        snapshots = []

        def transfer(url, destination, size, sha256, workers, **kwargs):
            if fail and destination.name == 'part-2.safetensors':
                raise OSError('one shard failed')
            destination.write_bytes(contents[destination.name])

        def snapshot(**kwargs):
            snapshots.append(kwargs['allow_patterns'])
            for name in kwargs['allow_patterns']:
                (kwargs['local_dir'] / name).write_bytes(contents[name])

        fake_hub = SimpleNamespace(
            HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: info),
            snapshot_download=snapshot, hf_hub_url=lambda *a, **kw: 'unused',
        )
        with tempfile.TemporaryDirectory() as root, patch.dict('sys.modules', {
                'huggingface_hub': fake_hub,
                'range_download': SimpleNamespace(download_ranges=transfer),
        }), patch.object(downloader.importlib.metadata, 'version', return_value='test'):
            model = downloader.ModelSpec('Qwen/Qwen3.5-27B')
            destination = Path(root) / model.repo_id
            with self.assertRaisesRegex(OSError, 'one shard failed'):
                downloader.download_one(model, Path(root), file_workers=2, range_workers=2)
            self.assertFalse((destination / 'download-manifest.json').exists())
            self.assertEqual(snapshots, [])
            fail = False
            downloader.download_one(model, Path(root), file_workers=2, range_workers=2)
            self.assertEqual(set(snapshots[0]), {'config.json', 'tokenizer.json'})
            manifest = json.loads((destination / 'download-manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual({item['path'] for item in manifest['files']}, set(contents))

    def test_opt_bin_weights_and_bpe_tokenizer_exclude_other_frameworks(self):
        names = ["config.json", "vocab.json", "merges.txt", "pytorch_model.bin.index.json",
                 "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
                 "flax_model.msgpack.index.json", "flax_model-00001-of-00002.msgpack",
                 "tf_model.h5.index.json", "tf_model-00001-of-00002.h5"]
        info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=name) for name in names])
        selected = {item["path"] for item in downloader.expected_files(info)}
        self.assertEqual(selected, set(names[:6]))

    def test_missing_bin_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root)
            name = "pytorch_model.bin.index.json"
            (destination / name).write_text('{"weight_map":{"w":"missing.bin"}}')
            with self.assertRaisesRegex(ValueError, "unselected shard"):
                downloader.verify_files(destination, [{"path": name, "size": None, "sha256": None}])


if __name__ == "__main__":
    unittest.main()
