import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from range_download import download_ranges


class Response(io.BytesIO):
    def __init__(self, data, start, end, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


class RangeDownloadTests(unittest.TestCase):
    def test_resume_and_final_checksum(self):
        data = bytes(range(251)) * 3
        checksum = hashlib.sha256(data).hexdigest()
        requests = []

        def fetch(request, timeout):
            start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
            requests.append((start, end))
            return Response(data[start : end + 1], start, end, len(data))

        with tempfile.TemporaryDirectory() as directory, patch("range_download.urlopen", fetch):
            path = Path(directory) / "weight.safetensors"
            download_ranges("https://example.invalid/weight", path, len(data), checksum, 3, 100)
            self.assertEqual(path.read_bytes(), data)
            count = len(requests)
            path.unlink()
            download_ranges("https://example.invalid/weight", path, len(data), checksum, 3, 100)
            self.assertEqual(len(requests), count)
            self.assertEqual(path.read_bytes(), data)

    def test_ignored_range_and_same_size_corruption_never_publish(self):
        for status, payload in ((200, b"good"), (206, b"evil")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weight.safetensors"
                with (
                    patch(
                        "range_download.urlopen",
                        lambda *a, **kw: Response(payload, 0, 3, 4, status),
                    ),
                    patch("range_download.time.sleep"),
                ):
                    with self.assertRaises(ValueError):
                        download_ranges(
                            "https://example.invalid/weight",
                            path,
                            4,
                            hashlib.sha256(b"good").hexdigest(),
                            1,
                            4,
                        )
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
