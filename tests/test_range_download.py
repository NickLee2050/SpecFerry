import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from range_download import DownloadRateLimited, download_ranges


class Response(io.BytesIO):
    def __init__(self, data, start, end, total, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes {start}-{end}/{total}'}


class RangeDownloadTests(unittest.TestCase):
    def test_resolved_url_is_reused_refreshed_and_not_needed_for_cached_parts(self):
        data = bytes(range(251)) * 3
        checksum = hashlib.sha256(data).hexdigest()
        resolutions = []
        transfers = []

        def fetch(request, timeout):
            byte_range = request.get_header('Range')
            if byte_range is None:
                resolutions.append(request.full_url)
                url = f'https://cdn.example.invalid/weight?v={len(resolutions)}&Expires=9999999999'
                response = Response(b'', 0, 0, len(data), 200)
                response.geturl = lambda: url
                return response
            transfers.append(request.full_url)
            if len(transfers) == 1:
                raise HTTPError(request.full_url, 403, 'expired', {}, None)
            start, end = map(int, byte_range.removeprefix('bytes=').split('-'))
            self.assertTrue(request.full_url.startswith('https://cdn.example.invalid/'))
            return Response(data[start:end+1], start, end, len(data))

        with tempfile.TemporaryDirectory() as directory, patch('range_download.urlopen', fetch), \
                patch('range_download.time.sleep'):
            path = Path(directory)/'weight.safetensors'
            download_ranges('https://origin.example.invalid/weight', path, len(data),
                            checksum, 1, 100, resolve_redirects=True)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(len(resolutions), 2)
            self.assertTrue(all('v=2&' in url for url in transfers[1:]))
            count = len(transfers)
            path.unlink()
            download_ranges('https://origin.example.invalid/weight', path, len(data),
                            checksum, 2, 100, resolve_redirects=True)
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(len(resolutions), 2)
            self.assertEqual(len(transfers), count)

    def test_rate_limit_stops_queued_requests_without_publishing(self):
        requests = []

        def limited(request, timeout):
            requests.append(request.full_url)
            raise HTTPError(request.full_url, 429, 'limit', {'Retry-After':'60'}, None)

        with tempfile.TemporaryDirectory() as directory, patch('range_download.urlopen', limited):
            path = Path(directory)/'weight.safetensors'
            with self.assertRaises(DownloadRateLimited):
                download_ranges('https://example.invalid/weight', path, 400,
                                hashlib.sha256(b'x'*400).hexdigest(), 2, 4)
            self.assertFalse(path.exists())
            self.assertLessEqual(len(requests), 2)

    def test_resume_and_final_checksum(self):
        data = bytes(range(251)) * 3
        checksum = hashlib.sha256(data).hexdigest()
        requests = []
        def fetch(request, timeout):
            start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
            requests.append((start, end))
            return Response(data[start:end + 1], start, end, len(data))
        with tempfile.TemporaryDirectory() as directory, patch('range_download.urlopen', fetch):
            path = Path(directory) / 'weight.safetensors'
            download_ranges('https://example.invalid/weight', path, len(data), checksum, 3, 100)
            self.assertEqual(path.read_bytes(), data)
            count = len(requests)
            path.unlink()
            download_ranges('https://example.invalid/weight', path, len(data), checksum, 3, 100)
            self.assertEqual(len(requests), count)
            self.assertEqual(path.read_bytes(), data)

    def test_ignored_range_and_same_size_corruption_never_publish(self):
        for status, payload in ((200, b'good'), (206, b'evil')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'weight.safetensors'
                with patch('range_download.urlopen', lambda *a, **kw: Response(payload, 0, 3, 4, status)), \
                     patch('range_download.time.sleep'):
                    with self.assertRaises(ValueError):
                        download_ranges('https://example.invalid/weight', path, 4,
                                        hashlib.sha256(b'good').hexdigest(), 1, 4)
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
