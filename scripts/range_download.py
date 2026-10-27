"""Bounded, resumable HTTP range transfers for public, SHA256-addressed weights."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from threading import Event, Lock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen


class DownloadRateLimited(RuntimeError):
    """Stop queued requests rather than amplify an upstream HTTP 429."""


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_ranges(url: str, destination: Path, size: int, sha256: str,
                    workers: int, chunk_size: int = 8 * 1024 * 1024,
                    *, resolve_redirects: bool = False):
    if size <= 0 or workers < 1 or chunk_size < 1:
        raise ValueError('positive size, worker count and chunk size required')
    if destination.is_file() and destination.stat().st_size == size and digest_file(destination) == sha256:
        return
    parts = destination.parent / '.cache' / 'specferry-ranges' / sha256
    parts.mkdir(parents=True, exist_ok=True)
    location_lock = Lock()
    location = None
    expires = 0
    limited = Event()

    def source_url(failed_url=None):
        nonlocal location, expires
        if not resolve_redirects:
            return url
        with location_lock:
            if location and ((failed_url and failed_url != location) or
                             (failed_url is None and time.time() < expires - 60)):
                return location
            # Resolve without Range so the signed URL is valid for any byte range.
            # Close the body immediately; this does not download the whole object.
            request = Request(url, headers={'Accept-Encoding':'identity', 'Cache-Control':'no-cache'})
            with urlopen(request, timeout=45) as response:
                resolved = response.geturl()
            if urlsplit(resolved).scheme != 'https':
                raise ValueError('Require HTTPS for resolved weight locations')
            query = parse_qs(urlsplit(resolved).query)
            location = resolved
            try:
                expires = float(query['Expires'][0])
            except (KeyError, ValueError):
                expires = time.time() + 1800
            return location

    def transfer(start):
        if limited.is_set():
            raise DownloadRateLimited('Paused after HTTP 429')
        end = min(start + chunk_size, size) - 1
        part = parts / f'{start}-{end}.part'
        stamp = part.with_suffix('.json')
        if part.exists() and stamp.exists() and part.stat().st_size == end - start + 1:
            if digest_file(part) == json.loads(stamp.read_text())['sha256']:
                return
        temporary = part.with_suffix('.tmp')
        for attempt in range(5):
            if limited.is_set():
                raise DownloadRateLimited('Paused after HTTP 429')
            effective_url = None
            try:
                effective_url = source_url()
                request = Request(effective_url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                started = time.monotonic()
                with urlopen(request, timeout=45) as response, temporary.open('wb') as stream:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise ValueError(f'server did not honor byte range {start}-{end}')
                    remaining = end - start + 1
                    digest = hashlib.sha256()
                    while remaining:
                        if time.monotonic() - started > 120:
                            raise TimeoutError(f'range deadline exceeded: {start}-{end}')
                        # read1 returns available bytes instead of waiting for a full
                        # megabyte while a server dribbles data indefinitely.
                        reader = getattr(response, 'read1', response.read)
                        block = reader(min(1024 * 1024, remaining))
                        if not block:
                            raise OSError(f'truncated range {start}-{end}')
                        stream.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise ValueError('range response exceeded its declared size')
                temporary.replace(part)
                stamp.write_text(json.dumps({'sha256': digest.hexdigest()}))
                return
            except (OSError, ValueError) as error:
                temporary.unlink(missing_ok=True)
                if isinstance(error, HTTPError) and error.code == 429:
                    limited.set()
                    raise DownloadRateLimited(
                        f'HTTP 429; Retry-After={error.headers.get("Retry-After", "unspecified")}') from error
                if isinstance(error, HTTPError) and error.code == 403 and resolve_redirects:
                    source_url(failed_url=effective_url)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    starts = list(range(0, size, chunk_size))
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(transfer, start) for start in starts]
        for count, future in enumerate(as_completed(pending), 1):
            try:
                future.result()
            except DownloadRateLimited:
                limited.set()
                for queued in pending:
                    queued.cancel()
                raise
            except Exception as error:
                failures.append(error)
                print(f'{destination.name}: range failed: {error}', flush=True)
            if count % 8 == 0 or count == len(starts):
                print(f'{destination.name}: verified ranges {count-len(failures)}/{len(starts)}'
                      f'; failed {len(failures)}', flush=True)
    if failures:
        raise failures[0]
    assembled = destination.with_suffix(destination.suffix + '.assembling')
    try:
        with assembled.open('wb') as output:
            for start in starts:
                end = min(start + chunk_size, size) - 1
                with (parts / f'{start}-{end}.part').open('rb') as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        output.write(block)
        if assembled.stat().st_size != size or digest_file(assembled) != sha256:
            raise ValueError(f'assembled weight SHA256 mismatch: {destination.name}')
        assembled.replace(destination)
    finally:
        assembled.unlink(missing_ok=True)
