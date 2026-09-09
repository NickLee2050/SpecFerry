"""Bounded, resumable HTTP range transfers for public, SHA256-addressed weights."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_ranges(url: str, destination: Path, size: int, sha256: str,
                    workers: int, chunk_size: int = 8 * 1024 * 1024):
    if size <= 0 or workers < 1 or chunk_size < 1:
        raise ValueError('positive size, worker count and chunk size required')
    if destination.is_file() and destination.stat().st_size == size and digest_file(destination) == sha256:
        return
    parts = destination.parent / '.cache' / 'specferry-ranges' / sha256
    parts.mkdir(parents=True, exist_ok=True)

    def transfer(start):
        end = min(start + chunk_size, size) - 1
        part = parts / f'{start}-{end}.part'
        stamp = part.with_suffix('.json')
        if part.exists() and stamp.exists() and part.stat().st_size == end - start + 1:
            if digest_file(part) == json.loads(stamp.read_text())['sha256']:
                return
        temporary = part.with_suffix('.tmp')
        for attempt in range(5):
            try:
                request = Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
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
            except (OSError, ValueError):
                temporary.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    starts = list(range(0, size, chunk_size))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(transfer, start) for start in starts]
        for count, future in enumerate(as_completed(pending), 1):
            future.result()
            if count % 8 == 0 or count == len(starts):
                print(f'{destination.name}: verified ranges {count}/{len(starts)}', flush=True)
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
