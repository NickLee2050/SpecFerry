"""Independent protocol fixtures for integration-test scripts, not runtime logic."""
import hashlib
import json
from pathlib import Path

PROTOCOL = 'specferry.draft.v1'


def prefix_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def tokenizer_digest(directory):
    digest = hashlib.sha256()
    for name in ('vocab.json', 'merges.txt'):
        data = (Path(directory) / name).read_bytes()
        digest.update(name.encode() + b'\0' + len(data).to_bytes(8, 'big') + data)
    return digest.hexdigest()
