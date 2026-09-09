"""Validate the downloaded checkpoint before loading its text-only model."""
import hashlib
import json
import math
from pathlib import Path
import struct

MODEL_ID = 'Qwen/Qwen3.5-0.8B'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
DTYPE_BYTES = {'F32': 4, 'BF16': 2, 'F16': 2, 'I64': 8, 'I32': 4, 'I8': 1, 'U8': 1, 'BOOL': 1}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def local_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'unsafe checkpoint path: {name}')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'checkpoint path escapes model directory: {name}')
    return path


def inventory(root: Path):
    manifest = json.loads((root / 'download-manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('repo_id') != MODEL_ID
            or manifest.get('resolved_revision') != REVISION):
        raise ValueError('a complete download manifest for the fixed first DLM revision is required')
    files = manifest['files']
    for entry in files:
        path = local_path(root, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError(f'missing or wrong-size checkpoint file: {path}')
        if entry.get('sha256') and sha256(path) != entry['sha256']:
            raise ValueError(f'checkpoint SHA256 mismatch: {path}')
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    selected = {entry['path'] for entry in files}
    if not set(index['weight_map'].values()) <= selected:
        raise ValueError('index references an unverified weight file')
    tensors = []
    groups = {group: {'tensors': 0, 'parameters': 0, 'bytes': 0} for group in ('text', 'vision', 'mtp')}
    names = set()
    for filename in sorted(set(index['weight_map'].values())):
        path = local_path(root, filename)
        with path.open('rb') as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError('truncated safetensors header')
            header_size = struct.unpack('<Q', size_bytes)[0]
            if not 2 <= header_size <= min(100 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError('invalid safetensors header length')
            header = json.loads(stream.read(header_size))
        intervals = []
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            if name in names or index['weight_map'].get(name) != filename:
                raise ValueError(f'duplicate tensor or index/header disagreement: {name}')
            names.add(name)
            shape = meta['shape']
            if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
                raise ValueError(f'invalid shape: {name}')
            count = math.prod(shape)
            size = count * DTYPE_BYTES[meta['dtype']]
            start, end = meta['data_offsets']
            if start < 0 or end - start != size or end > path.stat().st_size - 8 - header_size:
                raise ValueError(f'invalid tensor offsets: {name}')
            intervals.append((start, end))
            if name.startswith('model.language_model.'):
                group, mapped = 'text', 'model.' + name.removeprefix('model.language_model.')
            elif name.startswith('model.visual.'):
                group, mapped = 'vision', None
            elif name.startswith('mtp.'):
                group, mapped = 'mtp', None
            else:
                raise ValueError(f'unclassified checkpoint tensor: {name}')
            groups[group]['tensors'] += 1
            groups[group]['parameters'] += count
            groups[group]['bytes'] += size
            tensors.append({'source_name': name, 'text_name': mapped, 'group': group,
                            'file': filename, 'shape': shape, 'dtype': meta['dtype'],
                            'parameters': count, 'bytes': size, 'data_offsets': [start, end]})
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError('overlapping or non-contiguous safetensors payload')
            cursor = end
        if cursor + 8 + header_size != path.stat().st_size:
            raise ValueError('unaccounted bytes in safetensors file')
    if names != set(index['weight_map']):
        raise ValueError('some indexed tensors are missing from weight headers')
    if sum(group['bytes'] for group in groups.values()) != index['metadata']['total_size']:
        raise ValueError('index total_size does not match tensor payloads')
    config = json.loads((root / 'config.json').read_text())
    return {'status': 'verified', 'repo_id': MODEL_ID, 'revision': REVISION,
            'download_manifest_sha256': sha256(root / 'download-manifest.json'),
            'groups': groups, 'text_config': config['text_config'],
            'tied_embedding_head': config['text_config'].get('tie_word_embeddings', config.get('tie_word_embeddings')),
            'tensors': sorted(tensors, key=lambda item: item['source_name'])}
