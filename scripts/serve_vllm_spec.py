#!/usr/bin/env python3
"""Start a loopback vLLM OPT server using the external HTTP draft proposer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:89154ef00dd15368d2b293c167e5cc7dbb521fcfb2fbb77510e0d4df2b820e8f'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dlm-url', help='SpecFerry greedy draft callback; not an NP101 SDK endpoint')
    p.add_argument('--draft-source', choices=('external', 'disabled'), default='external',
                   help='disabled supplies empty drafts without HTTP for the matching target-only baseline')
    p.add_argument('--output', type=Path, required=True, help='New directory under .cache/runs')
    p.add_argument('--port', type=int, default=8011)
    p.add_argument('--name', default='specferry-tlm-spec')
    p.add_argument('--draft-tokens', type=int, default=4)
    p.add_argument('--timeout', type=float, default=30)
    p.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    p.add_argument('--checkpoint', choices=('facebook/opt-6.7b', 'facebook/opt-30b'),
                   default='facebook/opt-6.7b')
    p.add_argument('--profile-rounds', action='store_true',
                   help='Local experiment adapter; requires a literal loopback fixture URL')
    args = p.parse_args()
    if args.checkpoint == 'facebook/opt-30b' and args.dtype != 'float16':
        p.error('OPT-30B is authorized as an FP16 target; FP32 exceeds this test memory budget')
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    if not (1 <= args.draft_tokens <= 32 and 1024 <= args.port <= 65535):
        p.error('Require draft-tokens in 1..32 and port in 1024..65535')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('Timeout must be positive and finite')
    if args.draft_source == 'external':
        parsed = urlsplit(args.dlm_url or '')
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username:
            p.error('External drafts require an HTTP(S) DLM URL without embedded credentials')
    elif args.dlm_url:
        p.error('Do not supply --dlm-url when drafts are disabled')
    if args.profile_rounds and (args.draft_source != 'external' or
                               urlsplit(args.dlm_url).hostname not in ('127.0.0.1', '::1')):
        p.error('Round profiling requires an external literal-loopback fixture')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / '.cache' / 'runs') or output.exists():
        p.error('Choose a new directory under project .cache/runs')
    model = ROOT / '.cache/models' / args.checkpoint
    manifest = json.loads((model / 'download-manifest.json').read_text())
    if manifest['repo_id'] != args.checkpoint or manifest['status'] != 'complete':
        p.error('Require the verified authorized OPT checkpoint')
    library = ROOT / 'build/draft/libspecferry_draft.so'
    if not library.is_file():
        p.error('Build native/draft first; see README')
    output.mkdir(parents=True)
    shutil.copy2(library, output / library.name)
    hashes = {}
    hashes[library.name] = hashlib.sha256(library.read_bytes()).hexdigest()
    for relative in ('python/specferry/__init__.py', 'native/draft/client.cpp', 'native/draft/client.hpp',
                     'native/draft/CMakeLists.txt',
                     'python/specferry/vllm_proposer.py', 'scripts/serve_vllm_spec.py',
                     'python/specferry/vllm_profile.py',
                     'protocol/draft-v1.schema.json',
                     'requirements-tlm-runtime.txt'):
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    conda_root = Path(sys.prefix).parent.parent
    cmd = ['docker', 'run', '-d', '--init', '--pull', 'never', '--name', args.name,
           '--network', 'host', '--gpus', 'all', '--user', f'{os.getuid()}:{os.getgid()}',
           '--shm-size', '2g',
           '--mount', f'type=bind,source={conda_root},target={conda_root},readonly',
           '--mount', f'type=bind,source={model},target=/models/tlm,readonly',
           '--mount', f'type=bind,source={output},target=/results',
           '-e', 'PYTHONPATH=/results/source/python:/usr/local/lib/python3.12/dist-packages',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
           '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_HOME=/results/runtime-cache/huggingface',
           '-e', 'XDG_CACHE_HOME=/results/runtime-cache',
           '-e', 'XDG_CONFIG_HOME=/results/runtime-cache/config',
           '-e', 'VLLM_NO_USAGE_STATS=1',
           '-e', 'FLASHINFER_WORKSPACE_BASE=/results/runtime-cache/flashinfer',
           '-e', 'TORCHINDUCTOR_CACHE_DIR=/results/runtime-cache/torchinductor',
           '-e', 'TRITON_CACHE_DIR=/results/runtime-cache/triton',
           '-e', f'SPECFERRY_DRAFT_SOURCE={args.draft_source}',
           '-e', f'SPECFERRY_DLM_TIMEOUT={args.timeout}',
           '-e', 'SPECFERRY_DRAFT_LIBRARY=/results/libspecferry_draft.so',
           '-e', 'SPECFERRY_DLM_AUDIT=/results/proposer.jsonl']
    if args.draft_source == 'external':
        cmd += ['-e', f'SPECFERRY_DLM_URL={args.dlm_url}']
    if args.draft_source == 'external' and os.environ.get('SPECFERRY_DLM_TOKEN'):
        cmd += ['-e', 'SPECFERRY_DLM_TOKEN']
    proposer = 'specferry.vllm_proposer.ExternalDraftProposer'
    if args.profile_rounds:
        proposer = 'specferry.vllm_profile.ProfiledDraftProposer'
        (output / 'profile-control.json').write_text(json.dumps(
            {'run_id': 'startup', 'source': 'disabled', 'draft_tokens': args.draft_tokens})+'\n')
        cmd += ['-e', 'SPECFERRY_PROFILE_CONTROL=/results/profile-control.json',
                '-e', 'SPECFERRY_PROFILE_AUDIT=/results/rounds.jsonl']
    spec = {'method': 'custom_class', 'model': 'specferry.vllm_proposer.ExternalDraftProposer',
            'num_speculative_tokens': args.draft_tokens}
    spec['model'] = proposer
    kv_bytes = (4 if args.checkpoint == 'facebook/opt-30b' else 2) * 2**30
    cmd += ['--entrypoint', sys.executable, IMAGE, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', '/models/tlm', '--served-model-name', 'specferry-tlm-spec',
            '--host', '127.0.0.1', '--port', str(args.port), '--dtype', args.dtype,
            '--tensor-parallel-size', '1', '--max-model-len', '2048', '--max-num-seqs', '1',
            '--max-num-batched-tokens', '2048', '--kv-cache-memory-bytes', str(kv_bytes),
            '--gpu-memory-utilization', '0.3', '--enforce-eager', '--no-async-scheduling',
            '--no-enable-prefix-caching',
            '--generation-config', 'vllm', '--speculative-config', json.dumps(spec)]
    launch = ['sg', 'docker', '-c', shlex.join(cmd)]
    result = subprocess.run(launch, capture_output=True, text=True)
    record = {'started_at': datetime.now(timezone.utc).isoformat(), 'command': launch,
              'source_sha256': hashes, 'image_id': IMAGE, 'model_revision': manifest['resolved_revision'],
              'exit_code': result.returncode, 'container_id': result.stdout.strip(),
              'stderr': result.stderr, 'service_url': f'http://127.0.0.1:{args.port}',
              'conda_prefix': sys.prefix, 'vllm_version': '0.28.0',
              'dtype': args.dtype,
              'checkpoint': args.checkpoint, 'kv_cache_bytes': kv_bytes,
              'profile_rounds': args.profile_rounds,
              'draft_source': args.draft_source,
              'scope': 'NP101 execution is not inferred from starting a service; verify device evidence separately.'}
    (output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(result.stdout, end='')
    print(result.stderr, file=sys.stderr, end='')
    print(f'Evidence: {output}')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
