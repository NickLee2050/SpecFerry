#!/usr/bin/env python3
"""Launch the authorized Qwen3.5-27B standalone text validation on the DGX."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:89154ef00dd15368d2b293c167e5cc7dbb521fcfb2fbb77510e0d4df2b820e8f'
CHECKPOINT = 'Qwen/Qwen3.5-27B'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', default='specferry-tlm-qwen')
    p.add_argument('--port', type=int, default=8020)
    p.add_argument('--enforce-eager', action='store_true', help='Diagnostic fallback; recorded separately')
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    if not 1024 <= args.port <= 65535:
        p.error('Require an unprivileged port')
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.cache/runs'):
        p.error('Choose a new directory under .cache/runs')
    model = ROOT/'.cache/models'/CHECKPOINT
    manifest = json.loads((model/'download-manifest.json').read_text())
    if manifest['repo_id'] != CHECKPOINT or manifest['status'] != 'complete':
        p.error('Require the verified authorized Qwen checkpoint')
    out.mkdir(parents=True)
    hashes = {}
    for relative in ('scripts/serve_qwen_tlm.py', 'requirements-tlm-runtime.txt'):
        destination = out/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, destination)
        hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    conda = Path(sys.prefix).parent.parent
    cmd = ['docker', 'run', '-d', '--init', '--pull', 'never', '--name', args.name,
           '--network', 'host', '--gpus', 'all', '--user', f'{os.getuid()}:{os.getgid()}',
           '--shm-size', '2g',
           '--mount', f'type=bind,source={conda},target={conda},readonly',
           '--mount', f'type=bind,source={model},target=/models/tlm,readonly',
           '--mount', f'type=bind,source={out},target=/results',
           '-e', 'PYTHONPATH=/usr/local/lib/python3.12/dist-packages',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
           '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'VLLM_NO_USAGE_STATS=1',
           '-e', 'HF_HOME=/results/runtime-cache/huggingface',
           '-e', 'XDG_CACHE_HOME=/results/runtime-cache',
           '-e', 'XDG_CONFIG_HOME=/results/runtime-cache/config',
           '-e', 'FLASHINFER_WORKSPACE_BASE=/results/runtime-cache/flashinfer',
           '-e', 'TORCHINDUCTOR_CACHE_DIR=/results/runtime-cache/torchinductor',
           '-e', 'TRITON_CACHE_DIR=/results/runtime-cache/triton',
           '-e', 'OMP_NUM_THREADS=4', '-e', 'MAX_JOBS=4',
           '-e', 'TORCHINDUCTOR_COMPILE_THREADS=2',
           '--entrypoint', sys.executable, IMAGE, '-m', 'vllm.entrypoints.openai.api_server',
           '--model', '/models/tlm', '--served-model-name', 'specferry-tlm-qwen',
           '--host', '127.0.0.1', '--port', str(args.port), '--dtype', 'bfloat16',
           '--language-model-only', '--reasoning-parser', 'qwen3',
           '--mamba-ssm-cache-dtype', 'float32', '--tensor-parallel-size', '1',
           '--max-model-len', '2048', '--max-num-seqs', '1', '--max-num-batched-tokens', '2048',
           '--max-cudagraph-capture-size', '1', '--kv-cache-memory-bytes', '2147483648',
           '--gpu-memory-utilization', '0.55', '--generation-config', 'vllm',
           '--no-enable-prefix-caching']
    if args.enforce_eager:
        cmd += ['--enforce-eager']
    launch = ['sg', 'docker', '-c', shlex.join(cmd)]
    run = subprocess.run(launch, capture_output=True, text=True)
    record = {'started_at':datetime.now(timezone.utc).isoformat(), 'command':launch,
              'image_id':IMAGE, 'checkpoint':CHECKPOINT,
              'model_revision':manifest['resolved_revision'], 'dtype':'bfloat16',
              'ssm_state_dtype':'float32', 'language_model_only':True,
              'max_model_len':2048, 'concurrency':1, 'enforce_eager':args.enforce_eager,
              'source_sha256':hashes, 'conda_prefix':sys.prefix,
              'exit_code':run.returncode, 'container_id':run.stdout.strip(), 'stderr':run.stderr,
              'service_url':f'http://127.0.0.1:{args.port}',
              'scope':'Standalone target text generation; no draft model or NP101 integration.'}
    (out/'launch.json').write_text(json.dumps(record, indent=2)+'\n')
    print(run.stdout, end='')
    print(run.stderr, file=sys.stderr, end='')
    print('Evidence:', out)
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
