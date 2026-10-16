#!/usr/bin/env python3
"""Run an offline GPU verifier in the cached image using SpecFerry Conda Python.

The existing vLLM server is left running. Run without concurrent inference for
useful timing; telemetry and service counters are retained for review.
"""
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
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:89154ef00dd15368d2b293c167e5cc7dbb521fcfb2fbb77510e0d4df2b820e8f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory under project .cache/runs')
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--phase', choices=('all', 'correctness', 'performance'), default='all')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        parser.error('Use the Python 3.12 SpecFerry Conda interpreter')
    if args.timeout < 1 or args.repeats < 3:
        parser.error('Require a positive timeout and at least three repeats')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / '.cache' / 'runs'):
        parser.error('Output must be under project .cache/runs')
    if output.exists():
        parser.error('Choose a new output directory to preserve previous evidence')
    conda_root = Path(sys.prefix).parent.parent
    name = f'specferry-verifier-{os.getpid()}'
    output.mkdir(parents=True)
    hashes = {}
    for relative in ('scripts/validate_opt_verifier.py', 'python/specferry/verification.py',
                     'scripts/run_opt_verifier.py', 'requirements-tlm-runtime.txt'):
        source = ROOT / relative
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    cmd = ['docker', 'run', '--rm', '--init', '--pull', 'never', '--name', name,
           '--network', 'none', '--gpus', 'all', '--user', f'{os.getuid()}:{os.getgid()}',
           '--shm-size', '2g',
           '--mount', f'type=bind,source={conda_root},target={conda_root},readonly',
           '--mount', f'type=bind,source={ROOT},target=/workspace,readonly',
           '--mount', f'type=bind,source={output},target=/results',
           '-e', 'PYTHONPATH=/results/source/python:/usr/local/lib/python3.12/dist-packages',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
           '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_HOME=/tmp/specferry-hf',
           '-e', 'XDG_CACHE_HOME=/tmp/specferry-cache', '--entrypoint', 'timeout',
           IMAGE, str(args.timeout), sys.executable,
           '/results/source/scripts/validate_opt_verifier.py',
           '--model', '/workspace/.cache/models/facebook/opt-6.7b',
           '--output', '/results', '--dtype', args.dtype, '--phase', args.phase,
           '--repeats', str(args.repeats)]
    # sg applies the already-authorized docker group without sudo or a new login.
    launch = ['sg', 'docker', '-c', shlex.join(cmd)]
    record = {'started_at': datetime.now(timezone.utc).isoformat(), 'command': launch,
              'image_id': IMAGE, 'source_sha256': hashes, 'conda_prefix': sys.prefix}

    def metrics(stage):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=5) as response:
                (output / f'service-metrics-{stage}.txt').write_bytes(response.read())
        except OSError as error:
            record[f'service_metrics_{stage}_error'] = str(error)

    metrics('before')
    monitor = None
    result = 1
    try:
        with (output / 'gpu.csv').open('w') as telemetry, (output / 'run.log').open('w') as log:
            if shutil.which('nvidia-smi'):
                monitor = subprocess.Popen(['nvidia-smi',
                    '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu',
                    '--format=csv', '-l', '2'], stdout=telemetry, stderr=subprocess.STDOUT)
            print(f'Running {args.dtype}/{args.phase}; log: {output / "run.log"}', flush=True)
            result = subprocess.run(launch, stdout=log, stderr=subprocess.STDOUT).returncode
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        metrics('after')
        record.update(exit_code=result, finished_at=datetime.now(timezone.utc).isoformat())
        (output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Exit {result}; results: {output}', flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
