#!/usr/bin/env python3
"""Script: probe a draft endpoint, run the native terminal, and retain metrics.

The callback is a SpecFerry contract, not an NP101 SDK API. This script cannot
establish board execution; a device-side run/evidence record is still required.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]


def counters(body):
    values = {}
    for line in body.splitlines():
        if line.startswith('vllm:spec_decode_') and '_total{' in line and 'per_pos' not in line:
            key = line.split('{', 1)[0]
            values[key] = values.get(key, 0) + float(line.rsplit(' ', 1)[1])
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dlm-url', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:8011')
    p.add_argument('--model', default='specferry-tlm-spec')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--max-tokens', type=int, default=32)
    p.add_argument('--draft-tokens', type=int, default=4)
    p.add_argument('--timeout-ms', type=int, default=30000)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    if not 1 <= args.max_tokens <= 2047 or not 1 <= args.draft_tokens <= 32:
        p.error('Require max-tokens in 1..2047 and draft-tokens in 1..32')
    if not 1 <= args.timeout_ms <= 3600000:
        p.error('Require timeout-ms in 1..3600000')
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / '.cache/runs'):
        p.error('Choose a new output directory under .cache/runs')
    binary = ROOT / 'build/draft/specferry'
    if not binary.is_file():
        p.error('Build native/draft first')
    out.mkdir(parents=True)
    prompt = args.prompt_file.read_text() if args.prompt_file else 'The solar system consists of'
    (out/'input.txt').write_text(prompt)
    opener = build_opener(ProxyHandler({}))
    base = args.base_url.rstrip('/')
    report = {'status':'running', 'started_at':datetime.now(timezone.utc).isoformat(),
              'base_url':base, 'dlm_url':args.dlm_url, 'np101_execution_proven':False,
              'scope':'HTTP contract and live engine integration; counters require an otherwise idle service.',
              'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
              'source_revision':subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()}
    report['working_tree_dirty'] = bool(subprocess.check_output(
        ['git','status','--porcelain'], cwd=ROOT, text=True).strip())
    report['source_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ('scripts/check_dlm_link.py', 'native/draft/client.cpp',
                                           'native/draft/tools.cpp', 'protocol/draft-v1.schema.json')}

    def publish():
        (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')

    def http(path, payload=None):
        req = Request(base+path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json'})
        with opener.open(req, timeout=30) as response:
            return response.read().decode()

    def native(arguments, label):
        run = subprocess.run([str(binary), *arguments], capture_output=True, text=True)
        (out/f'{label}-stdout.txt').write_text(run.stdout)
        (out/f'{label}-stderr.txt').write_text(run.stderr)
        report[f'{label}_exit_code'] = run.returncode
        if run.returncode:
            raise RuntimeError(f'{label} failed; see preserved stdout/stderr')
        return run.stdout

    try:
        http('/health')
        report['models'] = json.loads(http('/v1/models'))
        prefix = json.loads(http('/tokenize', {'model':args.model, 'prompt':prompt,
                                               'add_special_tokens':True}))['tokens']
        (out/'prefix.json').write_text(json.dumps(prefix)+'\n')
        probe = native(['probe', '--model-dir',str(ROOT/'.cache/models/facebook/opt-6.7b'),
                        '--dlm-url',args.dlm_url, '--prefix-file',str(out/'prefix.json'),
                        '--draft-tokens',str(args.draft_tokens), '--timeout-ms',str(args.timeout_ms)], 'probe')
        report['probe'] = json.loads(probe)
        before = http('/metrics')
        (out/'metrics-before.prom').write_text(before)
        native(['generate', '--base-url',base, '--model',args.model,
                '--prompt-file',str(out/'input.txt'), '--max-tokens',str(args.max_tokens),
                '--output-dir',str(out/'generation')], 'terminal')
        after = http('/metrics')
        (out/'metrics-after.prom').write_text(after)
        start, end = counters(before), counters(after)
        delta = {key:end.get(key,0)-start.get(key,0) for key in set(start)|set(end)}
        report['speculative_metric_delta'] = delta
        drafted = delta.get('vllm:spec_decode_num_draft_tokens_total', 0)
        accepted = delta.get('vllm:spec_decode_num_accepted_tokens_total', 0)
        report['engine_draft_tokens_observed'] = drafted
        report['engine_accepted_tokens_observed'] = accepted
        report['accepted_over_drafted'] = accepted/drafted if drafted else None
        report['counter_limit'] = 'Counters may include accepted candidates truncated at the output budget; not a per-position acceptance model.'
        report['status'] = 'drafts_observed' if drafted > 0 else 'target_only_fallback'
        report['generation'] = json.loads((out/'generation/report.json').read_text())
    except Exception as error:
        report.update(status='failed', error=str(error))
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        publish()
    print(report['status'], out/'report.json')
    return 0 if report['status'] == 'drafts_observed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
