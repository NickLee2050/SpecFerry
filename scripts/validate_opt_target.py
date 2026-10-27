#!/usr/bin/env python3
"""Retain OPT target timings using the established benchmark and an English Q&A."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from probe_long_prompt import PROMPTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', choices=('facebook/opt-6.7b', 'facebook/opt-30b'), default='facebook/opt-30b')
    p.add_argument('--base-url', default='http://127.0.0.1:8030')
    p.add_argument('--model', default='specferry-tlm-opt-target')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.cache/runs'):
        p.error('Choose a new directory under .cache/runs')
    model = ROOT/'.cache/models'/args.checkpoint
    manifest = json.loads((model/'download-manifest.json').read_text())
    if manifest['repo_id'] != args.checkpoint or manifest['status'] != 'complete':
        p.error('Require the verified authorized checkpoint')
    out.mkdir(parents=True)
    report = {'status':'running', 'started_at':datetime.now(timezone.utc).isoformat(),
              'checkpoint':args.checkpoint, 'revision':manifest['resolved_revision'],
              'base_url':args.base_url, 'model':args.model, 'dtype':'float16',
              'scope':'Standalone target generation. No draft model or device acceptance-rate measurement.',
              'source_sha256':{}}
    for name in ('validate_opt_target.py', 'benchmark_tlm.py', 'probe_long_prompt.py'):
        target = out/'source/scripts'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'scripts'/name, target)
        report['source_sha256'][name] = hashlib.sha256(target.read_bytes()).hexdigest()

    def publish():
        temp = out/'report.tmp'
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temp.replace(out/'report.json')

    monitor = None
    try:
        publish()
        with (out/'gpu.csv').open('w') as gpu:
            monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,power.draw',
                                        '--format=csv','-l','2'],stdout=gpu,stderr=subprocess.STDOUT)
            subprocess.run([sys.executable,str(ROOT/'scripts/benchmark_tlm.py'),
                            '--base-url',args.base_url,'--model',args.model,
                            '--tokens','128','--output',str(out/'benchmark.json')],check=True)
            benchmark = json.loads((out/'benchmark.json').read_text())
            report['median_decode_tokens_per_second'] = benchmark['median_decode_tokens_per_second']
            report['median_ttft_seconds'] = benchmark['median_ttft_seconds']
            publish()
            prompt = 'Question: '+PROMPTS['en']+'\n\nAnswer:'
            payload = {'model':args.model,'prompt':prompt,'temperature':0,'top_p':1.0,'top_k':-1,
                       'repetition_penalty':1.0,'max_tokens':512,'ignore_eos':False,
                       'return_token_ids':True,'cache_salt':uuid.uuid4().hex}
            (out/'en-long-qa-input.txt').write_text(prompt)
            (out/'en-long-qa-request.json').write_text(json.dumps(payload,indent=2)+'\n')
            request = Request(args.base_url.rstrip('/')+'/v1/completions',
                              data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            started = time.perf_counter()
            with build_opener(ProxyHandler({})).open(request,timeout=600) as response:
                result = json.load(response)
            elapsed = time.perf_counter()-started
            (out/'en-long-qa-response.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            choice = result['choices'][0]
            ids = choice.get('token_ids') or []
            if len(ids)!=result['usage']['completion_tokens'] or choice['finish_reason'] not in ('stop','length'):
                raise RuntimeError('Incomplete Q&A token count or finish reason')
            if not choice['text'].strip():
                raise RuntimeError('Empty Q&A answer')
            (out/'en-long-qa-output.txt').write_text(choice['text'])
            report['qa'] = {'usage':result['usage'],'finish_reason':choice['finish_reason'],
                            'total_seconds':elapsed,'text':choice['text'],
                            'stopped_by_model':choice['finish_reason']=='stop'}
            reference = ROOT/'.cache/models/facebook/opt-6.7b'
            report['tokenizer_files_match_opt6_7b'] = {
                name:hashlib.sha256((model/name).read_bytes()).hexdigest()==
                     hashlib.sha256((reference/name).read_bytes()).hexdigest()
                for name in ('vocab.json','merges.txt')}
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        if monitor:
            monitor.terminate()
            monitor.wait(timeout=5)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        publish()
    print('Validated:',args.checkpoint,'decode=',report['median_decode_tokens_per_second'],flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
