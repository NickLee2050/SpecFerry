#!/usr/bin/env python3
"""Script: retain actual Qwen chat output, model stopping, and warm decode timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from probe_long_prompt import PROMPTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8020')
    p.add_argument('--model', default='specferry-tlm-qwen')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.cache/runs'):
        p.error('Choose a new directory under .cache/runs')
    out.mkdir(parents=True)
    opener = build_opener(ProxyHandler({}))
    base = args.base_url.rstrip('/')
    checkpoint = ROOT/'.cache/models/Qwen/Qwen3.5-27B'
    manifest = json.loads((checkpoint/'download-manifest.json').read_text())
    report = {'status':'running', 'started_at':datetime.now(timezone.utc).isoformat(),
              'base_url':base, 'model':args.model, 'checkpoint':manifest['repo_id'],
              'revision':manifest['resolved_revision'], 'temperature':0,
              'enable_thinking':False, 'stop_strings':[], 'repetition_penalty':1.0,
              'scope':'Standalone target text validation; no DLM, no speculative inference, no NP101.',
              'measurement':'Single-request client-observed SSE. TTFT includes prefill/API overhead. Decode rate excludes the first chunk. Timing is three short-prompt trials, not a general throughput claim.',
              'source_sha256':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                               for name in ('validate_qwen_tlm.py','probe_long_prompt.py')},
              'trials':[]}

    def publish():
        temp = out/'report.tmp'
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temp.replace(out/'report.json')

    def trial(name, prompt, limit, *, ignore_eos=False, category='sample'):
        payload = {'model':args.model, 'messages':[{'role':'user','content':prompt}],
                   'temperature':0, 'top_p':1.0, 'top_k':-1, 'repetition_penalty':1.0,
                   'max_tokens':limit, 'ignore_eos':ignore_eos,
                   'chat_template_kwargs':{'enable_thinking':False},
                   'stream':True, 'stream_options':{'include_usage':True},
                   'return_token_ids':True, 'cache_salt':uuid.uuid4().hex}
        (out/f'{name}-input.txt').write_text(prompt)
        (out/f'{name}-request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
        req = Request(base+'/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json'})
        started = time.perf_counter()
        tokens, arrivals, texts, reasoning = [], [], [], []
        done = False
        usage = finish = stop = None
        with opener.open(req, timeout=180) as response, (out/f'{name}-events.jsonl').open('w') as log:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                elapsed = time.perf_counter()-started
                log.write(json.dumps({'seconds':elapsed,'event':event},ensure_ascii=False)+'\n')
                if 'error' in event:
                    raise RuntimeError(f'SSE error: {event["error"]}')
                usage = event.get('usage') or usage
                for choice in event.get('choices',[]):
                    ids = choice.get('token_ids') or []
                    if ids:
                        arrivals.append((elapsed,len(ids)))
                        tokens.extend(ids)
                    delta = choice.get('delta') or {}
                    texts.append(delta.get('content') or '')
                    reasoning.append(delta.get('reasoning') or delta.get('reasoning_content') or '')
                    finish = choice.get('finish_reason') or finish
                    if choice.get('stop_reason') is not None:
                        stop = choice['stop_reason']
        total = time.perf_counter()-started
        text = ''.join(texts)
        (out/f'{name}-output.txt').write_text(text)
        if not done or not usage or not finish or len(tokens)!=usage['completion_tokens']:
            raise RuntimeError(f'Incomplete or inconsistent stream: {name}, IDs={len(tokens)}, usage={usage}')
        if ignore_eos and len(tokens)!=limit:
            raise RuntimeError(f'Fixed-token benchmark stopped early: {name}')
        if not text.strip() or not arrivals:
            raise RuntimeError(f'No visible generated answer: {name}')
        first, first_count = arrivals[0]
        window = arrivals[-1][0]-first
        row = {'name':name, 'category':category, 'usage':usage,
               'finish_reason':finish, 'stop_reason':stop, 'ignore_eos':ignore_eos,
               'stopped_by_model':finish=='stop' and not ignore_eos,
               'total_seconds':total, 'ttft_seconds':first,
               'decode_tokens_per_second':(len(tokens)-first_count)/window if window>0 else None,
               'end_to_end_tokens_per_second':len(tokens)/total,
               'token_ids':tokens, 'text':text, 'reasoning_text':''.join(reasoning)}
        report['trials'].append(row)
        publish()
        print(name, 'input=',usage['prompt_tokens'], 'output=',len(tokens), 'finish=',finish,
              'TTFT=',round(first,3), 'decode_tps=',round(row['decode_tokens_per_second'],2)
              if row['decode_tokens_per_second'] else None, flush=True)
        return row

    monitor = None
    try:
        with opener.open(base+'/health',timeout=5) as response:
            if response.status!=200:
                raise RuntimeError('Service is unhealthy')
        with opener.open(base+'/v1/models',timeout=10) as response:
            report['models'] = json.load(response)
        with (out/'gpu.csv').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,power.draw',
                                        '--format=csv','-l','2'],stdout=log,stderr=subprocess.STDOUT)
            publish()
            trial('warmup','Explain how a CPU and a GPU cooperate when running a language model. Write at least 400 words.',
                  64, ignore_eos=True, category='warmup')
            trial('zh-short','请用三条简短建议，说明如何公平比较两个语言模型推理系统的速度。总字数不超过180字。',512)
            trial('en-short','Give three practical tips for fairly comparing the speed of two language model inference systems. Use at most 120 words.',256)
            trial('zh-long',PROMPTS['zh'],1024)
            for index,prompt in enumerate((
                'Explain how a CPU and a GPU cooperate when running a language model. Write at least 400 words.',
                '请介绍语言模型推理中模型权重、提示处理和缓存分别起什么作用，不少于600字。',
                'Write a detailed story about a student building a small robot. Write at least 400 words.',
            ),1):
                trial(f'benchmark-{index}',prompt,128,ignore_eos=True,category='benchmark')
        rows = [r for r in report['trials'] if r['category']=='benchmark']
        report['median_decode_tokens_per_second'] = statistics.median(r['decode_tokens_per_second'] for r in rows)
        report['median_ttft_seconds'] = statistics.median(r['ttft_seconds'] for r in rows)
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
    return 0


if __name__=='__main__':
    raise SystemExit(main())
