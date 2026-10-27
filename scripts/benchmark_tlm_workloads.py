#!/usr/bin/env python3
"""Measure graph-enabled ordinary decoding on exactly the prepared token inputs."""
import argparse
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path
import statistics
import sys
import time
from urllib.request import Request,build_opener,ProxyHandler
import uuid

from benchmark_vllm_rounds import loopback
from validate_vllm_spec import publish


def stream(base,prefix,count):
    obj={'model':'specferry-tlm-opt-target','prompt':prefix,'max_tokens':count,
         'temperature':0,'top_p':1,'top_k':-1,'repetition_penalty':1,
         'ignore_eos':True,'add_special_tokens':False,'stream':True,'logprobs':0,
         'return_token_ids':True,'stream_options':{'include_usage':True},'cache_salt':uuid.uuid4().hex}
    req=Request(base+'/v1/completions',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
    started=time.perf_counter();arrivals=[];text=[];ids=[];usage=None;done=False
    with build_opener(ProxyHandler({})).open(req,timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            data=line[6:].strip()
            if data==b'[DONE]':done=True;break
            event=json.loads(data)
            if 'error' in event:raise RuntimeError(event['error'])
            usage=event.get('usage') or usage
            for choice in event.get('choices',[]):
                n=len((choice.get('logprobs') or {}).get('tokens',[]))
                if n:arrivals.append((time.perf_counter()-started,n))
                text.append(choice.get('text',''));ids.extend(choice.get('token_ids') or [])
    if not done or not usage or usage['completion_tokens']!=count or sum(n for _,n in arrivals)!=count:
        raise RuntimeError('Incomplete streamed output')
    if usage['prompt_tokens']!=len(prefix):raise RuntimeError('Prompt tokenization changed')
    return {'prompt_tokens':len(prefix),'output_tokens':count,'ttft_seconds':arrivals[0][0],
            'decode_seconds_per_token':(arrivals[-1][0]-arrivals[0][0])/(count-arrivals[0][1]),
            'total_seconds':time.perf_counter()-started,'token_ids':ids,'text':''.join(text)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8040')
    p.add_argument('--workloads',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,12) or Path(sys.prefix).name!='SpecFerry':p.error('Use SpecFerry Conda Python 3.12')
    base=loopback(a.base_url);data=json.loads(a.workloads.read_text());a.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','configuration':'FP16 OPT-30B, graph-enabled ordinary vLLM, one sequence, no prefix cache, 4GiB KV cache',
            'started_at':datetime.now(timezone.utc).isoformat(),'workloads':[]}
    root=Path(__file__).resolve().parents[1]
    report['source_sha256']={}
    for relative in ('scripts/benchmark_tlm_workloads.py','scripts/benchmark_vllm_rounds.py',
                     'scripts/validate_vllm_spec.py','scripts/draft_test_support.py'):
        target=a.output/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/relative).read_bytes())
        report['source_sha256'][relative]=hashlib.sha256(target.read_bytes()).hexdigest()
    report['workload_sha256']=hashlib.sha256(a.workloads.read_bytes()).hexdigest()
    try:
        for w in data['workloads']:
            warmup=stream(base,w['prompt_token_ids'],16)
            rows=[stream(base,w['prompt_token_ids'],data['plan']['output_tokens']) for _ in range(data['plan']['repeats'])]
            for row in rows:row['matches_profile_reference']=row['token_ids']==w['oracle_token_ids'][:row['output_tokens']]
            med=statistics.median(row['decode_seconds_per_token'] for row in rows)
            report['workloads'].append({'context_tokens':len(w['prompt_token_ids']),'warmup':warmup,
                                       'median_decode_seconds_per_token':med,'median_decode_tokens_per_second':1/med,
                                       'median_ttft_seconds':statistics.median(row['ttft_seconds'] for row in rows),'trials':rows})
            publish(a.output/'report.json',report)
            print('Graph target context',len(w['prompt_token_ids']),'decode',1/med,'tokens/s',flush=True)
        report['status']='complete'
    except Exception as error:report.update(status='failed',error=str(error));raise
    finally:report['finished_at']=datetime.now(timezone.utc).isoformat();publish(a.output/'report.json',report)


if __name__=='__main__':main()
