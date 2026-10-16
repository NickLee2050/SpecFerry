#!/usr/bin/env python3
"""Exercise native vLLM greedy verification via a loopback oracle DLM fixture.

Oracle candidates are synthetic. This measures integration correctness, not
NP101 performance or speculative speedup. No Transformers verifier is used.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from draft_test_support import tokenizer_digest

OPENER = build_opener(ProxyHandler({}))
PROMPTS = ['The solar system consists of',
           'In a quiet village near the mountains, there lived a',
           'To make a simple vegetable soup, first prepare']


def request(base, path, obj=None):
    req = Request(base + path, data=None if obj is None else json.dumps(obj).encode(),
                  headers={'Content-Type':'application/json'})
    with OPENER.open(req, timeout=120) as response:
        return json.load(response)


def publish(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def spec_metrics(base):
    with OPENER.open(base+'/metrics', timeout=10) as response:
        body=response.read().decode()
    values={}
    for line in body.splitlines():
        if line.startswith('vllm:spec_decode_') and '_total{' in line:
            key=line.split('{',1)[0]
            if 'per_pos' not in key:
                values[key]=values.get(key,0)+float(line.rsplit(' ',1)[1])
    return values


def complete(base, prefix, count, **kwargs):
    obj={'model':'specferry-tlm-spec','prompt':prefix,'temperature':0,
         'max_tokens':count,'ignore_eos':True,'return_token_ids':True,
         'add_special_tokens':False,'cache_salt':uuid.uuid4().hex, **kwargs}
    started=time.perf_counter()
    result=request(base,'/v1/completions',obj)
    return result,time.perf_counter()-started


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',default='http://127.0.0.1:8011')
    p.add_argument('--mock-url',default='http://127.0.0.1:8012')
    p.add_argument('--model',type=Path,default=Path('.cache/models/facebook/opt-6.7b'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dtype',choices=('float16','float32'),default='float16')
    p.add_argument('--continue-on-mismatch',action='store_true',
                   help='Retain numerical mismatches while exercising remaining cases; exit remains nonzero.')
    args=p.parse_args()
    if sys.version_info[:2] != (3,12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    if args.output.exists():
        p.error('Choose a new output file')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    summary={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
             'backend':'vLLM 0.28.0 native custom proposer + rejection sampler',
             'target':f'OPT-6.7B {args.dtype}; eager; synchronous; TP/PP/DP=1; concurrency=1',
             'np101_execution':False,'network':'real loopback HTTP callback; no device NIC measured',
             'draft_source':'precomputed same-engine greedy oracle, with synthetic corruption/failures',
             'cases':[],'traces':[]}
    base,mock=args.base_url.rstrip('/'),args.mock_url.rstrip('/')

    def mode(name, **extra):
        request(mock,'/control',{'mode':name,**extra})

    def record(name, trace, count, **kwargs):
        before=spec_metrics(base)
        result,elapsed=complete(base,trace['prompt_token_ids'],count,**kwargs)
        after=spec_metrics(base)
        choice=result['choices'][0]
        tokens=choice['token_ids']
        expected=trace['output_token_ids'][:count]
        row={'case':name,'prompt':trace['text'],'status':'pass' if tokens==expected else 'failed',
             'token_ids':tokens,'expected_token_ids':expected,'usage':result['usage'],
             'finish_reason':choice['finish_reason'],'wall_seconds':elapsed,
             'metric_delta':{key:after.get(key,0)-before.get(key,0) for key in set(before)|set(after)}}
        summary['cases'].append(row)
        publish(args.output,summary)
        if row['status']!='pass' and not args.continue_on_mismatch:
            raise RuntimeError(f'Native vLLM greedy mismatch in {name}: {trace["text"]!r}')
        print(name,row['status'],count,'tokens;',row['metric_delta'],flush=True)
        return row

    try:
        mode('no_draft')
        for text in PROMPTS:
            prefix=request(base,'/tokenize',{'model':'specferry-tlm-spec','prompt':text,
                                            'add_special_tokens':True})['tokens']
            result,_=complete(base,prefix,96)
            tokens=result['choices'][0]['token_ids']
            if len(tokens)!=96:
                raise RuntimeError('Reference stream is incomplete')
            summary['traces'].append({'text':text,'prompt_token_ids':prefix,'output_token_ids':tokens})
        fixtures={'tokenizer_sha256':tokenizer_digest(args.model),'traces':summary['traces']}
        publish(args.output.with_name(args.output.stem+'-traces.json'),fixtures)
        mode('no_draft',traces=summary['traces'])
        publish(args.output,summary)
        for scenario in ('all_accept','first_reject','middle_reject'):
            mode(scenario)
            for trace in summary['traces']:
                row=record(scenario,trace,48)
                accepted=row['metric_delta'].get('vllm:spec_decode_num_accepted_tokens_total',0)
                drafted=row['metric_delta'].get('vllm:spec_decode_num_draft_tokens_total',0)
                if drafted<=0 or (scenario=='first_reject' and accepted!=0):
                    raise RuntimeError('Expected native draft/rejection metrics are absent or inconsistent')
                if scenario!='first_reject' and accepted<=0:
                    raise RuntimeError('Native engine did not accept oracle draft tokens')
        mode('all_accept')
        for trace in summary['traces']:
            record('output_budget',trace,3)
        for scenario in ('stale','invalid_token','timeout'):
            mode(scenario)
            record(scenario+'_fallback',summary['traces'][0],8)

        # Exact same-engine comparison for explicit stop IDs, including a stop
        # that lands inside an accepted candidate block. These are not natural EOS.
        trace=summary['traces'][0]
        stop_id=trace['output_token_ids'][2]
        mode('no_draft')
        ordinary,_=complete(base,trace['prompt_token_ids'],24,stop_token_ids=[stop_id])
        mode('all_accept')
        speculative,_=complete(base,trace['prompt_token_ids'],24,stop_token_ids=[stop_id])
        a,b=ordinary['choices'][0],speculative['choices'][0]
        if a['token_ids']!=b['token_ids'] or b['finish_reason']!='stop':
            raise RuntimeError('Native stop-token behavior differs from target-only')
        summary['cases'].append({'case':'stop_token','status':'pass','stop_token_id':stop_id,
                                  'reference':ordinary,'speculative':speculative})

        mode('all_accept')
        obj={'model':'specferry-tlm-spec','prompt':trace['prompt_token_ids'],'temperature':0,
             'max_tokens':32,'ignore_eos':True,'return_token_ids':True,'add_special_tokens':False,
             'stream':True,'stream_options':{'include_usage':True}}
        req=Request(base+'/v1/completions',data=json.dumps(obj).encode(),
                    headers={'Content-Type':'application/json'})
        tokens=[];done=False;usage=None
        with OPENER.open(req,timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                if line[6:].strip()==b'[DONE]':
                    done=True;break
                event=json.loads(line[6:])
                usage=event.get('usage') or usage
                for choice in event.get('choices',[]):
                    tokens.extend(choice.get('token_ids') or [])
        if not done or tokens!=trace['output_token_ids'][:32] or usage['completion_tokens']!=32:
            raise RuntimeError('Native speculative SSE stream differs from target-only')
        summary['cases'].append({'case':'streaming','status':'pass','token_ids':tokens,'usage':usage})
        summary['status']='failed' if any(r['status']!='pass' for r in summary['cases']) else 'pass'
        summary['metrics_final']=spec_metrics(base)
    except Exception as error:
        summary.update(status='failed',error=str(error))
        raise
    finally:
        # Leave the fixture neutral: ordinary generation until a test mode is set.
        try:
            mode('no_draft')
        except OSError:
            pass
        summary['finished_at']=datetime.now(timezone.utc).isoformat()
        publish(args.output,summary)
    return 0 if summary['status']=='pass' else 1


if __name__=='__main__':
    raise SystemExit(main())
