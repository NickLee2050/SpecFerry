#!/usr/bin/env python3
"""Measure native target rounds with a local synthetic oracle; never contact a board."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from urllib.parse import urlsplit
import uuid

from validate_vllm_spec import request, complete, spec_metrics, publish
from draft_test_support import tokenizer_digest
from analyze_tlm_budget import account_output_budget

ROOT = Path(__file__).resolve().parents[1]


def loopback(url):
    if urlsplit(url).scheme != 'http' or urlsplit(url).hostname != '127.0.0.1':
        raise ValueError('Only literal IPv4 loopback HTTP is authorized')
    return url.rstrip('/')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8041')
    parser.add_argument('--mock-url', default='http://127.0.0.1:8042')
    parser.add_argument('--server-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/tlm-budget-workloads.json')
    parser.add_argument('--workloads', type=Path, default=ROOT/'handoff/tlm-workloads-v1.json')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        parser.error('Use SpecFerry Conda Python 3.12')
    base, mock = loopback(args.base_url), loopback(args.mock_url)
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'.cache/runs'):
        parser.error('Choose a new project .cache/runs directory')
    out.mkdir(parents=True)
    plan = json.loads(args.plan.read_text())
    prepared = json.loads(args.workloads.read_text())
    if prepared['plan_sha256'] != hashlib.sha256(args.plan.read_bytes()).hexdigest():
        raise ValueError('Prepared workload plan changed')
    if prepared['tokenizer_sha256'] != tokenizer_digest(ROOT/'.cache/models/facebook/opt-30b'):
        raise ValueError('Prepared tokenizer identity changed')
    publish(out/'plan.json', plan)
    control = args.server_output/'profile-control.json'
    audit = args.server_output/'rounds.jsonl'
    report = {'status':'running', 'started_at':datetime.now(timezone.utc).isoformat(),
              'scope':'Local synthetic oracle only; no NP101 connection or device inference.',
              'measurement':'Target round wall time from returning preceding drafts to the next proposer entry after CUDA synchronization; includes native verification, sampling, bookkeeping and scheduler/IPC. Excludes first prefill, terminal rounds without a subsequent proposer call, fixture call and audit write. Not isolated GPU kernel time.',
              'configuration':'FP16, vLLM 0.28.0, OPT-30B, eager, synchronous, max speculative tokens=8, prefix cache off, one sequence.',
              'cases':[], 'workloads':[]}

    report['source_sha256'] = {}
    for name in ('scripts/benchmark_vllm_rounds.py', 'scripts/validate_vllm_spec.py',
                 'scripts/draft_test_support.py', 'scripts/analyze_tlm_budget.py',
                 'configs/tlm-budget-workloads.json',
                 'handoff/tlm-workloads-v1.json'):
        target = out/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        report['source_sha256'][name]=hashlib.sha256(target.read_bytes()).hexdigest()

    def save(): publish(out/'report.json', report)

    def set_control(run_id, source, k):
        publish(control, {'run_id':run_id, 'source':source, 'draft_tokens':max(1,k)})

    def mode(name, **extra): request(mock, '/control', {'mode':name, **extra})

    def run_case(workload, source, k, scenario, repeat, count=None, retain=True):
        run_id = uuid.uuid4().hex
        set_control(run_id, source, k)
        mode(scenario)
        count = count or plan['output_tokens']
        before = spec_metrics(base)
        result, elapsed = complete(base, workload['prompt_token_ids'], count)
        after = spec_metrics(base)
        choice=result['choices'][0]
        ids=choice['token_ids']
        if len(ids)!=count or result['usage']['completion_tokens']!=count:
            raise RuntimeError('Incomplete fixed-output request')
        if result['usage']['prompt_tokens']!=len(workload['prompt_token_ids']):
            raise RuntimeError('Prompt tokenization changed')
        rows=[json.loads(line) for line in audit.read_text().splitlines()
              if line and json.loads(line).get('run_id')==run_id]
        accounted=account_output_budget(rows,len(workload['prompt_token_ids']),count)
        rounds=[row['target_round'] for row in rows if row['target_round'] is not None]
        if not rounds:
            raise RuntimeError('No native target rounds were measured')
        expected=workload['oracle_token_ids'][:count]
        mismatch=next((i for i,(a,b) in enumerate(zip(ids,expected)) if a!=b),None)
        row={'run_id':run_id,'context_tokens':len(workload['prompt_token_ids']),
             'source':source,'requested_k':k,'scenario':scenario,'repeat':repeat,
             'wall_seconds':elapsed,'output_tokens':len(ids),'token_ids':ids,
             'profiled_output_tokens_within_budget':accounted,
             'unverified_terminal_draft_tokens':sum(len(x['draft_token_ids']) for x in rows
                                                    if not x['proposal_was_verified']),
             'text':choice['text'],'usage':result['usage'],'finish_reason':choice['finish_reason'],
             'matches_empty_draft_reference':ids==expected,'first_mismatch_index':mismatch,
             'metric_delta':{key:after.get(key,0)-before.get(key,0) for key in set(before)|set(after)},
             'rounds':rounds,'adapter_rows':rows}
        if source=='disabled' and any(x['native_http'] is not None for x in rows):
            raise RuntimeError('Empty-draft baseline unexpectedly used HTTP')
        statuses=[x['native_http']['status'] for x in rows if x['native_http']]
        failure_case=scenario in ('stale','invalid_token','timeout')
        if source=='external' and (not statuses or any(s!=('fallback' if failure_case else 'ok') for s in statuses)):
            raise RuntimeError('Unexpected native protocol outcome')
        if failure_case and ids!=expected:
            raise RuntimeError('Failure fallback changed ordinary token output')
        if retain:
            report['cases'].append(row);save()
            med=statistics.median(x['seconds'] for x in rounds)
            print(f"context={len(workload['prompt_token_ids'])} {source} K={k} {scenario} repeat={repeat}: target-round {med*1000:.2f}ms, reference_match={ids==expected}",flush=True)
        return row

    try:
        save()
        tokenized=request(base,'/tokenize',{'model':'specferry-tlm-spec',
                           'prompt':plan['text']*8,'add_special_tokens':True})['tokens']
        if len(tokenized)<max(plan['context_tokens']):raise RuntimeError('Insufficient workload tokens')
        for context in plan['context_tokens']:
            prefix=tokenized[:context]
            shared=next(w['prompt_token_ids'] for w in prepared['workloads'] if len(w['prompt_token_ids'])==context)
            if prefix!=shared:raise RuntimeError('Live vLLM tokenizer differs from shared workload IDs')
            set_control(uuid.uuid4().hex,'disabled',1)
            mode('no_draft')
            result,_=complete(base,prefix,plan['output_tokens']+max(plan['draft_tokens'])+8)
            ids=result['choices'][0]['token_ids']
            if len(ids)!=plan['output_tokens']+max(plan['draft_tokens'])+8:raise RuntimeError('Incomplete oracle')
            report['workloads'].append({'case_id':f'context-{context}', 'prompt_token_ids':prefix,
                                        'oracle_token_ids':ids})
            print('Prepared local oracle for context',context,flush=True)
        traces=[{'prompt_token_ids':w['prompt_token_ids'],'output_token_ids':w['oracle_token_ids']}
                for w in report['workloads']]
        mode('no_draft',traces=traces)
        publish(out/'workloads.json',{'plan':plan,'tokenizer_sha256':tokenizer_digest(ROOT/'.cache/models/facebook/opt-30b'),
                                    'workloads':report['workloads']})
        for w in report['workloads']:
            run_case(w,'disabled',0,'no_draft',-1,count=12,retain=False)
            for k in plan['draft_tokens']:
                run_case(w,'external',k,'all_accept',-1,count=20,retain=False)
        order=[(w,source,k,scenario,repeat) for w in report['workloads']
               for source,k,scenario in [('disabled',0,'no_draft')]+
                   [('external',k,'all_accept') for k in plan['draft_tokens']]
               for repeat in range(plan['repeats'])]
        random.Random(plan['seed']).shuffle(order)
        for args_case in order:run_case(*args_case)
        for w in report['workloads']:
            for scenario in ('first_reject','middle_reject'):
                run_case(w,'external',4,scenario,0,count=16)
        for scenario in ('stale','invalid_token','timeout'):
            run_case(report['workloads'][0],'external',4,scenario,0,count=4)
        report['status']='complete'
        report['numerical_mismatch_cases']=sum(not c['matches_empty_draft_reference'] for c in report['cases'])
        report['correctness_limit']='FP16 differences versus empty-draft greedy are retained; no universal numerical equivalence claim. Protocol, sampled-prefix advancement, fixed-output counts and failure fallback are checked.'
    except Exception as error:
        report.update(status='failed',error=str(error));raise
    finally:
        set_control('finished','disabled',1)
        try: mode('no_draft')
        except OSError: pass
        report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    return 0


if __name__=='__main__':raise SystemExit(main())
