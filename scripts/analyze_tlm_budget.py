#!/usr/bin/env python3
"""Process measured target rounds into explicitly hypothetical serial SD budgets."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
import hashlib
import math
from pathlib import Path
import statistics
import sys


def account_output_budget(rows, prompt_tokens, output_tokens):
    """Keep engine acceptance separate from tokens surviving the API output cap."""
    total=0
    for index,row in enumerate(rows):
        sampled=len(row['sampled_token_ids'])
        before=row['prefix_tokens']-sampled-prompt_tokens
        used=max(0,min(sampled,output_tokens-before))
        row['output_tokens_within_budget']=used
        row['proposal_was_verified']=(index+1<len(rows) and rows[index+1]['target_round'] is not None)
        if row['target_round'] is not None:
            row['target_round']['output_tokens_within_budget']=used
            row['target_round']['accepted_prefix_tokens_within_output_budget']=min(
                row['target_round']['accepted_prefix_tokens'],used)
        total+=used
    if total>output_tokens:raise ValueError('Profiled output exceeds API budget')
    return total


def expected_committed(k, alpha):
    if type(k) is not int or k<1 or not math.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('Require positive K and acceptance in [0,1]')
    return sum(alpha**i for i in range(k+1))


def minimum_draft_rate(k, committed, target_step, verify, overhead=0):
    if type(k) is not int or k<1 or not all(math.isfinite(x) for x in (committed,target_step,verify,overhead)):
        raise ValueError('Invalid finite timing inputs')
    if not 1<=committed<=k+1 or target_step<=0 or verify<0 or overhead<0:
        raise ValueError('Invalid timing/count bounds')
    remaining=committed*target_step-verify-overhead
    return k/remaining if remaining>0 else None


def analyze(native, graph):
    if native['status']!='complete' or graph['status']!='complete':
        raise ValueError('Require complete real measurement reports')
    measurements=[];budgets=[];scenarios=[]
    for w in graph['workloads']:
        context=w['context_tokens'];cases=[c for c in native['cases'] if c['context_tokens']==context]
        baseline=[r['seconds'] for c in cases if c['source']=='disabled' for r in c['rounds']]
        if not baseline:raise ValueError('Missing matching empty-draft baseline')
        reference={'matching_empty_draft':statistics.median(baseline),
                   'ordinary_graph':w['median_decode_seconds_per_token']}
        for k in (1,2,4,8):
            selected=[c for c in cases if c['source']=='external' and c['requested_k']==k and c['scenario']=='all_accept']
            actual=[r for c in selected for r in c['rounds'] if r['draft_tokens']==k]
            full=[r['seconds'] for r in actual if r['accepted_prefix_tokens']==k]
            if not full:raise ValueError(f'No fully accepted real rounds for context={context}, K={k}')
            verify=statistics.median(full)
            http=[row['native_http']['http_seconds'] for c in selected for row in c['adapter_rows'] if row['native_http']]
            histogram=dict(sorted(Counter(r['accepted_prefix_tokens'] for r in actual).items()))
            visible_histogram=dict(sorted(Counter(r['accepted_prefix_tokens_within_output_budget']
                                                  for r in actual).items()))
            measurements.append({'context_tokens':context,'K':k,'full_accept_round_samples':len(full),
                                 'verification_round_seconds':verify,'min_seconds':min(full),'max_seconds':max(full),
                                 'actual_accepted_prefix_histogram':histogram,
                                 'accepted_prefix_histogram_within_output_budget':visible_histogram,
                                 'loopback_callback_median_seconds':statistics.median(http) if http else None,
                                 'reference_step_seconds':reference})
            for baseline_name,target in reference.items():
                for alpha in (.5,.8,.95,1.0):
                    committed=expected_committed(k,alpha)
                    for transport in (0,.001,.01,.05):
                        for synchronization in (0,.01):
                            overhead=transport+synchronization
                            threshold=minimum_draft_rate(k,committed,target,verify,overhead)
                            common={'context_tokens':context,'K':k,'baseline':baseline_name,
                                    'assumed_conditional_acceptance':alpha,'expected_committed_tokens':committed,
                                    'assumed_transport_seconds':transport,'assumed_sync_seconds':synchronization,
                                    'measured_target_step_seconds':target,'measured_verify_seconds':verify,
                                    'finite_draft_rate_can_win':threshold is not None,
                                    'minimum_draft_tokens_per_second':threshold}
                            budgets.append(common)
                            for rate in (3,5,10,20,50):
                                estimated=committed/(k/rate+verify+overhead)
                                scenarios.append({**common,'assumed_draft_tokens_per_second':rate,
                                                  'estimated_output_tokens_per_second':estimated,
                                                  'estimated_speedup_over_selected_baseline':estimated*target})
    return {'generated_at':datetime.now(timezone.utc).isoformat(),'measurements':measurements,
            'budgets':budgets,'scenarios':scenarios,
            'model':'Serial K-token draft + one measured target round + assumed transport/synchronization. An independent constant per-position acceptance is assumed; expected output=sum(alpha**i, i=0..K), including a target token.',
            'limits':['No NP101 or real DLM performance/acceptance measured.',
                      'Uses fully accepted verification time for all assumed acceptance values; partial rejection timing is separately retained in raw measurements.',
                      'Decode-only budget excludes initial prefill, startup and DLM catch-up except the explicit assumed sync delay. Threshold is optimistic when omitted costs are nonzero.',
                      'Target round includes verification, sampling, bookkeeping, scheduler/IPC and a CUDA synchronization; not isolated GPU execution.',
                      'Loopback HTTP callback time is an observed local control cost, not inter-host RTT; modeled transport delays are explicit assumptions.',
                      'Graph baseline differs from eager/synchronous integration. Both comparisons are reported; neither establishes cross-device speedup.',
                      'Rates at the computed threshold tie the baseline; acceleration requires a strictly greater draft rate.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sys.version_info[:2]!=(3,12) or Path(sys.prefix).name!='SpecFerry':p.error('Use SpecFerry Conda Python 3.12')
    a.output.mkdir(parents=True,exist_ok=False)
    result=analyze(json.loads(a.native.read_text()),json.loads(a.graph.read_text()))
    result['evidence']={'native':str(a.native),'ordinary_graph':str(a.graph)}
    source=a.output/'analyze_tlm_budget.py';source.write_bytes(Path(__file__).read_bytes())
    result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    result['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (a.native,a.graph)}
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    for key in ('measurements','budgets','scenarios'):
        rows=result[key]
        with (a.output/(key+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows({key:json.dumps(value) if isinstance(value,(dict,list)) else value
                              for key,value in row.items()} for row in rows)
    print('Saved',len(result['budgets']),'budgets and',len(result['scenarios']),'hypothetical scenarios to',a.output)


if __name__=='__main__':main()
