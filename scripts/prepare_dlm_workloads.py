#!/usr/bin/env python3
"""Offline tokenizer script: prepare shared token inputs and draft-contract requests."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

from draft_test_support import PROTOCOL,prefix_digest,tokenizer_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,12) or Path(sys.prefix).name!='SpecFerry':p.error('Use SpecFerry Conda Python 3.12')
    from transformers import AutoTokenizer
    plan=json.loads(a.plan.read_text());manifest=json.loads((a.model_dir/'download-manifest.json').read_text())
    if manifest['status']!='complete' or manifest['repo_id']!=plan['checkpoint']:raise ValueError('Checkpoint mismatch')
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    tokens=tokenizer.encode(plan['text']*32,add_special_tokens=True,truncation=True,max_length=max(plan['context_tokens']))
    if tokens[0]!=2:raise ValueError('Expected the authorized OPT BOS policy')
    fingerprint=tokenizer_digest(a.model_dir)
    workloads=[];requests=[]
    for n in plan['context_tokens']:
        prefix=tokens[:n]
        if len(prefix)!=n or n+plan['output_tokens']+max(plan['draft_tokens'])>2048:raise ValueError('Invalid context budget')
        workloads.append({'case_id':f'context-{n}','prompt_token_ids':prefix,
                          'max_output_tokens':plan['output_tokens'],
                          'required_context_tokens':n+plan['output_tokens']+max(plan['draft_tokens'])})
        for k in plan['draft_tokens']:
            requests.append({'protocol':PROTOCOL,'request_id':f'fixture-context-{n}-k-{k}',
                             'prefix_sha256':prefix_digest(prefix),'tokenizer_sha256':fingerprint,
                             'draft_mode':'greedy','prefix_token_ids':prefix,'max_draft_tokens':k})
    a.output.mkdir(parents=True,exist_ok=False)
    record={'version':plan['version'],'prepared_at':datetime.now(timezone.utc).isoformat(),
            'checkpoint':manifest['repo_id'],'model_revision':manifest['resolved_revision'],
            'tokenizer_sha256':fingerprint,'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),
            'token_policy':'OPT BOS 2 included exactly once. Reuse supplied integer IDs; do not tokenize again.',
            'generation':{'temperature':0,'ignore_eos':True,'output_tokens':plan['output_tokens']},
            'scope':'Offline prepared inputs; no DLM execution, acceptance or network measurement.',
            'capability_rule':'A future device run must skip cases exceeding its published context limit; no NP101 context support is inferred.',
            'workloads':workloads}
    (a.output/'workloads.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (a.output/'requests.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    print('Prepared',len(workloads),'shared inputs and',len(requests),'protocol requests without network access.')


if __name__=='__main__':main()
