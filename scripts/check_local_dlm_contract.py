#!/usr/bin/env python3
"""Replay prepared inputs against a loopback fixture using the real C++ client."""
import argparse
from datetime import datetime,timezone
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.request import Request,build_opener,ProxyHandler

from mock_dlm_server import make_handler
from draft_test_support import tokenizer_digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workloads',type=Path,default=ROOT/'handoff/tlm-workloads-v1.json')
    p.add_argument('--model-dir',type=Path,default=ROOT/'.cache/models/facebook/opt-30b')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,12) or Path(sys.prefix).name!='SpecFerry':p.error('Use SpecFerry Conda Python 3.12')
    a.output.mkdir(parents=True,exist_ok=False)
    data=json.loads(a.workloads.read_text());fingerprint=tokenizer_digest(a.model_dir)
    if data['tokenizer_sha256']!=fingerprint:raise ValueError('Tokenizer identity mismatch')
    traces=[{'prompt_token_ids':w['prompt_token_ids'],'output_token_ids':list(range(100,116))} for w in data['workloads']]
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(traces,fingerprint,a.output/'fixture-requests.jsonl'))
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    report={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
            'np101_connections':False,'scope':'Local native-client protocol replay; synthetic IDs, no DLM or GPU execution.',
            'cases':[]}
    def mode(name):
        req=Request(base+'/control',data=json.dumps({'mode':name}).encode(),headers={'Content-Type':'application/json'})
        with build_opener(ProxyHandler({})).open(req,timeout=2):pass
    def probe(prefix,k,timeout=1000):
        path=a.output/'prefix.json';path.write_text(json.dumps(prefix))
        result=subprocess.run([str(ROOT/'build/draft/specferry'),'probe','--model-dir',str(a.model_dir),
                               '--dlm-url',base+'/v1/draft','--prefix-file',str(path),
                               '--draft-tokens',str(k),'--timeout-ms',str(timeout)],capture_output=True,text=True,timeout=5)
        return result,json.loads(result.stdout)
    try:
        for w in data['workloads']:
            mode('all_accept')
            for k in (1,2,4,8):
                result,row=probe(w['prompt_token_ids'],k)
                if result.returncode or row['draft_token_ids']!=list(range(100,100+k)):
                    raise RuntimeError('Native client did not preserve supplied IDs or candidate length')
                report['cases'].append({'context_tokens':len(w['prompt_token_ids']),'K':k,'audit':row,'status':'pass'})
            # Simulate committing only a prefix of the old proposal, then replacing
            # the rest with an authoritative target token. No device cache is inferred.
            replacement=w['prompt_token_ids']+[100,101,999]
            result,row=probe(replacement,4)
            if result.returncode or row['draft_token_ids']!=[]:raise RuntimeError('Fixture reused a rejected suffix')
            report['cases'].append({'case':'rejected_suffix_not_reused','context_tokens':len(replacement),'status':'pass','audit':row})
        for scenario in ('stale','invalid_token','timeout'):
            mode(scenario);result,row=probe(data['workloads'][0]['prompt_token_ids'],4,timeout=50)
            if not result.returncode or row['status']!='fallback':raise RuntimeError('Failure did not fail closed')
            report['cases'].append({'case':scenario,'status':'pass','audit':row})
        report['status']='pass'
    except Exception as error:report.update(status='failed',error=str(error));raise
    finally:
        server.shutdown();server.server_close();thread.join()
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed',len(report['cases']),'local protocol cases; no board connection.')


if __name__=='__main__':main()
