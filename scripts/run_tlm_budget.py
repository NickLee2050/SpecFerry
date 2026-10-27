#!/usr/bin/env python3
"""Local experiment orchestration; never contacts NP101 or restores old services."""
from datetime import datetime,timezone
import argparse,hashlib,json,os,shlex,signal,socket,subprocess,sys,time,threading
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
parser=argparse.ArgumentParser(description='Run local OPT-30B measurements; wait for an idle GPU and stop own services afterward.')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--wait-for-gpu-seconds',type=int,default=3600)
args=parser.parse_args()
if sys.version_info[:2]!=(3,12) or Path(sys.prefix).name!='SpecFerry':parser.error('Use SpecFerry Conda Python 3.12')
OUT=args.output.resolve()
if OUT.exists() or not OUT.is_relative_to(ROOT/'.cache/runs'):parser.error('Choose a new directory under .cache/runs')
if not 0<=args.wait_for_gpu_seconds<=86400:parser.error('Invalid wait deadline')
OUT.mkdir(parents=True)
suffix=str(os.getpid())
profile_name='specferry-opt30b-budget-'+suffix;ordinary_name='specferry-opt30b-graph-'+suffix
state={'started_at':datetime.now(timezone.utc).isoformat(),'status':'starting','np101_connections':False,'started_containers':[],'stopped_containers':[],'restore_old_services':False}
mock=None;owned=[];monitor_stop=threading.Event();samples=[];opener=build_opener(ProxyHandler({}))
def save():(OUT/'orchestration.json').write_text(json.dumps(state,indent=2)+'\n')
def docker(*args,check=True):return subprocess.run(['sg','docker','-c',shlex.join(['docker',*args])],text=True,capture_output=True,check=check)
def run(args):subprocess.run([sys.executable,*args],check=True)
def wait_health(port,name):
 deadline=time.monotonic()+1200
 while time.monotonic()<deadline:
  if not json.loads(docker('inspect',name).stdout)[0]['State']['Running']:raise RuntimeError(name+' exited during startup')
  try:
   with opener.open(f'http://127.0.0.1:{port}/health',timeout=2) as response:
    if response.status==200:return
  except OSError:pass
  time.sleep(3)
 raise RuntimeError('Startup timeout '+name)
def stop(name):
 logs=docker('logs','--timestamps',name,check=False);(OUT/(name+'.log')).write_text(logs.stdout+logs.stderr)
 docker('stop','--timeout','20',name,check=False)
 info=json.loads(docker('inspect',name).stdout)[0]
 if info['State']['Running']:
  docker('kill',name)
  if json.loads(docker('inspect',name).stdout)[0]['State']['Running']:
   raise RuntimeError('Owned test container did not stop: '+name)
 state['stopped_containers'].append(name);save()
def gpu_pids():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
 return [int(line.strip()) for line in text.splitlines() if line.strip().isdigit()]
def require_free_ports(ports):
 for port in ports:
  with socket.socket() as probe:
   probe.bind(('127.0.0.1',port))
def wait_idle():
 deadline=time.monotonic()+args.wait_for_gpu_seconds
 while True:
  active=gpu_pids()
  if not active:
   state['other_gpu_pids']=[]
   return
  state.update(status='waiting_for_gpu',other_gpu_pids=active);save()
  if time.monotonic()>=deadline:raise RuntimeError('GPU wait deadline reached; no competing process was stopped')
  time.sleep(min(10,max(0,deadline-time.monotonic())))
def monitor():
 while not monitor_stop.wait(10):
  try:
   active=gpu_pids();allowed=[]
   for name in owned.copy():
    top=docker('top',name,'-eo','pid',check=False).stdout
    allowed.extend(int(x.strip()) for x in top.splitlines() if x.strip().isdigit())
   foreign=[pid for pid in active if pid not in allowed]
   row={'at':datetime.now(timezone.utc).isoformat(),'phase':state['status'],'gpu_pids':active,'foreign_pids':foreign}
   if state['status'] in ('profiling_native_rounds','measuring_graph_reference') and foreign:
    state['gpu_contended']=True
   samples.append(row)
   with (OUT/'gpu-process-samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
  except Exception as error:
   if state['status'] in ('profiling_native_rounds','measuring_graph_reference'):
    state['gpu_monitor_error']=str(error)
def interrupted(signum, frame):
 state.update(status='interrupted',exit_code=130)
 raise KeyboardInterrupt('Local experiment interrupted')
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
try:
 state['source_sha256']={}
 for relative in ('scripts/run_tlm_budget.py','scripts/benchmark_vllm_rounds.py',
                  'scripts/benchmark_tlm_workloads.py','scripts/analyze_tlm_budget.py',
                  'scripts/mock_dlm_server.py','scripts/serve_vllm_spec.py',
                  'scripts/serve_opt_target.py','configs/tlm-budget-workloads.json',
                  'handoff/tlm-workloads-v1.json'):
  target=OUT/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes((ROOT/relative).read_bytes())
  state['source_sha256'][relative]=hashlib.sha256(target.read_bytes()).hexdigest()
 save()
 threading.Thread(target=monitor,daemon=True).start()
 wait_idle()
 available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])/2**20
 if available<90:raise RuntimeError('Insufficient available memory')
 require_free_ports((8040,8041,8042))
 state['status']='launching_profile_server';save()
 mocklog=(OUT/'mock-server.log').open('w')
 mock=subprocess.Popen([sys.executable,'scripts/mock_dlm_server.py','--model','.cache/models/facebook/opt-30b','--port','8042','--log',str(OUT/'mock-requests.jsonl')],stdout=mocklog,stderr=subprocess.STDOUT)
 run(['scripts/serve_vllm_spec.py','--checkpoint','facebook/opt-30b','--profile-rounds','--dlm-url','http://127.0.0.1:8042/v1/draft','--name',profile_name,'--port','8041','--draft-tokens','8','--timeout','.2','--output',str(OUT/'profile-server')])
 owned.append(profile_name);state['started_containers'].append(profile_name);state['status']='loading_profile_server';save()
 wait_health(8041,profile_name)
 state['status']='profiling_native_rounds';save()
 run(['scripts/benchmark_vllm_rounds.py','--server-output',str(OUT/'profile-server'),'--output',str(OUT/'native')])
 state['status']='stopping_profile_server';save()
 stop(profile_name);owned.remove(profile_name)
 mock.terminate();mock.wait(timeout=10);mock=None
 wait_idle()
 state['status']='loading_graph_reference';save()
 require_free_ports((8040,))
 run(['scripts/serve_opt_target.py','--name',ordinary_name,'--port','8040','--output',str(OUT/'graph-server')])
 owned.append(ordinary_name);state['started_containers'].append(ordinary_name);save()
 wait_health(8040,ordinary_name)
 state['status']='measuring_graph_reference';save()
 run(['scripts/benchmark_tlm_workloads.py','--workloads',str(OUT/'native/workloads.json'),'--output',str(OUT/'graph')])
 state['status']='analyzing_results';save()
 if state.get('gpu_contended') or state.get('gpu_monitor_error'):raise RuntimeError('Competing GPU activity or monitor failure; retain raw timings but do not use them for budgets')
 run(['scripts/analyze_tlm_budget.py','--native',str(OUT/'native/report.json'),'--graph',str(OUT/'graph/report.json'),'--output',str(OUT/'analysis')])
 state['status']='complete';state['exit_code']=0
except Exception as error:
 state.update(status='failed',error=str(error),exit_code=1);print(type(error).__name__,str(error),flush=True)
finally:
 monitor_stop.set()
 for name in owned:stop(name)
 if mock is not None:
  mock.terminate()
  try:mock.wait(timeout=10)
  except subprocess.TimeoutExpired:mock.kill();mock.wait()
 state['finished_at']=datetime.now(timezone.utc).isoformat();save();print(json.dumps(state,indent=2),flush=True)
raise SystemExit(state['exit_code'])
