#!/usr/bin/env python3
"""Script: measure long-prompt SSE output and natural stopping on the local TLM."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

PROMPTS = {
    'en': (
        "I am a first-year graduate student working on a research project about running language models "
        "across two different devices, and I would like some practical advice on how to organize the "
        "next three months of work. One device is a small experimental accelerator with less than three "
        "gigabytes of usable memory, and the other is a more capable GPU machine connected to it over "
        "a network. I have already managed to run a small model on the experimental device, but its "
        "generation speed is only about two tokens per second. The larger model on the GPU machine "
        "can produce around fifteen tokens per second on its own. My idea is for the small model to "
        "suggest several tokens at a time and for the larger model to check them together, but I am "
        "worried that waiting for the small device and transferring information over the network will "
        "make the complete system slower rather than faster. I also need to deal with an unreliable "
        "device driver, a limited amount of memory for model weights and cached state, and the possibility "
        "that the models will disagree frequently. At the moment I can run basic inference and collect "
        "logs, but I do not yet have a complete application that sends a question from the small device, "
        "coordinates both models, and displays the final answer on that device. I would like to understand "
        "which measurements I should collect first, how I should define fair baseline experiments, and "
        "how to decide whether an optimization actually helps. I have limited time and cannot buy more "
        "hardware, so I need to choose a few experiments that are realistic and useful instead of trying "
        "every possible approach. I also want to avoid making claims based only on idealized simulations. "
        "Could you suggest a clear weekly plan, explain the most important risks in simple terms, and "
        "describe what a convincing final demonstration would look like? Please also explain what I "
        "should do if the small device remains slow after the basic implementation is complete, and how "
        "I can distinguish an interesting research result from an improvement that only works because "
        "the chosen baseline is weak. I would appreciate a concrete answer that a student could follow "
        "without needing a large engineering team."
    ),
    'zh': (
        '我是一名刚开始做科研的研究生，正在尝试让两台通过网络连接的设备协作运行语言模型，'
        '希望你帮我安排一下未来三个月的工作。其中一台是实验性的小型加速板卡，可用内存不到三GiB，'
        '另一台是性能更好的GPU主机。我已经在板卡上运行了一个小模型，但它每秒只能生成大约两个token，'
        'GPU上的大模型单独运行时每秒可以生成大约十五个token。我原本希望小模型一次提供多个候选，'
        '让大模型并行检查这些候选，从而减少大模型逐个生成的次数。但我担心小模型太慢，加上网络延迟以后，'
        '整个系统反而会更慢。现在还有几个实际困难：板卡驱动不够稳定，模型权重和缓存需要争用有限内存，'
        '两个模型的预测也可能经常不一致。目前我能运行基本推理并保存日志，但还没有完整打通从终端输入问题、'
        '协调两端推理到在终端显示最终答案的应用。我只有一个人做这个项目，不能增加硬件，时间也比较有限。'
        '你能不能给出一个按周安排的具体计划，告诉我最先应该测哪些数据，如何选择公平的对比基线，'
        '以及怎样判断优化是否真正有效？如果小模型一直很慢，我应该怎样调整研究方向？'
        '另外，请解释哪些结果只能说明系统可以运行，哪些结果才足以支持性能改进的结论，'
        '并描述一下最终演示应该包含哪些内容。我希望建议具体、容易执行，也不要把理想化模拟的结果当作真实硬件收益。'
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:8000')
    p.add_argument('--model', default='specferry-tlm')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        p.error('Use SpecFerry Conda Python 3.12')
    if args.output.exists():
        p.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    base = args.base_url.rstrip('/')
    opener = build_opener(ProxyHandler({}))
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'base_url': base, 'model': args.model, 'temperature': 0, 'ignore_eos': False,
              'stop_strings': [], 'trials': [],
              'scope': 'Ordinary vLLM TLM-only service, no draft model; one observation per scenario.'}

    def publish():
        temporary = args.output/'results.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temporary.replace(args.output/'results.json')

    def json_request(path, payload=None):
        req = Request(base+path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json'})
        with opener.open(req, timeout=30) as response:
            return json.load(response)

    def trial(language, mode, prompt, budget):
        payload = {'model':args.model, 'prompt':prompt, 'temperature':0, 'ignore_eos':False,
                   'stream':True, 'stream_options':{'include_usage':True},
                   'return_token_ids':True, 'cache_salt':uuid.uuid4().hex}
        if mode == 'null':
            payload['max_tokens'] = None
        elif mode == 'context_budget':
            payload['max_tokens'] = budget
        name = f'{language}-{mode}'
        (args.output/f'{name}-request.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        started = time.perf_counter()
        tokens, arrivals, texts = [], [], []
        done = False
        finish = stop_reason = usage = None
        request_id = None
        req = Request(base+'/v1/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type':'application/json'})
        with opener.open(req, timeout=30) as response, (args.output/f'{name}-events.jsonl').open('w') as raw:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                event = json.loads(data)
                elapsed = time.perf_counter()-started
                raw.write(json.dumps({'seconds':elapsed,'event':event},ensure_ascii=False)+'\n')
                if 'error' in event:
                    raise RuntimeError(event['error'])
                request_id = event.get('id',request_id)
                usage = event.get('usage') or usage
                for choice in event.get('choices',[]):
                    ids = choice.get('token_ids') or []
                    if ids:
                        arrivals.append((elapsed,len(ids)))
                        tokens.extend(ids)
                    texts.append(choice.get('text',''))
                    finish = choice.get('finish_reason') or finish
                    if choice.get('stop_reason') is not None:
                        stop_reason = choice['stop_reason']
        total = time.perf_counter()-started
        if not done or not usage or len(tokens)!=usage['completion_tokens'] or not finish:
            raise RuntimeError(f'Incomplete SSE response: {name}')
        first,first_count = arrivals[0]
        last = arrivals[-1][0]
        decode_tps = (len(tokens)-first_count)/(last-first) if last>first else None
        text = ''.join(texts)
        (args.output/f'{name}-output.txt').write_text(text)
        row = {'language':language,'mode':mode,'usage':usage,'context_budget':budget,
               'finish_reason':finish,'stop_reason':stop_reason,'last_token_id':tokens[-1] if tokens else None,
               'eos_seen':2 in tokens,'ttft_seconds':first,'total_seconds':total,
               'decode_tokens_per_second':decode_tps,'end_to_end_tokens_per_second':len(tokens)/total,
               'request_id':request_id,'token_ids':tokens,'text':text}
        report['trials'].append(row)
        publish()
        print(name, 'input=',usage['prompt_tokens'],'output=',usage['completion_tokens'],
              'finish=',finish,'EOS=',row['eos_seen'],'TTFT=',round(first,3),
              'decode_tps=',round(decode_tps,2) if decode_tps else None,flush=True)

    monitor = None
    try:
        with opener.open(base+'/health',timeout=5) as response:
            assert response.status==200
        report['models'] = json_request('/v1/models')
        max_len = next(m['max_model_len'] for m in report['models']['data'] if m['id']==args.model)
        report['max_model_len'] = max_len
        with (args.output/'gpu.csv').open('w') as telemetry:
            monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,power.draw',
                                        '--format=csv','-l','2'],stdout=telemetry,stderr=subprocess.STDOUT)
            lengths = {}
            for language,prompt in PROMPTS.items():
                (args.output/f'{language}-prompt.txt').write_text(prompt)
                ids = json_request('/tokenize',{'model':args.model,'prompt':prompt,
                                               'add_special_tokens':True})['tokens']
                lengths[language] = len(ids)
                if max_len-len(ids)<16:
                    raise RuntimeError('Test prompt leaves insufficient generation space')
            report['prompt_tokens'] = lengths
            publish()
            for mode in ('omitted','null','context_budget'):
                trial('en',mode,PROMPTS['en'],max_len-lengths['en'])
            trial('zh','context_budget',PROMPTS['zh'],max_len-lengths['zh'])
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        if monitor:
            monitor.terminate(); monitor.wait(timeout=5)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        publish()


if __name__=='__main__':
    main()
