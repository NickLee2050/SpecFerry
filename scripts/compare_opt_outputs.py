#!/usr/bin/env python3
"""Experiment script: compare the existing vLLM service with HF generate on OPT.

Uses the SpecFerry Conda interpreter and existing pinned container GPU packages.
No production inference logic, downloads, or service configuration changes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:89154ef00dd15368d2b293c167e5cc7dbb521fcfb2fbb77510e0d4df2b820e8f'


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def worker(args):
    import torch
    import transformers
    from transformers import AutoTokenizer, GenerationConfig, OPTForCausalLM

    out = args.output
    model_path = Path('/workspace/.cache/models/facebook/opt-6.7b')
    manifest = json.loads((model_path / 'download-manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['repo_id'] != 'facebook/opt-6.7b':
        raise RuntimeError('Require the verified authorized checkpoint')
    opener = build_opener(ProxyHandler({}))

    def request(endpoint, payload):
        req = Request(args.base_url + endpoint, data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=300) as response:
            return json.load(response)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fast = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    slow = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    long_question = (out / 'long-question.txt').read_text()
    prompts = {
        'long-plain': long_question,
        'long-qa': 'Question: ' + long_question + '\n\nAnswer:',
        'short-qa': (
            'Question: What is the capital of France?\nAnswer: The capital of France is Paris.\n\n'
            'Question: Why do people use a refrigerator?\n'
            'Answer: A refrigerator keeps food cold so that it stays fresh longer.\n\n'
            'Question: How can I fairly compare the speed of two language model inference systems?\nAnswer:'
        ),
    }
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'checkpoint': manifest['repo_id'], 'revision': manifest['resolved_revision'],
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'device': torch.cuda.get_device_name(), 'dtype': 'float16',
              'hf_attention': 'sdpa', 'vllm': '0.28.0 existing ordinary TLM service',
              'generation': {'do_sample': False, 'temperature_vllm': 0,
                             'max_new_tokens': args.max_tokens, 'eos_token_id': 2,
                             'repetition_penalty': 1.0, 'stop_strings': []},
              'cases': [], 'scope': 'Quality/correctness controls; not a throughput benchmark.'}
    save(out / 'report.json', report)
    model = OPTForCausalLM.from_pretrained(model_path, local_files_only=True,
              dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    print('Loaded Transformers OPT FP16 on', report['device'], flush=True)
    config = GenerationConfig(do_sample=False, max_new_tokens=args.max_tokens,
                              bos_token_id=2, eos_token_id=2, pad_token_id=1,
                              repetition_penalty=1.0, use_cache=True)
    with torch.inference_mode():
        for name, text in prompts.items():
            (out / f'{name}-input.txt').write_text(text)
            ids = fast.encode(text, add_special_tokens=True)
            slow_ids = slow.encode(text, add_special_tokens=True)
            server_ids = request('/tokenize', {'model': 'specferry-tlm', 'prompt': text,
                                               'add_special_tokens': True})['tokens']
            if ids != slow_ids or ids != server_ids:
                save(out / f'{name}-tokenizer-mismatch.json',
                     {'fast': ids, 'slow': slow_ids, 'vllm': server_ids})
                raise RuntimeError(f'Tokenizer mismatch: {name}')
            save(out / f'{name}-input-token-ids.json', ids)
            payload = {'model': 'specferry-tlm', 'prompt': ids, 'add_special_tokens': False,
                       'temperature': 0, 'top_p': 1.0, 'top_k': -1,
                       'repetition_penalty': 1.0, 'presence_penalty': 0.0,
                       'frequency_penalty': 0.0, 'max_tokens': args.max_tokens,
                       'ignore_eos': False, 'return_token_ids': True,
                       'skip_special_tokens': True, 'cache_salt': uuid.uuid4().hex}
            save(out / f'{name}-vllm-request.json', payload)
            started = time.perf_counter()
            response = request('/v1/completions', payload)
            vllm_seconds = time.perf_counter() - started
            save(out / f'{name}-vllm-response.json', response)
            choice = response['choices'][0]
            v_ids = choice['token_ids']
            (out / f'{name}-vllm-output.txt').write_text(choice['text'])
            tensor = torch.tensor([ids], device='cuda', dtype=torch.long)
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                       generation_config=config, return_dict_in_generate=True,
                                       output_scores=True)
            torch.cuda.synchronize()
            hf_seconds = time.perf_counter() - started
            hf_raw_ids = generated.sequences[0, len(ids):].tolist()
            # Some serving versions omit the EOS ID from their returned output.
            def without_eos(tokens):
                return tokens[:-1] if tokens and tokens[-1] == 2 else tokens
            hf_ids, normalized_v = without_eos(hf_raw_ids), without_eos(v_ids)
            hf_text = fast.decode(hf_raw_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
            (out / f'{name}-transformers-output.txt').write_text(hf_text)
            first = next((i for i, (a, b) in enumerate(zip(normalized_v, hf_ids)) if a != b), None)
            row = {'name': name, 'prompt_tokens': len(ids), 'all_three_tokenizers_equal': True,
                   'vllm_seconds': vllm_seconds, 'transformers_seconds': hf_seconds,
                   'vllm_token_ids': v_ids, 'transformers_raw_token_ids': hf_raw_ids,
                   'vllm_finish_reason': choice['finish_reason'],
                   'transformers_finish_reason': 'eos' if hf_raw_ids[-1] == 2 else 'length',
                   'equal_token_ids_ignoring_terminal_eos': normalized_v == hf_ids,
                   'common_prefix_tokens': first if first is not None else min(len(hf_ids), len(normalized_v)),
                   'first_different_token_index': first,
                   'vllm_text': choice['text'], 'transformers_text': hf_text}
            if first is not None:
                values, indices = generated.scores[first][0].float().topk(5)
                scores = generated.scores[first][0].float()
                row['first_difference'] = {
                    'vllm_id': normalized_v[first], 'transformers_id': hf_ids[first],
                    'vllm_piece': fast.decode([normalized_v[first]]),
                    'transformers_piece': fast.decode([hf_ids[first]]),
                    'hf_top5_ids': indices.tolist(), 'hf_top5_logits': values.tolist(),
                    'hf_top1_minus_vllm_token_logit': float(values[0] - scores[normalized_v[first]])}
                prefix = ids + hf_ids[:first]
                diag_payload = dict(payload, prompt=prefix, max_tokens=1, logprobs=5,
                                    return_tokens_as_token_ids=True, cache_salt=uuid.uuid4().hex)
                save(out / f'{name}-first-difference-vllm.json',
                     request('/v1/completions', diag_payload))
                prefix_tensor = torch.tensor([prefix], device='cuda', dtype=torch.long)
                logits = model(prefix_tensor).logits[0, -1].float()
                values, indices = logits.topk(5)
                row['first_difference']['hf_reprefill_top5'] = {
                    'ids': indices.tolist(), 'logits': values.tolist()}
            report['cases'].append(row)
            save(out / 'report.json', report)
            print(name, 'tokenizers=equal', 'vllm=', len(v_ids), 'HF=', len(hf_raw_ids),
                  'equal=', row['equal_token_ids_ignoring_terminal_eos'],
                  'first_diff=', first, flush=True)
            del generated
    report.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
    save(out / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
        parser.error('Use SpecFerry Conda Python 3.12')
    if args.worker:
        worker(args)
        return 0
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / '.cache' / 'runs'):
        parser.error('Choose a new output directory under .cache/runs')
    out.mkdir(parents=True)
    shutil.copy2(ROOT / '.cache/runs/long-prompt-20261016/en-prompt.txt', out / 'long-question.txt')
    source = out / 'compare_opt_outputs.py'
    shutil.copy2(__file__, source)
    cmd = ['docker', 'run', '--rm', '--init', '--pull', 'never',
           '--name', f'specferry-compare-{os.getpid()}', '--network', 'host', '--gpus', 'all',
           '--user', f'{os.getuid()}:{os.getgid()}', '--shm-size', '2g',
           '--mount', f'type=bind,source={Path(sys.prefix).parent.parent},target={Path(sys.prefix).parent.parent},readonly',
           '--mount', f'type=bind,source={ROOT},target=/workspace,readonly',
           '--mount', f'type=bind,source={out},target=/results',
           '-e', 'PYTHONPATH=/usr/local/lib/python3.12/dist-packages',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HF_HUB_OFFLINE=1',
           '-e', 'TRANSFORMERS_OFFLINE=1', '-e', 'HF_HOME=/tmp/specferry-hf',
           '-e', 'XDG_CACHE_HOME=/tmp/specferry-cache', '--entrypoint', 'timeout',
           IMAGE, '1200', sys.executable, '/results/compare_opt_outputs.py',
           '--worker', '--output', '/results', '--base-url', args.base_url,
           '--max-tokens', str(args.max_tokens)]
    launch = ['sg', 'docker', '-c', shlex.join(cmd)]
    provenance = {'started_at': datetime.now(timezone.utc).isoformat(), 'command': launch,
                  'image_id': IMAGE, 'script_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    save(out / 'launch.json', provenance)
    with (out / 'run.log').open('w') as log:
        result = subprocess.run(launch, stdout=log, stderr=subprocess.STDOUT)
    provenance.update(exit_code=result.returncode, finished_at=datetime.now(timezone.utc).isoformat())
    save(out / 'launch.json', provenance)
    print((out / 'run.log').read_text(), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
