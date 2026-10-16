#!/usr/bin/env python3
"""Validate and benchmark full greedy candidate verification on an OPT GPU model.

Run with the SpecFerry Conda interpreter and the pinned container GPU libraries.
This is a Transformers/SDPA reference backend, NOT the vLLM serving engine.
No network, new model, training, quantization or NP101 execution is involved.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from specferry.verification import GreedyVerifier


def publish(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class OPTBackend:
    def __init__(self, model, torch):
        self.model = model
        self.torch = torch

    def cache_length(self, cache):
        return 0 if cache is None else int(cache.get_seq_length())

    def crop(self, cache, length):
        if cache is None:
            if length:
                raise ValueError('Cannot crop an absent cache to a positive length')
        else:
            remove = self.cache_length(cache) - length
            if remove < 0:
                raise ValueError('Cannot extend cache by cropping')
            if remove:
                cache.crop(-remove)

    def prefill(self, tokens):
        if not tokens:
            return None
        ids = self.torch.tensor([tokens], device='cuda', dtype=self.torch.long)
        out = self.model.model.decoder(input_ids=ids, use_cache=True, return_dict=True)
        self.torch.cuda.synchronize()
        return out.past_key_values

    def forward(self, tokens, cache):
        torch = self.torch
        started = time.perf_counter()
        ids = torch.tensor([tokens], device='cuda', dtype=torch.long)
        attention = torch.ones((1, self.cache_length(cache) + len(tokens)),
                               device='cuda', dtype=torch.long)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        out = self.model.model.decoder(input_ids=ids, attention_mask=attention,
                                      past_key_values=cache, use_cache=True, return_dict=True)
        events[1].record()
        # Project EVERY candidate position, unlike the previous HTTP chunk proxy.
        logits = self.model.lm_head(out.last_hidden_state)
        events[2].record()
        predictions = logits.argmax(dim=-1)
        events[3].record()
        events[3].synchronize()
        ids_out = predictions[0].cpu().tolist()
        return ids_out, out.past_key_values, {
            'decoder_gpu_seconds': events[0].elapsed_time(events[1]) / 1000,
            'lm_head_gpu_seconds': events[1].elapsed_time(events[2]) / 1000,
            'argmax_gpu_seconds': events[2].elapsed_time(events[3]) / 1000,
            'model_and_select_wall_seconds': time.perf_counter() - started,
            'logit_shape': list(logits.shape),
        }


def reference_greedy(model, torch, prompt, count):
    """Independent ordinary OPTForCausalLM.forward path; no verifier calls."""
    inputs = torch.tensor([prompt], dtype=torch.long, device='cuda')
    cache = None
    generated = []
    for _ in range(count):
        output = model(input_ids=inputs, past_key_values=cache, use_cache=True, return_dict=True)
        token = int(output.logits[0, -1].argmax().item())
        generated.append(token)
        cache = output.past_key_values
        inputs = torch.tensor([[token]], dtype=torch.long, device='cuda')
    return generated


def wrong_token(token, vocab_size):
    return (token + 1) % vocab_size


def correctness(backend, tokenizer, model, torch, output_dir):
    records = []
    prompts = ['The solar system consists of',
               'In a quiet village near the mountains, there lived a',
               'To make a simple vegetable soup, first prepare']
    for text in prompts:
        prefix = tokenizer.encode(text)
        expected = reference_greedy(model, torch, prefix, 64)
        base = GreedyVerifier(backend, prefix)
        # Cache rollback is checked by generating four further ordinary tokens.
        for case in ('all_accept', 'first_reject', 'middle_reject', 'output_budget', 'no_draft'):
            base.rewind_for_benchmark(prefix)
            draft = expected[:8].copy() if case != 'no_draft' else []
            if case == 'first_reject':
                draft[0] = wrong_token(draft[0], model.config.vocab_size)
            if case == 'middle_reject':
                draft[3] = wrong_token(draft[3], model.config.vocab_size)
            result = base.step(draft, max_output=3 if case == 'output_budget' else None)
            emitted = result['output_tokens'].copy()
            for _ in range(4):
                emitted.extend(base.step([])['output_tokens'])
            if emitted != expected[:len(emitted)]:
                raise RuntimeError(f'Greedy mismatch: {case}, prompt={text!r}, got={emitted}, expected={expected}')
            records.append({'prompt': text, 'case': case, 'status': 'pass',
                            'round': result, 'checked_tokens': emitted})
        # The protocol's stop-ID handling also runs against real GPU predictions.
        # This is an explicit synthetic stop ID, NOT a naturally generated EOS.
        base.rewind_for_benchmark(prefix)
        base.eos_token_id = expected[2]
        result = base.step(expected[:8])
        stop_index = expected.index(expected[2])
        if not result['stopped_on_eos'] or result['output_tokens'] != expected[:stop_index + 1]:
            raise RuntimeError('Synthetic stop-ID check failed')
        records.append({'prompt': text, 'case': 'synthetic_stop_id', 'status': 'pass',
                        'synthetic_stop_id': expected[2], 'round': result})
        base.eos_token_id = None
        base.rewind_for_benchmark(prefix)
        emitted = []
        rounds = []
        while len(emitted) < 48:
            k = min(8, 48 - len(emitted) - 1)
            draft = expected[len(emitted):len(emitted) + k].copy()
            if draft and len(rounds) % 3:
                index = 0 if len(rounds) % 3 == 1 else len(draft) // 2
                draft[index] = wrong_token(draft[index], model.config.vocab_size)
            result = base.step(draft, max_output=48 - len(emitted))
            emitted.extend(result['output_tokens'])
            rounds.append(result)
            if emitted != expected[:len(emitted)]:
                raise RuntimeError(f'Multi-round mismatch at round {len(rounds)}')
        records.append({'prompt': text, 'case': 'multi_round', 'status': 'pass',
                        'output_tokens': emitted, 'rounds': rounds})
        publish(output_dir / 'correctness.json', {'status': 'running', 'cases': records})
        print(f'Correctness passed: {text}', flush=True)
    publish(output_dir / 'correctness.json', {'status': 'pass', 'cases': records,
        'comparison': f'Exact token IDs versus independent OPTForCausalLM.forward greedy loop, same {model.dtype}/SDPA model.',
        'scope': 'Three prompts; synthetic candidate corruption; 48-token multi-round sequences. Not a proof across all inputs.',
        'eos_scope': 'Real EOS ID 2 control-flow cases are unit-tested with a deterministic backend. GPU stop-ID test is synthetic.'})


def benchmark(backend, tokenizer, model, torch, output_dir, repeats):
    source = tokenizer.encode('A research team studies efficient language model inference. ' * 220)
    rows = []
    rng = random.Random(20261015)
    for context in (128, 512, 1536):
        prefix = source[:context]
        expected = reference_greedy(model, torch, prefix, 40)
        verifier = GreedyVerifier(backend, prefix)
        shapes = [(k, case) for k in (0, 1, 2, 4, 8, 16, 32)
                  for case in (('all_accept',) if k == 0 else ('all_accept', 'first_reject', 'middle_reject'))]
        samples = {key: [] for key in shapes}

        def trial(k, case):
            verifier.rewind_for_benchmark(prefix)
            draft = expected[:k].copy()
            if case != 'all_accept':
                index = 0 if case == 'first_reject' else k // 2
                draft[index] = wrong_token(draft[index], model.config.vocab_size)
            result = verifier.step(draft)
            result['exact_reference_match'] = result['output_tokens'] == expected[:len(result['output_tokens'])]
            result['reference_tokens'] = expected[:len(result['output_tokens'])]
            if result['logit_shape'][1] != k + 1:
                raise RuntimeError('Missing candidate-position logits')
            return result

        for k, case in shapes:
            for _ in range(2):
                trial(k, case)
        order = shapes * repeats
        rng.shuffle(order)
        for k, case in order:
            samples[(k, case)].append(trial(k, case))
        for (k, case), trials in samples.items():
            fields = ('total_seconds', 'decoder_gpu_seconds', 'lm_head_gpu_seconds',
                      'argmax_gpu_seconds', 'model_and_select_wall_seconds',
                      'decision_seconds', 'rollback_seconds')
            row = {'context_tokens': context, 'draft_tokens': k, 'case': case,
                   'medians': {field: statistics.median(t[field] for t in trials) for field in fields},
                   'min_total_seconds': min(t['total_seconds'] for t in trials),
                   'max_total_seconds': max(t['total_seconds'] for t in trials), 'trials': trials}
            rows.append(row)
            if case == 'all_accept':
                print(f'Full verification context={context} K={k}: {row["medians"]["total_seconds"]*1000:.2f} ms', flush=True)
        publish(output_dir / 'verification-benchmark.json', {'status': 'running', 'rows': rows})
    publish(output_dir / 'verification-benchmark.json', {'status': 'pass', 'rows': rows,
        'backend': f'Transformers OPT, {model.dtype}, torch SDPA, eager execution; no vLLM kernels or CUDA graphs.',
        'correctness_scope': 'Timing only. Per-trial exact_reference_match records numerical disagreements; this status is not a correctness pass.',
        'method': 'Cached prefix, all K+1-position LM-head logits, greedy argmax, accept/reject and KV crop. Two warmups per case, shuffled trials.',
        'excludes': 'Initial prefill, external DLM, network; input preparation and CPU decision are included in total wall time.'})
    return rows


def simulate(rows, output_dir):
    records = []
    for row in rows:
        if row['case'] != 'all_accept' or not row['draft_tokens']:
            continue
        context, k = row['context_tokens'], row['draft_tokens']
        ordinary = next(x for x in rows if x['context_tokens'] == context and x['draft_tokens'] == 0)
        target_tps = 1 / ordinary['medians']['total_seconds']
        verify = row['medians']['total_seconds']
        for rate in (2, 10, 20, 50, 100):
            for a in (0.5, 0.8, 0.95, 1.0):
                for rtt in (0, .01, .05):
                    length = sum(a**i for i in range(k + 1))
                    throughput = length / (k/rate + verify + rtt)
                    records.append({'context_tokens': context, 'K': k, 'draft_rate': rate,
                        'conditional_acceptance': a, 'rtt_seconds': rtt, 'expected_output': length,
                        'estimated_tokens_per_second': throughput, 'target_only_tokens_per_second': target_tps,
                        'speedup_same_backend': throughput / target_tps})
    publish(output_dir / 'simulation.json', {'status': 'complete', 'scenarios': records,
        'type': 'Analytical estimate based on full-verifier measurements, hypothetical draft rate and acceptance.',
        'limitations': 'Serial cycle; excludes DLM catch-up and startup. Uses all-accepted verification median for every acceptance rate. No NP101 performance claim.'})


def timed_mock(backend, tokenizer, model, torch, output_dir):
    prefix = tokenizer.encode('In a quiet village near the mountains, there lived a')
    expected = reference_greedy(model, torch, prefix, 48)
    records = []
    for rate in (2, 20, 50, 100):
        for acceptance in (1.0, .8):
            verifier = GreedyVerifier(backend, prefix)
            rng = random.Random(410 + int(acceptance * 100))
            emitted, rounds = [], []
            started = time.perf_counter()
            while len(emitted) < 24:
                k = min(4, 24 - len(emitted) - 1)
                draft = expected[len(emitted):len(emitted) + k].copy()
                for i in range(k):
                    if rng.random() > acceptance:
                        draft[i] = wrong_token(draft[i], model.config.vocab_size)
                        break
                wait_start = time.perf_counter()
                time.sleep(k / rate)
                # A 10 ms simulated round trip; no actual network measurement.
                time.sleep(.01)
                waited = time.perf_counter() - wait_start
                result = verifier.step(draft, max_output=24 - len(emitted))
                result['simulated_wait_actual_seconds'] = waited
                rounds.append(result)
                emitted.extend(result['output_tokens'])
                if emitted != expected[:len(emitted)]:
                    raise RuntimeError('Mock-DLM loop changed target greedy tokens')
            duration = time.perf_counter() - started
            records.append({'draft_rate': rate, 'assumed_acceptance': acceptance, 'K': 4,
                'simulated_rtt_seconds': .01, 'actual_wall_seconds': duration,
                'output_tokens': emitted, 'measured_tokens_per_second': len(emitted) / duration,
                'rounds': rounds})
            publish(output_dir / 'timed-mock.json', {'status': 'running', 'scenarios': records})
            print(f'Mock DLM rate={rate}, acceptance={acceptance}: {len(emitted)/duration:.2f} tokens/s', flush=True)
    publish(output_dir / 'timed-mock.json', {'status': 'pass', 'scenarios': records,
        'type': 'Real GPU verifier with sleep-based synthetic drafting/RTT and precomputed oracle candidates.',
        'excludes': 'Oracle preparation and initial prefill. This is not a real DLM, real network, or NP101 run.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--dtype', choices=('float16', 'float32'), default='float16')
    parser.add_argument('--phase', choices=('all', 'correctness', 'performance'), default='all',
                        help='Performance-only never implies that correctness passed.')
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('--repeats must be at least 3')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
               'backend': f'transformers-opt-sdpa-{args.dtype}', 'np101_execution': False,
               'requested_phase': args.phase,
               'vllm_verification_integration': False, 'phases': []}
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, OPTForCausalLM
        if sys.version_info[:2] != (3, 12) or Path(sys.prefix).name != 'SpecFerry':
            raise RuntimeError('Use the Python 3.12 SpecFerry Conda environment')
        manifest = json.loads((args.model / 'download-manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['repo_id'] != 'facebook/opt-6.7b':
            raise RuntimeError('Require the verified, authorized OPT-6.7B checkpoint')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        summary.update({'python': sys.version, 'conda_prefix': sys.prefix,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'numpy': importlib.metadata.version('numpy'), 'device': torch.cuda.get_device_name(),
            'device_capability': list(torch.cuda.get_device_capability()),
            'model_revision': manifest['resolved_revision'], 'repeats': args.repeats})
        publish(args.output / 'summary.json', summary)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = OPTForCausalLM.from_pretrained(args.model, local_files_only=True,
                    dtype=getattr(torch, args.dtype), attn_implementation='sdpa').to('cuda').eval()
        backend = OPTBackend(model, torch)
        print('Loaded OPT-6.7B on', summary['device'], flush=True)
        with torch.inference_mode():
            if args.phase in ('all', 'correctness'):
                correctness(backend, tokenizer, model, torch, args.output)
                summary['phases'].append('correctness_pass'); publish(args.output / 'summary.json', summary)
            if args.phase in ('all', 'performance'):
                rows = benchmark(backend, tokenizer, model, torch, args.output, args.repeats)
                summary['phases'].append('benchmark_complete'); publish(args.output / 'summary.json', summary)
                simulate(rows, args.output)
                timed_mock(backend, tokenizer, model, torch, args.output)
                summary['phases'].extend(['analytical_simulation_complete', 'timed_mock_pass'])
        summary['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        summary['status'] = 'pass'
    except Exception as error:
        summary.update(status='failed', error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    summary['finished_at'] = datetime.now(timezone.utc).isoformat()
    publish(args.output / 'summary.json', summary)
    return 0 if summary['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
