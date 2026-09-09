"""CPU reference validation: official prefill, sequential prompt consumption, and state traces."""
from contextlib import contextmanager
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as impl

from .checkpoint import inventory, REVISION, sha256
from .model import load_text_model, load_tokenizer, encode, source_record
from .trace import Capture, cache_tensors, compare, finite, save_arrays, tensor_info

TOLERANCES = json.loads(Path(__file__).with_name('tolerances.json').read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


@contextmanager
def precision(mode):
    original = impl.l2norm
    if mode == 'deployment-fp16':
        def l2_fp32(x, dim=-1, eps=1e-6):
            value = x.float()
            return (value * torch.rsqrt(value.square().sum(dim=dim, keepdim=True) + eps)).to(x.dtype)
        impl.l2norm = l2_fp32
    try:
        yield
    finally:
        impl.l2norm = original


def forward(model, ids, cache=None, keep=1):
    result = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=keep)
    finite(result.logits, 'logits')
    return result


def generate(model, tokenizer, prompt_ids, sequential, count, directory):
    cache, outputs, times, logits, shapes = None, [], [], [], []
    start = time.perf_counter()
    if sequential:
        for position in range(prompt_ids.shape[1]):
            result = forward(model, prompt_ids[:, position:position + 1], cache)
            cache = result.past_key_values
    else:
        result = forward(model, prompt_ids)
        cache = result.past_key_values
    prefill_seconds = time.perf_counter() - start
    eos = model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for step in range(count):
        logits.append(result.logits[:, -1].detach().clone())
        token = int(result.logits[0, -1].argmax())
        outputs.append(token)
        shapes.append({name: tensor_info(value) for name, value in cache_tensors(cache).items()})
        if token in eos or step + 1 == count:
            break
        start = time.perf_counter()
        result = forward(model, torch.tensor([[token]]), cache)
        cache = result.past_key_values
        times.append(time.perf_counter() - start)
    text = tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if not text.strip() or '\ufffd' in text:
        raise ValueError(f'empty or invalid generated text: {text!r}')
    save_arrays(directory / 'logits.npz', {str(i): value for i, value in enumerate(logits)})
    result = {'input_ids': prompt_ids[0].tolist(), 'output_ids': outputs, 'output_text': text,
              'prompt_mode': 'sequential' if sequential else 'official-prefill',
              'prefill_seconds': prefill_seconds, 'decode_seconds': times,
              'cache_shapes': shapes, 'consumed_length': prompt_ids.shape[1] + len(outputs) - 1,
              'eos_reached': outputs[-1] in eos,
              'quality_check': 'nonempty, valid Unicode; semantic output is retained for human review'}
    write(directory / 'generation.json', result)
    return result, logits


def alignment(model, ids, directory, mode, capture_intermediates=True):
    checkpoints = (1, 2, 4, 8)
    capture = Capture()
    cache, seq, seq_logits = None, {}, []
    from contextlib import nullcontext
    with capture.attach(model) if capture_intermediates else nullcontext():
        for i in range(8):
            capture.step = i + 1
            result = forward(model, ids[:, i:i + 1], cache)
            cache = result.past_key_values
            seq_logits.append(result.logits.detach().clone())
            if i + 1 in checkpoints:
                seq[i + 1] = cache_tensors(cache)
                if capture_intermediates:
                    capture.save(cache_tensors(cache, (0, 3)), 'cache')
    if capture_intermediates:
        save_arrays(directory / 'layer-0-3-sequential.npz', capture.values)
        write(directory / 'trace-index.json', {k: tensor_info(v) for k, v in capture.values.items()})
        fixed = fixed_input_recurrence(capture.values)
        write(directory / 'fixed-input-recurrence.json', fixed)
        if not all(x['state']['pass'] and x['output']['pass'] for x in fixed):
            raise ValueError('fixed-input recurrent reference comparison failed')
    results = []
    for count in checkpoints:
        # No custom prefill: invoke the unmodified official model forward.
        output = forward(model, ids[:, :count], keep=0)
        normal = cache_tensors(output.past_key_values)
        if normal.keys() != seq[count].keys():
            raise ValueError('prefill and sequential paths expose different cache fields')
        reference_logits = torch.cat(seq_logits[:count], dim=1)
        budget = TOLERANCES['fp16_pipeline' if mode == 'deployment-fp16' else 'fp32_pipeline']
        logit_result = compare(output.logits, reference_logits, **budget)
        states = {}
        for name, value in normal.items():
            state_budget = (TOLERANCES['fp32_state_from_fp16_pipeline']
                            if mode == 'deployment-fp16' and value.dtype == torch.float32 else budget)
            states[name] = compare(value, seq[count][name], **state_budget)
        save_arrays(directory / f'prefill-{count}-states.npz', normal)
        save_arrays(directory / f'sequential-{count}-states.npz', seq[count])
        results.append({'tokens': count, 'logits': logit_result, 'states': states,
                        'pass': logit_result['pass'] and all(s['pass'] for s in states.values())})
    write(directory / 'prefill-alignment.json', results)
    return results


def fixed_input_recurrence(values):
    """Check the documented recurrence against real official function IO.

    Each case receives identical saved Q/K/V, gates, and previous state, so it
    must meet the small FP32 operator budget even in an FP16 model pipeline.
    """
    results = []
    for step in (2, 4, 8):
        prefix = f'token.{step:04d}.layer.0.torch_recurrent_gated_delta_rule'
        q, k, v = (values[f'{prefix}.input.{i}'] for i in range(3))
        q, k = impl.l2norm(q), impl.l2norm(k)
        q, k, v = (x[:, 0].float() for x in (q, k, v))
        q = q / q.shape[-1] ** .5
        decay = values[f'{prefix}.kwargs.g'][:, 0].float().exp()
        beta = values[f'{prefix}.kwargs.beta'][:, 0].float()
        previous = values[f'{prefix}.kwargs.initial_state']
        decayed = previous.float() * decay[..., None, None]
        correction = beta[..., None] * (v - torch.einsum('bhk,bhkv->bhv', k, decayed))
        state = decayed + torch.einsum('bhk,bhv->bhkv', k, correction)
        expected_output = values[f'{prefix}.output.0']
        out = torch.einsum('bhk,bhkv->bhv', q, state)[:, None].to(expected_output.dtype)
        results.append({'consumed_tokens': step,
                        'state': compare(state, values[f'{prefix}.output.1'], **TOLERANCES['fixed_input_fp32_state']),
                        'output': compare(out, expected_output, **TOLERANCES['fixed_input_fp16_output'])})
    return results


def teacher_forcing(model, input_ids, forced_ids, directory):
    output = forward(model, input_ids)
    cache = output.past_key_values
    saved = {'logits.0': output.logits.detach().clone()}
    for i, token in enumerate(forced_ids):
        output = forward(model, torch.tensor([[token]]), cache)
        cache = output.past_key_values
        saved[f'logits.{i + 1}'] = output.logits.detach().clone()
    saved.update(cache_tensors(cache, (0, 3)))
    save_arrays(directory / 'teacher-forcing.npz', saved)
    write(directory / 'teacher-forcing.json', {'input_ids': input_ids[0].tolist(), 'forced_ids': forced_ids,
                                               'consumed_length': input_ids.shape[1] + len(forced_ids)})
    return saved


@torch.inference_mode()
def run(root, output, threads=8, max_new_tokens=32):
    output.mkdir(parents=True, exist_ok=True)
    completion = output / 'reference-summary.json'
    completion.unlink(missing_ok=True)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    metadata = inventory(root)
    write(output / 'checkpoint-inventory.json', metadata)
    tokenizer = load_tokenizer(root)
    cases = [('chinese', '中国的首都是哪里？请只回答城市名。'), ('english', 'What is 2 + 3? Answer briefly.')]
    source = source_record()
    shutil.copy2(source['path'], output / 'modeling_qwen3_5.reference.py')
    policy = {
        'revision': REVISION, 'device': 'host-cpu', 'threads': threads,
        'source': source, 'attention': 'eager', 'external_kernels': False,
        'local_source_sha256': {name: sha256(Path(__file__).with_name(name)) for name in
                               ('runner.py', 'model.py', 'trace.py', 'checkpoint.py', 'tolerances.json')},
        'original': 'Native BF16 values expanded losslessly to FP32; all CPU arithmetic uses the official FP32 model.',
        'deployment': {
            'weights': 'BF16 -> FP16; native F32 A_log and gated-norm weights stay FP32.',
            'projections_conv_mlp_residual_kv': 'FP16 inputs and stored outputs; PyTorch CPU kernel accumulation.',
            'main_rmsnorm_qk_rmsnorm': 'FP32 square/mean/rsqrt and (1 + weight); cast output back to input dtype.',
            'qk_l2norm': 'Explicit FP32 square/sum/rsqrt; cast normalized Q/K back to FP16.',
            'delta_recurrence': 'FP32 decay, state, dot products and updates; core output rounded to FP16.',
            'gated_norm': 'FP32 reduction; follow installed reference cast before weight and FP32 SiLU(gate).',
            'attention_softmax_rope_constants': 'FP32 sensitive computation; official cast boundaries.',
        },
        'sampling': {'greedy': True, 'enable_thinking': False, 'logits_processors': []},
        'dependencies': {name: importlib.metadata.version(name) for name in
                         ('torch', 'transformers', 'numpy', 'safetensors', 'tokenizers', 'huggingface_hub')},
    }
    write(output / 'reference-policy.json', policy)
    write(output / 'tolerances.json', TOLERANCES)
    summaries, failures = [], []
    original_generations = {}
    for mode in ('original-fp32', 'deployment-fp16'):
        print(f'Loading {mode}', flush=True)
        model = load_text_model(root, metadata, mode)
        directory = output / 'reference-trace' / mode
        directory.mkdir(parents=True, exist_ok=True)
        with precision(mode):
            rendered, check_ids = encode(tokenizer, cases[0][1])
            comparisons = alignment(model, check_ids, directory, mode)
            if not all(item['pass'] for item in comparisons):
                failures.append(mode + ': frozen prefill/state tolerance exceeded')
            for case_name, prompt in cases:
                case_dir = directory / case_name
                case_dir.mkdir(exist_ok=True)
                rendered, ids = encode(tokenizer, prompt)
                if case_name == 'english':
                    comparisons = alignment(model, ids, case_dir, mode, capture_intermediates=False)
                    if not all(item['pass'] for item in comparisons):
                        failures.append(mode + ': English holdout prefill/state tolerance exceeded')
                if ids.shape[1] > 256 or ids.shape[1] + max_new_tokens > 512:
                    raise ValueError('CPU reference limits exceeded: at most 256 prompt tokens and 512 total tokens')
                write(case_dir / 'input.json', {'prompt': prompt, 'rendered': rendered,
                                               'input_ids': ids[0].tolist(), 'enable_thinking': False})
                generated, logits = generate(model, tokenizer, ids, mode == 'deployment-fp16',
                                             max_new_tokens, case_dir)
                expected_answer = '北京' if case_name == 'chinese' else '5'
                generated['basic_answer_pass'] = expected_answer in generated['output_text']
                if not generated['basic_answer_pass']:
                    failures.append(mode + ': basic answer check failed for ' + case_name)
                # Validate our greedy loop against official GenerationMixin, without custom processors.
                official = model.generate(ids, attention_mask=torch.ones_like(ids),
                                          max_new_tokens=min(4, max_new_tokens), do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                          repetition_penalty=1.0)
                generated['official_generate_first_ids'] = official[0, ids.shape[1]:].tolist()
                generated['official_prefix_agrees'] = generated['output_ids'][:len(generated['official_generate_first_ids'])] == generated['official_generate_first_ids']
                if mode == 'original-fp32' and not generated['official_prefix_agrees']:
                    failures.append(mode + ': custom greedy loop disagrees with official generate')
                if mode == 'original-fp32':
                    original_generations[case_name] = generated['output_ids'][:4]
                forced = teacher_forcing(model, ids, original_generations[case_name], case_dir)
                if mode == 'deployment-fp16':
                    prior = output / 'reference-trace' / 'original-fp32' / case_name / 'teacher-forcing.npz'
                    with np.load(prior, allow_pickle=False) as original:
                        cross = {}
                        for key, actual in forced.items():
                            baseline = torch.from_numpy(original[key]).float()
                            diff = actual.float() - baseline
                            cross[key] = {'max_abs_error': diff.abs().max().item(),
                                          'rmse': diff.square().mean().sqrt().item()}
                            if key.startswith('logits.'):
                                cross[key]['top1_agrees'] = actual.argmax().item() == baseline.argmax().item()
                    write(case_dir / 'cross-precision-teacher-forcing.json', cross)
                write(case_dir / 'generation.json', generated)
                print(f'{mode} / {case_name}: {generated["output_text"]}', flush=True)
                summaries.append({'mode': mode, 'case': case_name, 'output_text': generated['output_text'],
                                  'output_ids': generated['output_ids'], 'eos_reached': generated['eos_reached'],
                                  'basic_answer_pass': generated['basic_answer_pass'],
                                  'official_prefix_agrees': generated['official_prefix_agrees']})
        del model
        gc.collect()
    tolerances = {**TOLERANCES, 'holdout_validation_passed': not failures,
                  'cross_precision_note': 'Pipeline budgets do not replace fixed-input operator budgets or promise identical cross-precision generation.'}
    write(output / 'tolerances.json', tolerances)
    result = {'status': 'passed' if not failures else 'needs_attention', 'revision': REVISION,
              'hardware_execution': False, 'models_downloaded': 1, 'cases': summaries, 'failures': failures}
    write(completion, result)
    return 0 if not failures else 1
