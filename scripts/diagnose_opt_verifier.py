"""Compare reference and post-rollback KV/logits at a known FP16 divergence."""
import argparse
import json
from pathlib import Path

from validate_opt_verifier import OPTBackend, GreedyVerifier, reference_greedy, wrong_token


def main():
    import torch
    from transformers import AutoTokenizer, OPTForCausalLM
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = OPTForCausalLM.from_pretrained(args.model, local_files_only=True,
              dtype=getattr(torch, args.dtype), attn_implementation='sdpa').to('cuda').eval()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = tok.encode('To make a simple vegetable soup, first prepare')
    backend = OPTBackend(model, torch)
    rows = []
    with torch.inference_mode():
        expected = reference_greedy(model, torch, prompt, 16)
        v = GreedyVerifier(backend, prompt)
        draft = expected[:8].copy(); draft[3] = wrong_token(draft[3], model.config.vocab_size)
        result = v.step(draft)
        assert result['output_tokens'] == expected[:4]
        ref = model(input_ids=torch.tensor([prompt], device='cuda'), use_cache=True)
        ref_cache = ref.past_key_values
        for token in expected[:3]:
            ref = model(input_ids=torch.tensor([[token]], device='cuda'),
                        past_key_values=ref_cache, use_cache=True)
            ref_cache = ref.past_key_values
        actual_cache = v.cache
        for index in range(4, 10):
            kv_error = max(float((a.keys - b.keys).abs().max().item())
                           for a, b in zip(actual_cache.layers, ref_cache.layers))
            vv_error = max(float((a.values - b.values).abs().max().item())
                           for a, b in zip(actual_cache.layers, ref_cache.layers))
            ids = torch.tensor([[expected[index - 1]]], device='cuda')
            out_a = model(input_ids=ids, past_key_values=actual_cache, use_cache=True)
            out_b = model(input_ids=ids, past_key_values=ref_cache, use_cache=True)
            actual_cache, ref_cache = out_a.past_key_values, out_b.past_key_values
            a, b = out_a.logits[0, -1].float(), out_b.logits[0, -1].float()
            va, ia = a.topk(5); vb, ib = b.topk(5)
            row = {'generated_index': index, 'cache_length': backend.cache_length(actual_cache),
                   'max_key_abs_difference': kv_error, 'max_value_abs_difference': vv_error,
                   'max_logits_abs_difference': float((a-b).abs().max().item()),
                   'actual_argmax': int(a.argmax()), 'reference_argmax': int(b.argmax()),
                   'actual_top5': list(zip(ia.cpu().tolist(),va.cpu().tolist())),
                   'reference_top5': list(zip(ib.cpu().tolist(),vb.cpu().tolist())),
                   'reference_margin': float(vb[0]-vb[1]),
                   'actual_margin': float(va[0]-va[1])}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({'dtype':args.dtype, 'expected':expected,
                                     'rejection_round':result, 'rows':rows},indent=2)+'\n')


if __name__ == '__main__':
    main()
