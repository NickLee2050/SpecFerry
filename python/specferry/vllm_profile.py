"""Local measurement hook required by vLLM's proposer ABI, not a scheduler.

The interval between returning drafts and entering the next proposer includes
native verification, sampling, bookkeeping, scheduler/IPC and device waiting.
It is a target-round wall time, NOT an isolated CUDA kernel measurement.
"""
import ctypes as C
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from .vllm_proposer import ExternalDraftProposer


class ProfiledDraftProposer(ExternalDraftProposer):
    def __init__(self, vllm_config):
        if urlsplit(os.environ.get('SPECFERRY_DLM_URL', '')).hostname not in ('127.0.0.1', '::1'):
            raise ValueError('Profiling may only contact a literal loopback fixture')
        super().__init__(vllm_config)
        self.maximum_k = self.k
        self.control = Path(os.environ['SPECFERRY_PROFILE_CONTROL'])
        self.audit = Path(os.environ['SPECFERRY_PROFILE_AUDIT'])
        self.pending = None
        self.native.sf_draft_last_result.argtypes = [C.c_void_p]
        self.native.sf_draft_last_result.restype = C.c_char_p

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        import torch
        torch.cuda.synchronize()
        arrived = time.perf_counter()
        control = json.loads(self.control.read_text())
        if control['source'] not in ('disabled', 'external'):
            raise ValueError('Invalid local profiling source')
        k = control['draft_tokens']
        if type(k) is not int or not 1 <= k <= self.maximum_k:
            raise ValueError('Invalid local profiling draft length')
        self.disabled = control['source'] == 'disabled'
        self.k = k
        if len(sampled_token_ids) != 1:
            raise RuntimeError('Round measurements require exactly one sequence')
        length = int(num_tokens_no_spec[0])
        prefix = [int(x) for x in token_ids_cpu[0][:length]]
        sampled = [int(x) for x in sampled_token_ids[0]]
        row = {'run_id': control['run_id'], 'source': control['source'],
               'prefix_tokens': length, 'sampled_token_ids': sampled,
               'target_round': None}
        previous = self.pending
        if previous and previous['run_id'] == control['run_id']:
            delta = length - len(previous['prefix'])
            if delta != len(sampled) or prefix[:len(previous['prefix'])] != previous['prefix']:
                raise RuntimeError('Authoritative prefix did not advance by sampled output')
            accepted = len(sampled) - 1
            if not 0 <= accepted <= len(previous['draft']):
                raise RuntimeError('Invalid sampled length for the preceding draft')
            if sampled[:accepted] != previous['draft'][:accepted]:
                raise RuntimeError('Accepted prefix differs from supplied candidate IDs')
            row['target_round'] = {
                'prefix_tokens': len(previous['prefix']),
                'draft_tokens': len(previous['draft']),
                'accepted_prefix_tokens': accepted,
                'sampled_tokens_before_output_budget': len(sampled),
                'seconds': arrived - previous['ended'],
            }
        started = time.perf_counter()
        drafts = super().propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings)
        row['draft_adapter_seconds'] = time.perf_counter() - started
        row['draft_token_ids'] = drafts[0]
        row['native_http'] = (None if self.disabled else
                              json.loads(self.native.sf_draft_last_result(self.handle)))
        with self.audit.open('a') as stream:
            stream.write(json.dumps(row, separators=(',', ':'))+'\n')
        # Exclude control parsing, callback and audit I/O from the next interval.
        self.pending = {'run_id': control['run_id'], 'prefix': prefix,
                        'draft': drafts[0], 'ended': time.perf_counter()}
        return drafts
