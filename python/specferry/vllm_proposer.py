"""Required vLLM 0.28.0 Python hook; transport and protocol live in C++."""
import atexit
import ctypes as C
import importlib.metadata
import os


class ExternalDraftProposer:
    def __init__(self, vllm_config):
        if importlib.metadata.version('vllm') != '0.28.0':
            raise RuntimeError('ExternalDraftProposer requires vLLM 0.28.0')
        config = vllm_config
        parallel = config.parallel_config
        if (parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
                or parallel.data_parallel_size != 1):
            raise ValueError('Require TP=PP=DP=1')
        if config.scheduler_config.max_num_seqs != 1 or config.scheduler_config.async_scheduling:
            raise ValueError('Require max_num_seqs=1 and --no-async-scheduling')
        if config.model_config.hf_config.model_type != 'opt':
            raise ValueError('Only OPT is enabled')
        self.k = config.speculative_config.num_speculative_tokens
        self.max_model_len = config.model_config.max_model_len
        self.handle = None
        source = os.environ.get('SPECFERRY_DRAFT_SOURCE', 'external')
        if source not in ('external', 'disabled'):
            raise ValueError('SPECFERRY_DRAFT_SOURCE must be external or disabled')
        self.disabled = source == 'disabled'
        if self.disabled:
            return
        self.native = C.CDLL(os.environ['SPECFERRY_DRAFT_LIBRARY'])
        self.native.sf_draft_create.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int,
                                                C.c_long, C.c_char_p]
        self.native.sf_draft_create.restype = C.c_void_p
        self.native.sf_draft_last_error.restype = C.c_char_p
        self.native.sf_draft_propose.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_size_t,
                                                C.c_size_t, C.POINTER(C.c_int32), C.c_size_t]
        self.native.sf_draft_propose.restype = C.c_int
        self.native.sf_draft_destroy.argtypes = [C.c_void_p]
        self.handle = self.native.sf_draft_create(
            str(config.model_config.tokenizer).encode(), os.environ['SPECFERRY_DLM_URL'].encode(),
            config.model_config.get_vocab_size(), self.max_model_len,
            round(float(os.environ.get('SPECFERRY_DLM_TIMEOUT', '30')) * 1000),
            os.environ.get('SPECFERRY_DLM_AUDIT', '').encode())
        if not self.handle:
            raise RuntimeError(self.native.sf_draft_last_error().decode())
        atexit.register(self.close)

    def close(self):
        if self.handle:
            self.native.sf_draft_destroy(self.handle)
            self.handle = None

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None):
        if self.disabled:
            return [[] for _ in sampled_token_ids]
        # vLLM 0.28 calls this after bookkeeping: sampled IDs are already in
        # token_ids_cpu. Binding conversion and this framework invariant stay here.
        proposed = []
        for i, sampled in enumerate(sampled_token_ids):
            length = int(num_tokens_no_spec[i])
            k = min(self.k, self.max_model_len - length)
            if not sampled or k <= 0:
                proposed.append([])
                continue
            prefix = [int(token) for token in token_ids_cpu[i][:length]]
            if prefix[-len(sampled):] != list(sampled):
                raise RuntimeError('vLLM committed-prefix ABI changed')
            inputs = (C.c_int32 * length)(*prefix)
            outputs = (C.c_int32 * k)()
            count = self.native.sf_draft_propose(self.handle, inputs, length, k, outputs, k)
            proposed.append(list(outputs[:count]))
        return proposed
