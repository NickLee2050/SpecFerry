import importlib.util
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'scripts'))
from draft_test_support import tokenizer_digest
from specferry.vllm_proposer import ExternalDraftProposer

spec = importlib.util.spec_from_file_location('mock_dlm_server', ROOT/'scripts/mock_dlm_server.py')
mock_dlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_dlm)


class ExternalProposerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'vocab.json').write_text('{}')
        (self.root/'merges.txt').write_text('#version: 0.2\n')
        self.config = NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                            data_parallel_size=1),
                         scheduler_config=NS(max_num_seqs=1, async_scheduling=False),
                         model_config=NS(hf_config=NS(model_type='opt'), max_model_len=24,
                            tokenizer=str(self.root), get_vocab_size=lambda:50272),
                         speculative_config=NS(num_speculative_tokens=4))
        self.traces = [{'prompt_token_ids':[2,10], 'output_token_ids':[11,12,13,14,15,16,17]}]
        self.server = ThreadingHTTPServer(('127.0.0.1',0), mock_dlm.make_handler(
            self.traces, tokenizer_digest(self.root), self.root/'requests.jsonl', delay=.15))
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval':.01}, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.env = patch.dict(os.environ, {'SPECFERRY_DLM_URL':self.base+'/v1/draft',
            'SPECFERRY_DLM_TIMEOUT':'.05','SPECFERRY_DLM_AUDIT':str(self.root/'audit.jsonl'),
            'SPECFERRY_DRAFT_LIBRARY':str(ROOT/'build/draft/libspecferry_draft.so')})
        self.env.start()
        self.version = patch('importlib.metadata.version', return_value='0.28.0')
        self.version.start()
        self.proposer = ExternalDraftProposer(self.config)

    def tearDown(self):
        self.proposer.close()
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.env.stop(); self.version.stop(); self.tmp.cleanup()

    def mode(self, name, **extra):
        req=Request(self.base+'/control',data=json.dumps({'mode':name, **extra}).encode(),
                    headers={'Content-Type':'application/json'})
        with urlopen(req, timeout=2):
            pass

    def call(self, prefix=(2,10,11), sampled=(11,)):
        return self.proposer.propose([list(sampled)],[len(prefix)],[list(prefix)])

    def test_authoritative_prefix_includes_new_tokens_once(self):
        self.mode('all_accept')
        self.assertEqual(self.call(),[[12,13,14,15]])
        self.assertEqual(self.call((2,10,11,12,13,14),(12,13,14)),[[15,16,17]])
        records=[json.loads(line) for line in (self.root/'requests.jsonl').read_text().splitlines()]
        self.assertEqual([r['prefix_tokens'] for r in records],[3,6])
        self.assertTrue(all(r['matched_oracle'] for r in records))

    def test_stale_invalid_and_timeout_fail_closed(self):
        for mode in ('stale','invalid_token','timeout'):
            self.mode(mode)
            self.assertEqual(self.call(),[[]],mode)
        records=[json.loads(line) for line in (self.root/'audit.jsonl').read_text().splitlines()]
        self.assertEqual([r['status'] for r in records],['fallback']*3)

    def test_rejected_suffix_not_reused_across_rounds(self):
        self.mode('first_reject')
        self.assertEqual(self.call(),[[13,13,14,15]])
        self.mode('all_accept')
        self.assertEqual(self.call((2,10,11,12),(12,)),[[13,14,15,16]])

    def test_no_draft_and_context_limit(self):
        self.assertEqual(self.call(),[[]])
        self.proposer.max_model_len=3
        self.assertEqual(self.call(),[[]])
        self.assertEqual(self.proposer.propose([[]],[0],[[]]),[[]])

    def test_cache_contract_violation_never_contacts_device(self):
        with self.assertRaises(RuntimeError):
            self.call(sampled=(99,))
        self.assertFalse((self.root/'requests.jsonl').exists())

    def test_unsupported_engine_configuration_fails_at_startup(self):
        self.config.scheduler_config.async_scheduling=True
        with self.assertRaises(ValueError):
            ExternalDraftProposer(self.config)
        self.config.scheduler_config.async_scheduling=False
        self.config.parallel_config.tensor_parallel_size=2
        with self.assertRaises(ValueError):
            ExternalDraftProposer(self.config)

    def test_invalid_token_forms_rejected(self):
        for values in ([True],[1.5],[-1],[50272],[1,2,3,4,5]):
            self.mode('all_accept',draft_override=values)
            self.assertEqual(self.call(),[[]])

    def test_disabled_baseline_does_not_load_transport_or_contact_device(self):
        with patch.dict(os.environ, {'SPECFERRY_DRAFT_SOURCE':'disabled',
                                     'SPECFERRY_DRAFT_LIBRARY':'/missing/library.so'}):
            baseline = ExternalDraftProposer(self.config)
            try:
                self.assertEqual(baseline.propose([[11]],[3],[[2,10,11]]),[[]])
                self.assertFalse((self.root/'requests.jsonl').exists())
            finally:
                baseline.close()

    def test_optional_remote_timings_are_audited_without_changing_drafts(self):
        self.mode('all_accept', timing_override={
            'draft_seconds': .25, 'synchronization_seconds': .1, 'server_seconds': .4})
        self.assertEqual(self.call(), [[12,13,14,15]])
        row=json.loads((self.root/'audit.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['reported_dlm_timing']['draft_seconds'], .25)
        self.assertGreater(row['request_bytes'], 0)
        self.assertGreaterEqual(row['prepare_seconds'], 0)
        self.assertGreaterEqual(row['response_validation_seconds'], 0)
        self.mode('all_accept', timing_override={
            'draft_seconds': -1, 'synchronization_seconds': True, 'server_seconds': 'wrong'})
        self.assertEqual(self.call(), [[12,13,14,15]])
        row=json.loads((self.root/'audit.jsonl').read_text().splitlines()[-1])
        self.assertEqual(row['reported_dlm_timing'], {})


if __name__ == '__main__':
    unittest.main()
