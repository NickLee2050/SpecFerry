import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from tests import test_vllm_proposer as proposer_tests
from specferry.vllm_profile import ProfiledDraftProposer


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.fixture=proposer_tests.ExternalProposerTests();self.fixture.setUp();self.fixture.proposer.close()
        self.control=self.fixture.root/'control.json';self.audit=self.fixture.root/'rounds.jsonl'
        self.set_control('a','external',4)
        self.env=patch.dict(os.environ,{'SPECFERRY_PROFILE_CONTROL':str(self.control),'SPECFERRY_PROFILE_AUDIT':str(self.audit)})
        self.env.start()
        self.torch=patch.dict(sys.modules,{'torch':NS(cuda=NS(synchronize=lambda:None))});self.torch.start()
        self.p=ProfiledDraftProposer(self.fixture.config)
        self.fixture.mode('all_accept')

    def tearDown(self):
        self.p.close();self.torch.stop();self.env.stop();self.fixture.tearDown()

    def set_control(self,name,source,k):
        self.control.write_text(json.dumps({'run_id':name,'source':source,'draft_tokens':k}))

    def test_committed_prefix_records_actual_acceptance_and_resets_between_trials(self):
        self.assertEqual(self.p.propose([[11]],[3],[[2,10,11]]),[[12,13,14,15]])
        self.p.propose([[12,13,14,15,16]],[8],[[2,10,11,12,13,14,15,16]])
        rows=[json.loads(x) for x in self.audit.read_text().splitlines()]
        self.assertIsNone(rows[0]['target_round'])
        self.assertEqual(rows[1]['target_round']['accepted_prefix_tokens'],4)
        self.assertEqual(rows[1]['target_round']['sampled_tokens_before_output_budget'],5)
        self.set_control('b','disabled',1)
        self.assertEqual(self.p.propose([[11]],[3],[[2,10,11]]),[[]])
        row=json.loads(self.audit.read_text().splitlines()[-1])
        self.assertIsNone(row['target_round']);self.assertIsNone(row['native_http'])

    def test_bad_prefix_advance_is_rejected(self):
        self.p.propose([[11]],[3],[[2,10,11]])
        with self.assertRaises(RuntimeError):self.p.propose([[12]],[4],[[2,10,99,12]])

    def test_remote_endpoint_is_rejected_without_contact(self):
        with patch.dict(os.environ,{'SPECFERRY_DLM_URL':'http://example.invalid/v1/draft'}):
            with self.assertRaises(ValueError):ProfiledDraftProposer(self.fixture.config)
