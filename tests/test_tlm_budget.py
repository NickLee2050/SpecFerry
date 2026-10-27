import unittest
from scripts.analyze_tlm_budget import expected_committed, minimum_draft_rate, analyze, account_output_budget


class BudgetTests(unittest.TestCase):
    def test_output_budget_does_not_count_accepted_but_discarded_tail(self):
        rows=[{'prefix_tokens':65,'sampled_token_ids':[0],'target_round':None}]
        for length in (74,83,92):
            rows.append({'prefix_tokens':length,'sampled_token_ids':[0]*9,
                         'target_round':{'accepted_prefix_tokens':8}})
        self.assertEqual(account_output_budget(rows,64,20),20)
        self.assertEqual(rows[-1]['target_round']['accepted_prefix_tokens'],8)
        self.assertEqual(rows[-1]['target_round']['accepted_prefix_tokens_within_output_budget'],1)
        self.assertFalse(rows[-1]['proposal_was_verified'])

    def test_zero_and_perfect_acceptance_include_one_target_token(self):
        self.assertEqual(expected_committed(4,0),1)
        self.assertEqual(expected_committed(4,1),5)
        self.assertAlmostEqual(expected_committed(2,.5),1.75)

    def test_threshold_ties_baseline_and_can_be_impossible(self):
        threshold=minimum_draft_rate(4,5,.25,.3,.01)
        self.assertAlmostEqual(5/(4/threshold+.3+.01),4)
        self.assertIsNone(minimum_draft_rate(1,1,.25,.25))
        self.assertIsNone(minimum_draft_rate(4,2,.25,.6))
        self.assertGreater(minimum_draft_rate(4,5,.25,.3,.05),minimum_draft_rate(4,5,.25,.3,0))

    def test_invalid_measurement_inputs_are_rejected(self):
        for k,alpha in ((0,.8),(True,.8),(4,-.1),(4,1.1),(4,float('nan'))):
            with self.assertRaises(ValueError):expected_committed(k,alpha)
        for args in ((4,6,.25,.2),(4,5,0,.2),(4,5,.25,-.1),(4,5,.25,.2,-1),(4,5,.25,float('inf'))):
            with self.assertRaises(ValueError):minimum_draft_rate(*args)

    def test_measured_verification_and_both_baselines_are_kept_separate(self):
        cases=[{'context_tokens':128,'source':'disabled','rounds':[{'seconds':.3}]}]
        for k in (1,2,4,8):
            cases.append({'context_tokens':128,'source':'external','requested_k':k,'scenario':'all_accept',
                          'rounds':[{'draft_tokens':k,'accepted_prefix_tokens':k,
                                     'accepted_prefix_tokens_within_output_budget':k,'seconds':.28}],
                          'adapter_rows':[{'native_http':{'http_seconds':.001}}]})
        result=analyze({'status':'complete','cases':cases},
                       {'status':'complete','workloads':[{'context_tokens':128,'median_decode_seconds_per_token':.25}]})
        self.assertEqual(len(result['measurements']),4)
        self.assertEqual({x['baseline'] for x in result['budgets']},{'matching_empty_draft','ordinary_graph'})
        self.assertTrue(all(x['measured_verify_seconds']==.28 for x in result['budgets']))
        self.assertTrue(all(x['assumed_draft_tokens_per_second'] in (3,5,10,20,50) for x in result['scenarios']))
