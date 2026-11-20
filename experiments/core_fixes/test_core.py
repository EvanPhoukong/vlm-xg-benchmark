"""Offline controls for direct learning, source isolation, past frames and cost."""
import unittest
import common as c


class Core(unittest.TestCase):
    def test_example_labels_and_no_query_overlap(self):
        examples=c.read(c.OUT/'training/examples.json'); pool=c.read(c.OUT/'heldout/pool.json')
        self.assertEqual(len(examples),48)
        self.assertEqual(sum(x['goal'] for x in examples),sum(x['shuffled_goal'] for x in examples))
        self.assertNotEqual([x['goal'] for x in examples],[x['shuffled_goal'] for x in examples])
        self.assertFalse({x['match_id'] for x in examples}&{x['match_id'] for x in pool})
        split=c.read(c.OUT/'data/match_split.json')
        self.assertFalse(set(split['selected_match_ids'])&set(split['previous_match_ids']))

    def test_budget_waits_for_inflight_and_keeps_uncertain_charges(self):
        pending={'model':'gpt-6-astra','state':'started','input_bound':2000}
        self.assertAlmostEqual(c.reserve_cost(pending),.2298)
        self.assertFalse(c.can_reserve([pending]*17,pending))
        completed={'model':'gpt-6-astra','state':'completed','input_bound':2000,'usage':{'input_tokens':500,'output_tokens':100}}
        self.assertTrue(c.can_reserve([completed]*17,pending))
        with self.assertRaises(ValueError): c.can_reserve([{**pending,'state':'failed_or_uncertain'}],pending)

    def test_payload_has_training_outcomes_only_for_marked_arms(self):
        job=c.read(c.OUT/'pilot/protocol.json')['schedule'][0]
        vanilla=c.payload({**job,'arm':'luna'}); trained=c.payload({**job,'arm':'examples'})
        a=vanilla['input'][0]['content']; b=trained['input'][0]['content']
        self.assertEqual(sum(p['type']=='input_image' for p in a),1)
        self.assertEqual(sum(p['type']=='input_image' for p in b),49)
        self.assertTrue(all('goal' not in j and 'original_logistic' not in j for j in c.read(c.OUT/'pilot/protocol.json')['schedule']))
        self.assertEqual(set(vanilla),{'model','store','service_tier','reasoning','max_output_tokens','input','text'})

    def test_motion_never_uses_future_frames(self):
        selected=c.read(c.OUT/'motion/selection.json')
        self.assertGreaterEqual(len(selected),30)
        for row in selected:
            frames=c.read(c.OUT/'motion/frames'/(row['shot_id']+'.json'))
            self.assertEqual([x['offset_frames'] for x in frames],[-25,-12,0])
            self.assertTrue(all(x['source_frame']<=row['source_start_frame'] for x in frames))
        jobs=c.read(c.OUT/'motion/protocol.json')['schedule']
        for sid in {j['shot_id'] for j in jobs}:
            subset={j['condition']:j for j in jobs if j['shot_id']==sid}
            self.assertEqual(subset['single_ball']['images'][0],subset['history']['images'][-1])


if __name__=='__main__': unittest.main()
