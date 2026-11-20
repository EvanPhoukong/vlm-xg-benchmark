import unittest
import common as c
class GuardTests(unittest.TestCase):
    def test_only_effort_changes(self):
        variants=[c.request_body(c.raw_prompt(),b'image',e) for e in ('low','medium','high')]
        for v in variants: v.pop('reasoning')
        self.assertEqual(variants[0],variants[1]); self.assertEqual(variants[1],variants[2])
        self.assertEqual(set(variants[0]['input'][0]['content'][0]),{'type','image_url','detail'})
    def test_uncertain_charge_stops(self):
        with self.assertRaises(ValueError): c.check_budget([{'state':'started'}],'reasoning',500)
    def test_global_and_stage_caps(self):
        for ledger in [[{'state':'completed','stage':'reasoning','usage':{'output_tokens':700000}}], [{'state':'completed','stage':'confirmation','usage':{'output_tokens':4000000}}]]:
            with self.assertRaises(ValueError): c.check_budget(ledger,'reasoning',500)
    def test_new_match_groups_exclude_previous_data(self):
        split=c.read(c.OUT/'data/match_split.json')
        d=set(split['development_match_ids']); t=set(split['confirmation_match_ids']); old=set(split['used_match_ids'])
        self.assertFalse(d&t); self.assertFalse(d&old); self.assertFalse(t&old)
        self.assertEqual(len(c.read(c.OUT/'reasoning/selection.json')),50)
        self.assertEqual(len(c.read(c.OUT/'confirmation/selection.json')),150)
    def test_frozen_image_hashes(self):
        p=c.read(c.OUT/'reasoning/protocol.json')
        self.assertEqual(len(p['schedule']),150)
        for j in p['schedule']: self.assertEqual(c.sha(c.OUT/j['image']),j['image_sha256'])
if __name__=='__main__': unittest.main()
