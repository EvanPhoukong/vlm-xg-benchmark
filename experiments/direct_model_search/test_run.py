"""Offline spending-bound, no-label-input and raw-probability checks."""
import copy, json, unittest
import run as r


class Integrity(unittest.TestCase):
    def response(self,value=.25):
        return {'model':'gpt-6-luna','status':'completed','reasoning':{'effort':'low'},
            'usage':{'input_tokens':500,'output_tokens':90},
            'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'xg':value,'reason':'Visible geometry.'})}]}]}

    def test_raw_probabilities_preserved(self):
        for value in (0,.00123,.93,1): self.assertEqual(r.validate(self.response(value),r.ARMS['luna_original'])['xg'],value)
        for value in (-.1,1.1,True,float('nan')):
            with self.assertRaises(ValueError): r.validate(self.response(value),r.ARMS['luna_original'])

    def test_identity_usage_truncation(self):
        for field,value in [('model','other'),('status','incomplete'),('usage',{})]:
            response=self.response(); response[field]=value
            with self.assertRaises(ValueError): r.validate(response,r.ARMS['luna_original'])

    def test_concurrent_reservations(self):
        pending={'state':'started','model':'gpt-6.1-sol','input_bound':2000}
        self.assertAlmostEqual(r.exposure([pending]),.04596)
        with self.assertRaises(ValueError): r.reserve([copy.copy(pending) for _ in range(66)],pending)
        r.reserve([],pending); pending['state']='failed_or_uncertain'
        with self.assertRaises(ValueError): r.reserve([pending],pending)

    def test_input_has_no_labels_or_tools(self):
        payload=r.body(r.ARMS['luna_uncertainty'],r.prompts()['uncertainty'],b'example')
        self.assertEqual(set(payload),{'model','store','service_tier','reasoning','max_output_tokens','input','text'})
        self.assertEqual([p['type'] for p in payload['input'][0]['content']],['input_image','input_text'])
        self.assertNotIn('baseline_xg',json.dumps(payload)); self.assertNotIn('sb_xg',json.dumps(payload))


if __name__=='__main__': unittest.main()
