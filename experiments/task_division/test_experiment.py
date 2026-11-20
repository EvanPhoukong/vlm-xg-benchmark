"""Integrity checks for outcome-blind Luna-only divided-task inference."""
import copy, json, tempfile, unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pandas as pd
import run_experiment as e

class DivisionTests(unittest.TestCase):
    def setUp(self):
        self.positions={'shooter':{'lateral':2.1,'depth':10.5},'goalkeeper':None}
        self.defenders={'defenders':[{'lateral':-1.,'depth':6.}]}

    def test_fusion_uses_only_measurements_and_preserves_missingness(self):
        answer=e.fuse(self.positions,self.defenders)
        self.assertEqual(answer,{**self.positions,**self.defenders})
        answer['shooter']['depth']=99
        self.assertEqual(self.positions['shooter']['depth'],10.5)
        features=e.robust.external_features([{'shooter':None,'goalkeeper':None,'defenders':[]}])
        self.assertTrue(features.isna().all().all())

    def test_invalid_coordinates_and_foreign_fields_rejected(self):
        invalid=[dict(self.positions,goal=1),
        {'shooter':{'lateral':True,'depth':0},'goalkeeper':None},
        {'shooter':{'lateral':0,'depth':np.nan},'goalkeeper':None},
        {'shooter':{'lateral':0,'depth':100},'goalkeeper':None}]
        for answer in invalid:
            with self.assertRaises(ValueError): e.validate('shooter_keeper',answer)
        with self.assertRaises(ValueError): e.validate('defenders',{'defenders':[None]})

    def test_only_luna_family_allowed(self):
        e.check_model('gpt-6-luna'); e.check_model('gpt-6-luna-2026-10-01')
        for name in ['gpt-6-sol','gpt-6-astra','gpt-5.6-luna','other']:
            with self.assertRaises(ValueError): e.check_model(name)

    def test_request_contains_only_image_and_fixed_prompt(self):
        client=Mock()
        client.responses.create.return_value=Mock(model=e.MODEL,status='completed',
        output_text=json.dumps(self.positions),id='fake',usage=None)
        answer,_=e.ask(client,'shooter_keeper',b'fake image')
        self.assertEqual(answer,self.positions)
        request=client.responses.create.call_args.kwargs
        self.assertEqual(request['model'],'gpt-6-luna'); self.assertFalse(request['store'])
        content=request['input'][0]['content']
        self.assertEqual([v['type'] for v in content],['input_image','input_text'])
        self.assertEqual(content[1]['text'],e.COMMON+'\n'+e.ROLES['shooter_keeper'])
        self.assertEqual(set(request['text']['format']['schema']['properties']),{'shooter','goalkeeper'})

    def test_partial_cohort_cannot_be_scored(self):
        with self.assertRaisesRegex(ValueError,'Full 600-response'):
            e.evaluate(None,None,None,{'shots':['shot']},'sig',{})

    def test_paired_interval_reproducible_and_primary_adjusted(self):
        frame=pd.DataFrame({'match_id':[1,1,2,3],'goal':[0,1,0,0],
        'a':[.1,.6,.2,.3],'b':[.2,.5,.1,.2]})
        primary=e.paired_interval(frame,'a','b',True)
        secondary=e.paired_interval(frame,'a','b',False)
        self.assertAlmostEqual(primary['brier_difference'],np.mean((frame.a-frame.goal)**2-(frame.b-frame.goal)**2))
        self.assertLessEqual(primary['ci_low'],secondary['ci_low'])
        self.assertGreaterEqual(primary['ci_high'],secondary['ci_high'])
        self.assertEqual(primary['confidence_level'],.975)
        self.assertEqual(primary,e.paired_interval(frame,'a','b',True))

    def test_protected_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'input'; path.write_text('original')
            protocol={'protected_inputs_sha256':{str(path):e.sha(path)}}
            e.verify_protected(protocol); path.write_text('changed')
            with self.assertRaises(ValueError): e.verify_protected(protocol)

    def test_cache_rejects_duplicates_changed_inputs_and_model(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder); protocol={'images_sha256':{'shot':'imagehash'}}
            entry={'shot_id':'shot','role':'shooter_keeper','signature':'sig',
            'image_sha256':'imagehash','returned_model':e.MODEL,'answer':self.positions,
            'request_sha256':e.request_hash('shooter_keeper','imagehash')}
            path=folder/'requests.jsonl'; path.write_text(json.dumps(entry)+'\n')
            self.assertEqual(len(e.load_cache(folder,protocol,'sig')),1)
            for field in ['signature','image_sha256','request_sha256','returned_model']:
                bad=copy.deepcopy(entry); bad[field]='changed'
                path.write_text(json.dumps(bad)+'\n')
                with self.assertRaises(ValueError): e.load_cache(folder,protocol,'sig')
            path.write_text((json.dumps(entry)+'\n')*2)
            with self.assertRaises(ValueError): e.load_cache(folder,protocol,'sig')

if __name__=='__main__': unittest.main()
