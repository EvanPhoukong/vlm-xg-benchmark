import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import common as c
import analyze
class AliasScoreCheck(unittest.TestCase):
    def test_identical_candidate_reuses_one_prediction(self):
        with tempfile.TemporaryDirectory() as name,patch.object(c,'OUT',Path(name)):
            folder=c.OUT/'confirmation'; folder.mkdir(); (c.OUT/'image.png').write_bytes(b'test-image')
            jobs=[]; attempts=[]; rows=[]
            for i,goal,probability in [(1,0,.2),(2,1,.8)]:
                sid=str(i); job={'shot_id':sid,'condition':'original_low','effort':'low','image':'image.png','image_sha256':c.sha(c.OUT/'image.png')}; jobs.append(job)
                response={'model':c.MODEL,'status':'completed','reasoning':{'effort':'low'},'usage':{'input_tokens':10,'output_tokens':20},'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'xg':probability,'reason':'Synthetic test only'})}]}]}
                c.save(folder/f'{i}.json',response)
                attempts.append({**job,'stage':'confirmation','state':'completed','response_file':f'confirmation/{i}.json','usage':response['usage'],'answer':{'xg':probability,'reason':'Synthetic test only'}})
                rows.append({'shot_id':sid,'match_id':i,'goal':goal,'distance':10,'baseline_xg':probability,'baseline_image_xg':probability,'sb_xg':probability})
            c.save(folder/'protocol.json',{'schedule':jobs,'metadata':{'aliases':{'candidate':'original_low'}}})
            c.save(folder/'selection.json',rows); c.save(c.OUT/'attempts.json',attempts); c.save(c.OUT/'candidate.json',{'effort':'low','renderer':'standard'})
            with contextlib.redirect_stdout(io.StringIO()): analyze.score('confirmation')
            result=c.read(folder/'summary.json')
            self.assertEqual(result['scores']['candidate'],result['scores']['original_low'])
            self.assertEqual(result['primary'][1]['brier_difference'],0)
            self.assertEqual(len(attempts),2)
if __name__=='__main__': unittest.main()
