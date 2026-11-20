import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('run_experiment.py')); p=importlib.util.module_from_spec(s); s.loader.exec_module(p)
class SafetyTests(unittest.TestCase):
    def test_only_reasoning_changes(self):
        a=p.body('generic',b'image','low'); b=p.body('generic',b'image','medium')
        self.assertEqual(a.pop('reasoning'),{'effort':'low'}); self.assertEqual(b.pop('reasoning'),{'effort':'medium'}); self.assertEqual(a,b)
        self.assertNotIn('baseline_xg',str(a)); self.assertNotIn('goal',str(a))
    def test_budget_reserves_next_response(self):
        for ledger in [[{'state':'started'}],[{'state':'completed'}]*24,[{'state':'completed','usage':{'output_tokens':22000}}]]:
            with self.assertRaises(ValueError): p.reserve(ledger,100)
    def test_other_model_and_incomplete_stop(self):
        for response in [{'model':'gpt-6-sol','status':'completed'},{'model':p.MODEL,'status':'incomplete'}]:
            with self.assertRaises(ValueError): p.validate(response)
if __name__=='__main__': unittest.main()
