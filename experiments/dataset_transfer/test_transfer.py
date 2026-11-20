"""Meaningful transfer-data, leakage and shared-budget guards; no network."""
import copy,importlib.util,math,unittest
from pathlib import Path
import run as runner
sp=importlib.util.spec_from_file_location('transfer_prepare',Path(__file__).with_name('prepare.py'))
prep=importlib.util.module_from_spec(sp); sp.loader.exec_module(prep)


class Transfer(unittest.TestCase):
    def test_parallel_reservations_and_uncertain_charges(self):
        entry={'state':'started','model':'gpt-6-luna','input_bound':2000}
        self.assertAlmostEqual(runner.exposure([entry]*6),.013788)
        runner.reserve([],entry)
        with self.assertRaises(ValueError): runner.reserve([copy.copy(entry) for _ in range(435)],entry)
        failed={**entry,'state':'failed_or_uncertain'}
        with self.assertRaises(ValueError): runner.reserve([failed],entry)

    def test_disjoint_match_groups_and_fixed_counts(self):
        plan=runner.base.read(runner.OUT/'data_plan.json')
        all_test_ids=set()
        for name,spec in plan['cohorts'].items():
            self.assertFalse(set(spec['train_ids'])&set(spec['test_ids']))
            self.assertFalse(set(spec['train_ids']+spec['test_ids'])&set(plan['prior_used_match_ids']))
            rows=runner.base.read(runner.OUT/name/'selection.json')
            self.assertEqual(len(rows),500); self.assertEqual(len({r['shot_id'] for r in rows}),500)
            self.assertTrue(set(r['match_id'] for r in rows)<=set(spec['test_ids']))
            self.assertFalse(all_test_ids&set(r['shot_id'] for r in rows)); all_test_ids.update(r['shot_id'] for r in rows)

    def test_baselines_reproduce_image_geometry(self):
        for name in runner.COHORTS:
            for row in runner.base.read(runner.OUT/name/'selection.json'):
                d,a=prep.original.geometry(row['x'],row['y'])
                self.assertAlmostEqual(row['distance'],d); self.assertAlmostEqual(row['angle'],a)
                self.assertTrue(60<=row['x']<=120 and 0<=row['y']<=80)
                for key in ('original_logistic','original_no_header'):
                    self.assertTrue(math.isfinite(row[key]) and 0<row[key]<1)

    def test_metrica_has_opposing_keeper_and_no_rendered_outcomes(self):
        rows=runner.base.read(runner.OUT/'metrica_v2/selection.json')
        self.assertEqual(len(rows),46)
        provenance=runner.base.read(runner.OUT/'metrica_v2/provenance.json')
        for game in provenance['games']:
            for team in ('Home','Away'):
                roles=[x for x in game['roles'] if x['team']==team]
                self.assertEqual(roles[0]['goalkeeper'],roles[1]['goalkeeper'])
                self.assertNotEqual(roles[0]['attack_right'],roles[1]['attack_right'])
        for row in rows:
            self.assertTrue(any(not p['teammate'] and p['position']['name']=='Goalkeeper' for p in row['ff']))
        tree=__import__('ast').parse(__import__('inspect').getsource(prep.original.render))
        attributes={n.attr for n in __import__('ast').walk(tree) if isinstance(n,__import__('ast').Attribute) and isinstance(n.value,__import__('ast').Name) and n.value.id=='row'}
        self.assertEqual(attributes,{'ff','x','y'})


if __name__=='__main__': unittest.main()
