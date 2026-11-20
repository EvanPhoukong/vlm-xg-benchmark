import copy
import unittest
import vision_trials as trial


def answer(x=0):
    return {'shooter': {'lateral': x, 'depth': 12}, 'goalkeeper': {'lateral': 0, 'depth': 1},
            'defenders': [{'lateral': x, 'depth': 6}], 'xg': .1, 'reason': 'Example'}


class VisionTests(unittest.TestCase):
    def test_consensus_aligns_defenders_and_averages_two_agreeing_reads(self):
        result = trial.consensus([answer(0), answer(1)])
        self.assertAlmostEqual(result['shooter']['lateral'], .5)
        self.assertAlmostEqual(result['defenders'][0]['lateral'], .5)

    def test_third_read_median_rejects_one_large_shooter_error(self):
        result = trial.consensus([answer(0), answer(30), answer(.2)])
        self.assertAlmostEqual(result['shooter']['lateral'], .2)

    def test_defender_order_cannot_change_disagreement(self):
        a = answer(); a['defenders'].append({'lateral': 10, 'depth': 20})
        b = copy.deepcopy(a); b['defenders'].reverse()
        self.assertEqual(trial.disagreement(a, b), 0)

    def test_missing_marker_requires_third_read_and_majority_controls_visibility(self):
        a, b = answer(), answer(); b['goalkeeper'] = None
        self.assertGreater(trial.disagreement(a, b), 2)
        self.assertIsNotNone(trial.consensus([a, b, a])['goalkeeper'])
        self.assertIsNone(trial.consensus([a, b, b])['goalkeeper'])


if __name__ == '__main__':
    unittest.main()
