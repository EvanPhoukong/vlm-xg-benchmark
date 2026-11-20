import unittest
import numpy as np
import pandas as pd
from confirm_frozen import adjusted_comparison
from context_trial import metadata_for
from fetch_context import CATEGORICAL, NUMERIC


class ConfirmationTests(unittest.TestCase):
    def test_multiple_comparison_interval_contains_the_95_percent_interval(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame({'match_id': np.repeat(np.arange(20), 4), 'goal': np.arange(80)%7 == 0,
                              'first': rng.uniform(.02, .4, 80), 'second': rng.uniform(.02, .4, 80)})
        result = adjusted_comparison(frame, 'first', 'second')
        self.assertLessEqual(result['ci_97_5_low'], result['ci_low'])
        self.assertGreaterEqual(result['ci_97_5_high'], result['ci_high'])

    def test_context_join_follows_shot_ids_and_excludes_outcomes(self):
        rows = []
        for shot in ['a', 'b']:
            rows.append({'shot_id': shot, 'goal': 1, 'sb_xg': .99,
                         **{c: shot for c in CATEGORICAL}, **{c: int(shot == 'a') for c in NUMERIC}})
        metadata = pd.DataFrame(rows)
        result = metadata_for(pd.DataFrame({'shot_id': ['b', 'a']}), metadata)
        self.assertEqual(result.iloc[0].shot_type, 'b')
        self.assertEqual(list(result.columns), CATEGORICAL+NUMERIC)


if __name__ == '__main__':
    unittest.main()
