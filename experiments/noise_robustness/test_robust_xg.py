import unittest
import numpy as np
import pandas as pd
import robust_xg as study


def sample_frame(n=1):
    return pd.DataFrame([{'shot_id': f'shot-{i}', 'match_id': i//4, 'group_id': f'wc/{i//4}',
                          'x': 105+i%8, 'y': 34+i%12, 'header': i%2, 'goal': int(i%5 == 0),
                          'sb_xg': .1, 'defenders_in_cone': 1,
                          'ff': [{'teammate': False, 'location': [118, 40], 'position': {'name': 'Goalkeeper'}},
                                 {'teammate': False, 'location': [113, 39], 'position': {'name': 'Center Back'}}]}
                         for i in range(n)])


def zero_pool():
    return pd.DataFrame([{'shot_id': 'donor', 'match_id': 0, 'group_id': 'wc/0',
                          'shooter_dx': 0., 'shooter_dy': 0., 'shooter_missing': False,
                          'keeper_dx': 0., 'keeper_dy': 0., 'keeper_missing': False}])


class RobustnessTests(unittest.TestCase):
    def test_held_out_match_cannot_supply_errors(self):
        errors = pd.concat([zero_pool(), zero_pool().assign(group_id='wc/1', match_id=1)], ignore_index=True)
        pool = study.training_error_pool(errors, {'wc/0'}, {'wc/1'})
        self.assertEqual(set(pool.group_id), {'wc/0'})
        with self.assertRaises(ValueError):
            study.training_error_pool(errors, {'wc/0'}, {'wc/0'})

    def test_outcomes_and_commercial_xg_cannot_change_augmented_inputs(self):
        frame = sample_frame(8)
        before, parents = study.augmented_features(frame, zero_pool(), 3, 7)
        frame.goal = 1-frame.goal; frame.sb_xg = .999; frame.header = 1-frame.header
        after, other = study.augmented_features(frame, zero_pool(), 3, 7)
        pd.testing.assert_frame_equal(before, after)
        np.testing.assert_array_equal(parents, other)

    def test_noise_changes_positions_before_geometry_is_recomputed(self):
        frame = sample_frame()
        x, _ = study.augmented_features(frame, zero_pool().assign(shooter_dx=4.), 2, 7, jitter_sd=0, dropout=0)
        s, k, d = study.scene(frame.iloc[0])
        expected = study.spatial.geometry_features(s+[4, 0], k, d)
        for name in study.spatial.RICH:
            self.assertAlmostEqual(x.iloc[1][name], expected[name])
        self.assertNotEqual(x.iloc[0].distance, x.iloc[1].distance)

    def test_missing_keeper_and_defenders_do_not_use_reference_substitutes(self):
        x, _ = study.augmented_features(sample_frame(), zero_pool().assign(keeper_missing=True), 1, 7, clean_first=False, dropout=1)
        self.assertTrue(np.isnan(x.iloc[0].keeper_distance))
        self.assertEqual(x.iloc[0].defender_coverage, 0)
        self.assertEqual(x.iloc[0].close_defenders, 0)

    def test_zero_noise_weighted_duplicates_preserve_the_fitted_model(self):
        frame = sample_frame(80)
        clean = study.spatial.known_features(frame)[study.spatial.RICH]
        copied, parents = study.augmented_features(frame, zero_pool(), 4, 7, jitter_sd=0, dropout=0)
        first = study.fit_head(clean, frame.goal)
        second = study.fit_head(copied, frame.goal.to_numpy()[parents], np.full(len(parents), .25))
        np.testing.assert_allclose(study.predict(first, clean), study.predict(second, clean), atol=1e-7)

    def test_test_measurements_do_not_accept_true_shot_coordinates(self):
        x = study.external_features([{'shooter': {'lateral': 2., 'depth': 12.}, 'goalkeeper': None, 'defenders': []}])
        self.assertAlmostEqual(x.iloc[0].distance, np.hypot(2, 12))
        self.assertTrue(np.isnan(x.iloc[0].keeper_distance))
        self.assertNotIn('header', x)


if __name__ == '__main__':
    unittest.main()
