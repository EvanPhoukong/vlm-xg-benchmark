"""Offline checks for hybrid feature isolation and whole-match training splits."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hybrid_xg as hybrid
import geometry_diagnostic as geo
import xg_vlm_pilot as pilot


def reference_shots():
    rows = []
    for index in range(80):
        x = 92 + index % 24
        y = 29 + index % 22
        distance, angle = pilot.geometry(x, y)
        rows.append({'shot_id': f'shot-{index}', 'match_id': index // 4,
                     'x': x, 'y': y, 'goal': int(index % 5 == 0), 'sb_xg': .15,
                     'distance': distance, 'angle': angle, 'header': index % 2,
                     'defenders_in_cone': index % 3,
                     'baseline_xg': .15, 'vlm_xg': .18,
                     'ff': [{'teammate': False, 'position': {'name': 'Goalkeeper'},
                             'location': [118, 38 + index % 5]}]})
    return pd.DataFrame(rows)


def extracted_positions(reference):
    rows = []
    for shot in reference.itertuples():
        sl, sd, kl, kd = hybrid.reference_positions(shot)
        rows.append({'shot_id': shot.shot_id, 'shooter_lateral': sl, 'shooter_depth': sd,
                     'goalkeeper_lateral': kl, 'goalkeeper_depth': kd})
    return pd.DataFrame(rows)


class HybridTests(unittest.TestCase):
    def test_feature_geometry_and_keeper_cone(self):
        central = hybrid.position_features(0, 12, 0, 2)
        self.assertEqual(central['distance'], 12)
        self.assertAlmostEqual(central['angle'], 2 * math.atan(4 / 12))
        self.assertEqual(central['keeper_depth'], 2)
        self.assertEqual(central['keeper_off_axis'], 0)
        self.assertEqual(central['keeper_in_cone'], 1)
        self.assertEqual(hybrid.position_features(0, 12, 10, 2)['keeper_in_cone'], 0)
        self.assertEqual(hybrid.position_features(0, 12, 0, 15)['keeper_in_cone'], 0)
        self.assertEqual(hybrid.position_features(0, 12, 3, 2)['keeper_off_axis'], 3)
        first = hybrid.position_features(-3, 12, -2, 1)
        reflected = hybrid.position_features(3, 12, 2, 1)
        self.assertEqual(first, reflected)

    def test_prediction_features_cannot_use_outcome_header_or_true_coordinates(self):
        extracted = extracted_positions(reference_shots().iloc[:5])
        before = hybrid.vlm_features(extracted)
        for column in ['goal', 'header', 'sb_xg', 'distance', 'angle', 'defenders_in_cone',
                       'x', 'y', 'true_shooter_lateral', 'true_goalkeeper_lateral']:
            extracted[column] = np.arange(len(extracted)) * 1000
        pd.testing.assert_frame_equal(before, hybrid.vlm_features(extracted))
        self.assertEqual(before.columns.tolist(), hybrid.FEATURES)

    def test_missing_geometry_is_retained_and_never_replaced_with_truth(self):
        pool = reference_shots()
        evaluation = pool.iloc[::3].copy()
        extracted = extracted_positions(evaluation)
        extracted.loc[0, ['shooter_lateral', 'shooter_depth', 'goalkeeper_lateral', 'goalkeeper_depth']] = np.nan
        predicted, audit = hybrid.grouped_predictions(pool, evaluation, extracted)
        self.assertEqual(len(predicted), len(evaluation))
        self.assertTrue(predicted.loc[0, ['vlm_distance', 'vlm_angle', 'vlm_keeper_depth']].isna().all())
        self.assertTrue(np.isfinite(predicted.hybrid_xg).all())

    def test_all_selected_shots_scored_once_and_entire_matches_excluded(self):
        pool = reference_shots()
        evaluation = pool.iloc[::4].copy()  # Other shots from the same matches remain in the pool.
        predicted, audit = hybrid.grouped_predictions(pool, evaluation, extracted_positions(evaluation))
        tested = []
        for fold in audit:
            self.assertFalse(set(fold['training_match_ids']).intersection(fold['heldout_match_ids']))
            for match_id in fold['heldout_match_ids']:
                self.assertNotIn(match_id, fold['training_match_ids'])
            self.assertGreater(fold['heldout_pool_shots'], fold['evaluation_shots'])
            tested += fold['evaluation_shot_ids']
        self.assertEqual(sorted(tested), sorted(evaluation.shot_id))
        self.assertEqual(len(set(tested)), len(tested))
        np.testing.assert_allclose(predicted.hybrid_xg, predicted.reference_geometry_xg, atol=1e-12)

    def test_heldout_labels_cannot_change_their_own_predictions_or_preprocessing(self):
        pool = reference_shots()
        evaluation = pool.iloc[::2].copy()
        extracted = extracted_positions(evaluation)
        predicted, audit = hybrid.grouped_predictions(pool, evaluation, extracted)
        fold_matches = set(audit[0]['heldout_match_ids'])
        changed_pool = pool.copy()
        changed_pool.loc[changed_pool.match_id.isin(fold_matches), 'goal'] = 1 - changed_pool.loc[changed_pool.match_id.isin(fold_matches), 'goal']
        changed_evaluation = changed_pool.set_index('shot_id').loc[evaluation.shot_id].reset_index()
        changed, changed_audit = hybrid.grouped_predictions(changed_pool, changed_evaluation, extracted)
        mask = predicted.match_id.isin(fold_matches)
        np.testing.assert_allclose(predicted.loc[mask, 'hybrid_xg'], changed.loc[mask, 'hybrid_xg'], atol=1e-12)
        self.assertEqual(audit[0]['fitted_geometry_model'], changed_audit[0]['fitted_geometry_model'])

    def test_heldout_feature_values_cannot_change_training_preprocessing(self):
        pool = reference_shots()
        evaluation = pool.iloc[::2].copy()
        extracted = extracted_positions(evaluation)
        _, before = hybrid.grouped_predictions(pool, evaluation, extracted)
        fold_matches = set(before[0]['heldout_match_ids'])
        changed = extracted.copy()
        mask = evaluation.match_id.isin(fold_matches).to_numpy()
        changed.loc[mask, 'shooter_depth'] = 60
        _, after = hybrid.grouped_predictions(pool, evaluation, changed)
        self.assertEqual(before[0]['fitted_geometry_model'], after[0]['fitted_geometry_model'])

    def test_bad_source_or_missing_extraction_stops_instead_of_dropping_shots(self):
        pool = reference_shots()
        evaluation = pool.iloc[:12].copy()
        extracted = extracted_positions(evaluation)
        with self.assertRaisesRegex(ValueError, 'Exactly one VLM extraction'):
            hybrid.grouped_predictions(pool, evaluation, extracted.iloc[:-1])
        changed = evaluation.copy()
        changed.loc[0, 'match_id'] = 99999
        with self.assertRaisesRegex(ValueError, 'Shot and match IDs disagree'):
            hybrid.validate_frames(pool, changed)
        with self.assertRaisesRegex(ValueError, 'duplicate IDs'):
            hybrid.validate_frames(pool, pd.concat([evaluation, evaluation.iloc[[0]]]))


if __name__ == '__main__':
    unittest.main()
