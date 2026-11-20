"""Checks for held-out-match calibration and reproducible report outputs."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate_xg as calibration


def sample_predictions():
    return pd.DataFrame([{
        "shot_id": f"shot-{i}", "match_id": i // 3, "goal": int(i % 5 == 0),
        "vlm_xg": 0.1 + 0.05 * (i % 8), "baseline_xg": 0.11 + 0.02 * (i % 6),
        "sb_xg": 0.15 + 0.01 * (i % 10),
    } for i in range(60)])


class CalibrationTests(unittest.TestCase):
    def test_held_out_outcomes_cannot_change_their_own_corrected_predictions(self):
        frame = sample_predictions()
        folds = calibration.make_folds(frame)
        original, _ = calibration.cross_calibrate(frame, folds)
        changed = frame.copy()
        testing = folds[0][1]
        changed.loc[testing, "goal"] = 1 - changed.loc[testing, "goal"]
        mutated, _ = calibration.cross_calibrate(changed, folds)
        for column in ["offset_xg", "sigmoid_xg", "training_rate_xg"]:
            np.testing.assert_allclose(original.loc[testing, column], mutated.loc[testing, column])

    def test_each_match_is_tested_in_only_one_fold(self):
        frame = sample_predictions()
        result, audit = calibration.cross_calibrate(frame, calibration.make_folds(frame))
        self.assertTrue((result.groupby("match_id").calibration_fold.nunique() == 1).all())
        self.assertTrue((result.calibration_fold >= 0).all())
        for fold in audit:
            self.assertFalse(set(fold["training_match_ids"]) & set(fold["testing_match_ids"]))

    def test_calibration_rejects_match_overlap(self):
        frame = sample_predictions()
        invalid_fold = [(np.arange(1, len(frame)), np.array([0]))]
        with self.assertRaisesRegex(ValueError, "both calibration training and testing"):
            calibration.cross_calibrate(frame, invalid_fold)

    def test_bad_or_duplicate_predictions_are_not_silently_dropped(self):
        frame = sample_predictions()
        duplicate = pd.concat([frame, frame.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "shot_id"):
            calibration.validate_predictions(duplicate)
        for bad in [np.nan, -0.1, 1.1]:
            with self.subTest(value=bad):
                invalid = frame.copy()
                invalid.loc[0, "vlm_xg"] = bad
                with self.assertRaisesRegex(ValueError, "finite probabilities"):
                    calibration.validate_predictions(invalid)

    def test_match_bootstrap_preserves_constant_difference_and_sign(self):
        frame = pd.DataFrame({"match_id": [1, 1, 2, 3, 3, 3], "goal": 1,
                              "vlm_xg": 0.8, "baseline_xg": 0.2})
        result = calibration.match_bootstrap(frame, "vlm_xg", "baseline_xg", n_boot=100)
        for value in [result["brier_difference"], result["ci_low"], result["ci_high"]]:
            self.assertAlmostEqual(value, -0.6)

    def test_reports_are_created_without_network_or_changes_to_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "shots.csv"
            output = directory / "calibration_test"
            sample_predictions().to_csv(source, index=False)
            original = source.read_bytes()
            with patch.object(socket.socket, "connect", side_effect=AssertionError("Network requested")):
                report = calibration.evaluate(source, output, n_boot=100)
            self.assertEqual(source.read_bytes(), original)
            self.assertIn("60 shots", report)
            self.assertIn("other folds", report)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertFalse(metadata["match_overlap"])
            self.assertEqual(metadata["shots"], 60)
            for name in ["report.md", "predictions.csv", "metrics.csv", "comparisons.csv",
                         "fold_audit.json", "calibration_comparison.png"]:
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
