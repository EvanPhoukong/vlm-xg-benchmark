"""Offline controls for the larger Sol evaluation; never sends API requests."""
from fractions import Fraction
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import expand_sol as expansion


class ExpansionTests(unittest.TestCase):
    def test_timestamp_preserves_nonzero_video_start(self):
        self.assertEqual(expansion.frame_pts(90000,25,Fraction(1,90000)),180000)
        self.assertEqual(expansion.frame_pts(90000,0,Fraction(1,90000)),90000)
        with self.assertRaises(ValueError):
            expansion.frame_pts(0,1,Fraction(1,24))

    def test_all_audited_shots_scheduled_once_without_rebuying_pilot(self):
        plan=expansion.check_plan()
        jobs=expansion.schedule(plan,["sol"])
        self.assertEqual((plan["shots"],plan["goals"],plan["matches"]),(306,44,10))
        self.assertEqual(len({r["shot_id"] for r in plan["cohort"]}),306)
        self.assertEqual(len(jobs),276)
        self.assertEqual(sum(r["reuse_original"] for r in plan["cohort"]),30)
        self.assertTrue(all(r["frame_index"]==r["event_frame"]-9 for r in jobs))
        self.assertTrue(all(r["arm"]=="sol" and r["condition"]=="original" for r in jobs))
        self.assertTrue(all("goal" not in r for r in jobs))

    def test_maximum_reservations_fit_remaining_budget(self):
        budget=expansion.check_plan()["budget"]
        self.assertLess(budget["prior_conservative_usd"]+budget["max_new_reservations_usd"],10.)
        self.assertEqual(budget["automatic_retries"],0)

    def test_baseline_never_fits_evaluated_match(self):
        models=expansion.c.read(expansion.OLD/"lomo_models.json")
        self.assertEqual(len(models),10)
        for match,arms in models.items():
            for fit in arms.values():
                self.assertNotIn(match,fit["train_matches"])
                self.assertEqual(len(fit["train_matches"]),9)

    def test_missing_frames_stop_before_api_and_release_campaign_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"old").mkdir()
            (root/"new").mkdir()
            with patch.object(expansion,"OLD",root/"old"), patch.object(expansion,"OUT",root/"new"), \
                 patch.object(expansion,"check_plan",return_value={}), \
                 patch.object(expansion.engine,"api") as api:
                originals={key:getattr(expansion.engine,key) for key in ("OUT","MODELS","check_protocol","schedule","api")}
                try:
                    with self.assertRaisesRegex(ValueError,"Extract and audit"):
                        expansion.paid_run(execute=True)
                finally:
                    for key,value in originals.items():
                        setattr(expansion.engine,key,value)
                api.assert_not_called()
                self.assertFalse((root/"old/runner.lock").exists())
                self.assertFalse((root/"new/ledger.json").exists())


if __name__=="__main__":
    unittest.main()
