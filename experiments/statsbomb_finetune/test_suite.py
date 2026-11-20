"""Offline leakage, data integrity and launch-guard tests. No inference calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
import prepare
import train


def event(outcome="Saved", header=False):
    return {"id":"example-id","type":{"name":"Shot"},"location":[105,42],"shot":{
        "type":{"name":"Open Play"},"outcome":{"name":outcome},"statsbomb_xg":.12,
        "body_part":{"name":"Head" if header else "Right Foot"},
        "freeze_frame":[{"location":[118,40],"teammate":False,"position":{"name":"Goalkeeper"}},
                        {"location":[110,41],"teammate":False,"position":{"name":"Defender"}},
                        {"location":[104,35],"teammate":True,"position":{"name":"Forward"}}]}}


class DataTests(unittest.TestCase):
    def test_outcomes_change_labels_but_never_images(self):
        a,_ = common.extract(event(),"123","1:1")
        modified = event("Goal",header=True)
        modified["shot"]["statsbomb_xg"] = .99
        b,_ = common.extract(modified,"123","1:1")
        self.assertEqual((a["goal"],b["goal"]),(0,1))
        self.assertEqual(common.image_payload(a),common.image_payload(b))
        with tempfile.TemporaryDirectory() as directory:
            p,q = Path(directory)/"a.png",Path(directory)/"b.png"
            common.render(a,p)
            common.render(b,q)
            self.assertEqual(common.sha(p),common.sha(q))
            b["x"] -= 5
            common.render(b,q)
            self.assertNotEqual(common.sha(p),common.sha(q))

    def test_inputs_strip_all_outcome_and_context_metadata(self):
        row,_ = common.extract(event(),"123","1:1")
        payload = common.image_payload(row)
        self.assertEqual(set(payload),{"x","y","players"})
        self.assertNotIn("goal",json.dumps(payload))
        self.assertNotIn("statsbomb_xg",json.dumps(payload))
        self.assertNotIn("header",json.dumps(payload))

    def test_penalty_missing_keeper_and_invalid_shots_rejected(self):
        modified = event()
        modified["shot"]["type"]["name"] = "Penalty"
        self.assertEqual(common.extract(modified,"1","1:1")[1],"penalty")
        modified = event()
        modified["shot"]["freeze_frame"] = modified["shot"]["freeze_frame"][1:]
        self.assertEqual(common.extract(modified,"1","1:1")[1],"opposing_keeper_not_recorded")
        modified = event()
        modified["location"][0] = float("nan")
        self.assertEqual(common.extract(modified,"1","1:1")[1],"invalid_shooter_location")

    def test_post_angle_handles_both_sides_of_goal(self):
        d,a = common.geometry(119,40)
        self.assertAlmostEqual(d,1)
        self.assertGreater(a,2)
        self.assertAlmostEqual(common.geometry(110,30)[1],common.geometry(110,50)[1])

    def test_match_split_deterministic_and_disjoint(self):
        metadata = {str(i):{"competition_id":1 if i<20 else 2,"season_id":1} for i in range(40)}
        a,b = prepare.split_matches(metadata),prepare.split_matches(dict(reversed(list(metadata.items()))))
        self.assertEqual(a,b)
        sets = [set(a[s]) for s in ("train","validation","test")]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),40)

    def test_jsonl_rejects_teacher_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/"train.jsonl").write_text(json.dumps({"shot_id":"a","match_id":"b","image":"a.png",
                "image_sha256":"test","label":0,"statsbomb_xg":.1})+"\n")
            protocol = {"jsonl_sha256":{"train":common.sha(out/"train.jsonl")}}
            with patch.object(train,"OUT",out),self.assertRaises(ValueError):
                train.load_split("train",protocol)

    def test_missing_runtime_aborts_before_loading_training_or_test(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train,"OUT",Path(directory)), \
                 patch.object(train,"preflight",return_value={"status":"blocked_missing_training_runtime","missing_packages":["torch"]}), \
                 patch.object(train,"load_split",side_effect=AssertionError("Must not load data")):
                self.assertEqual(train.train("lora",smoke=True),2)

    def test_actual_frozen_data_match_integrity(self):
        if not (common.OUT/"protocol.json").exists():
            self.skipTest("Prepare the corpus first")
        protocol,data = common.check_data()
        self.assertEqual({k:len(v) for k,v in data.items()},common.LIMITS)
        self.assertEqual(protocol["budget"]["new_paid_api_calls"],0)
        self.assertEqual(len({r["shot_id"] for rs in data.values() for r in rs}),2900)
        loaded = {s:train.load_split(s,protocol) for s in ("train","validation")}
        self.assertFalse({r["match_id"] for r in loaded["train"]}&{r["match_id"] for r in loaded["validation"]})


if __name__ == "__main__":
    unittest.main()
