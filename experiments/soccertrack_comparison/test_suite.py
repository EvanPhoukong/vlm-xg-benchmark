"""Offline statistical correctness and paid-dispatch protection tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import common
import prepare
import run


class StatisticalTests(unittest.TestCase):
    def test_constant_feature_has_analytic_intercept(self):
        rows = [{"match_id": str(i % 3), "goal": int(i < 3), "feature": 7.} for i in range(10)]
        fitted = common.fit_logistic(rows, ["feature"])
        np.testing.assert_allclose(common.predict(fitted, rows), [.3] * 10, atol=1e-8)

    def test_optimizer_satisfies_independent_gradient_check(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(150, 2))
        y = rng.binomial(1, common.sigmoid(x @ np.array([1.2, -.8]) - 1.))
        rows = [{"match_id": str(i % 5), "goal": int(y[i]), "a": float(v[0]), "b": float(v[1])} for i,v in enumerate(x)]
        model = common.fit_logistic(rows, ["a", "b"])
        beta = np.array(model["beta"])
        z = np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]])

        def objective(b):
            t = z @ b
            return (np.logaddexp(0, t) - y*t).sum() + .5 * (b[1:]**2).sum()

        grad = [(objective(beta + np.eye(3)[i]*1e-5) - objective(beta - np.eye(3)[i]*1e-5)) / 2e-5 for i in range(3)]
        self.assertLess(max(abs(v) for v in grad), 1e-5)
        self.assertGreater(beta[1], 0)
        self.assertLess(beta[2], 0)

    def test_holdout_values_do_not_change_training_scaler(self):
        train = [{"match_id": "train", "goal": i % 2, "x": float(i)} for i in range(10)]
        model = common.fit_logistic(train, ["x"])
        before = json.dumps(model, sort_keys=True)
        common.predict(model, [{"x": -100000.}, {"x": 100000.}])
        self.assertEqual(before, json.dumps(model, sort_keys=True))
        self.assertEqual(model["mean"], [4.5])

    def test_metric_ties_endpoints_and_unclipped_brier(self):
        m = common.metrics([0, 1], [.5, .5])
        self.assertEqual(m["auc"], .5)
        self.assertEqual(m["brier"], .25)
        m = common.metrics([0, 1], [0, 1])
        self.assertEqual(m["brier"], 0)
        self.assertEqual(m["auc"], 1)

    def test_geometry_center_and_mirror(self):
        distance, angle = prepare.geometry(.5, 1-11/105)
        self.assertAlmostEqual(distance, 11.)
        self.assertAlmostEqual(angle, 2*np.arctan(3.66/11))
        np.testing.assert_allclose(prepare.geometry(.3, .8), prepare.geometry(.7, .8))


class BudgetTests(unittest.TestCase):
    def test_prior_and_independent_model_caps(self):
        ledger = [{"model": "a", "state": "completed", "charged_bound_usd": 8.},
                  {"model": "b", "state": "completed", "charged_bound_usd": 9.}]
        self.assertTrue(common.can_reserve(ledger, "a", 1., prior=1.))
        self.assertFalse(common.can_reserve(ledger, "a", 1.00001, prior=1.))
        self.assertTrue(common.can_reserve(ledger, "b", 1.))

    def test_uncertain_or_reserved_requests_stop_dispatch(self):
        for state in ("reserved", "uncertain"):
            with self.assertRaises(ValueError):
                common.can_reserve([{"model": "a", "state": state, "charged_bound_usd": 1.}], "b", .1)

    def test_hidden_reasoning_included_via_total_output_usage(self):
        usage = {"input_tokens": 1000, "output_tokens": 1000,
                 "input_tokens_details": {"cached_tokens": 500}, "output_tokens_details": {"reasoning_tokens": 900}}
        self.assertAlmostEqual(common.token_cost(usage, [10, 1, 50]), .0555)
        self.assertAlmostEqual(common.token_cost(usage, [10, 1, 50], True), .0625)

    def test_invalid_prediction_is_rejected_without_probability_fix(self):
        for value in (1.5, -1, True, float("nan")):
            response = {"model": "gpt-6-luna", "status": "completed", "reasoning": {"effort": "low"},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({"xg": value, "reason": "test"})}]}]}
            with self.assertRaises(ValueError):
                run.validate_answer(response, "gpt-6-luna")

    def dispatch_fixture(self, directory, fail=False):
        root = Path(directory)
        out = root / "out"
        out.mkdir()
        (root / "frame.png").write_bytes(b"test-image-bytes")
        common.save(out / "reused_sol.json", [])
        common.save(out / "protocol.json", {"test": True})
        protocol = {"cohort": [{"shot_id": "shot1", "image": "frame.png", "sha256": common.sha(root / "frame.png")}],
                    "consistency": {"shot_ids": []}, "budget": {"prior_conservative_usd": {}}}
        count = []

        def fake_api(endpoint, key, data=None):
            if endpoint == "models":
                return {"data": [{"id": common.MODELS["luna"]["model"]}]}
            if endpoint == "responses/input_tokens":
                return {"input_tokens": 100}
            if endpoint == "responses":
                ledger = common.read(out / "ledger.json")
                self.assertEqual(ledger[-1]["state"], "reserved")
                self.assertGreater(ledger[-1]["charged_bound_usd"], 0)
                self.assertLessEqual(common.exposure(ledger, data["model"]), 10)
                self.assertNotIn("goal", data)
                count.append(endpoint)
                if fail:
                    raise RuntimeError("simulated uncertain network failure")
                return {"model": data["model"], "status": "completed", "service_tier": "default",
                    "reasoning": {"effort": "low"}, "usage": {"input_tokens": 100, "output_tokens": 50},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"xg":0.12,"reason":"test"}'}]}]}
            raise AssertionError(endpoint)

        return root, out, protocol, count, fake_api

    def test_reservation_precedes_dispatch_and_resume_does_not_rebill(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out, protocol, calls, fake_api = self.dispatch_fixture(directory)
            with patch.object(run, "ROOT", root), patch.object(run, "OUT", out), patch.object(run, "check_protocol", return_value=protocol), \
                 patch.object(run, "credential", return_value="private-test-value"), patch.object(run, "api", side_effect=fake_api), \
                 patch.object(sys, "argv", ["run.py", "--models", "luna", "--execute"]):
                run.main()
                run.main()
            self.assertEqual(len(calls), 1)
            self.assertEqual(common.read(out / "ledger.json")[0]["state"], "completed")
            self.assertFalse((out / "runner.lock").exists())

    def test_failed_request_retains_full_reservation_and_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root, out, protocol, calls, fake_api = self.dispatch_fixture(directory, fail=True)
            with patch.object(run, "ROOT", root), patch.object(run, "OUT", out), patch.object(run, "check_protocol", return_value=protocol), \
                 patch.object(run, "credential", return_value="private-test-value"), patch.object(run, "api", side_effect=fake_api), \
                 patch.object(sys, "argv", ["run.py", "--models", "luna", "--execute"]):
                for _ in range(2):
                    with self.assertRaises(SystemExit):
                        run.main()
            entry = common.read(out / "ledger.json")[0]
            self.assertEqual(entry["state"], "uncertain")
            self.assertEqual(entry["reserved_usd"], entry["charged_bound_usd"])
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
