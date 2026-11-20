"""Validate the first complete block independently while the run continues; no scoring."""
import json
import numpy as np

import audit as v

c, ROOT, OUT = v.c, v.ROOT, v.OUT
p = c.read(OUT / "protocol.json")
ledger = c.read(OUT / "attempts.json")[:50]
assert len(ledger) == 50 and all(r["state"] == "completed" for r in ledger)
assert len({r["job"]["match_id"] for r in ledger}) == 10
assert len({r["job"]["arm"] for r in ledger}) == 5
historical = {r["job"]["shot_id"]: r for r in c.read(v.e.SOURCE / "attempts.json") if r["job"]["arm"] == "diagram"}
for entry in ledger:
    assert v.fingerprint(v.request(entry["job"], p)) == entry["request_sha256"]
    assert abs(v.cost(entry["usage"], True)-entry["conservative_usd"]) < 1e-12
    if entry["job"]["arm"] == "zero":
        assert historical[entry["job"]["shot_id"]]["request_sha256"] == entry["request_sha256"]
pool = c.read(OUT / "pool.json")
models = c.read(OUT / "teacher_models.json")
examples = c.read(OUT / "examples.json")
for i, mid in enumerate(sorted(models)):
    training = [r for r in pool if r["match_id"] != mid]
    assert set(models[mid]["training_ids"]) == {r["shot_id"] for r in training}
    model = c.fit_logistic(training, v.e.spatial.FEATURES)
    np.testing.assert_allclose(c.predict(model, pool), c.predict(models[mid]["model"], pool), rtol=0, atol=1e-10)
    assert examples[mid] == v.e.examples_for(pool, mid, i, model)
result = {"passed": True, "requests_independently_rebuilt": 50, "matches": 10,
          "conditions": 5, "teacher_exclusion_checks": 10, "accuracy_scored": False}
c.save(OUT / "preflight.json", result)
print(json.dumps(result), flush=True)
