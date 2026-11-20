"""Shared SoccerTrack utilities. Importing never makes API calls."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "soccertrack-v2"
OUT = ROOT / "results/soccertrack_comparison_v1"
PILOT = ROOT / "results/soccertrack_sol_v1/manifest.json"
TRAIN = ("117092", "118575", "118576", "118577", "118578", "128058")
VALIDATION = ("117093", "132877")
TEST = ("128057", "132831")
SEED = 2026111301
PROMPT = '''You are a professional football (soccer) analyst.
The image is a panoramic match frame taken shortly before the shot.
Estimate the probability that this shot results in a goal (expected goals, xG),
assuming an average professional finisher.
Reply with JSON only: {"xg": <number between 0 and 1>, "reason": "<one short sentence>"}'''
SCHEMA = {"type": "object", "properties": {
    "xg": {"type": "number", "minimum": 0, "maximum": 1},
    "reason": {"type": "string"}}, "required": ["xg", "reason"],
    "additionalProperties": False}
# Standard USD/million input, cached input, output tokens. Checked 2026-11-13 UTC.
MODELS = {
    "luna": {"provider": "openai", "model": "gpt-6-luna", "rates": [.10, .01, .50]},
    "terra": {"provider": "openai", "model": "gpt-6.1-terra", "rates": [1., .05, 5.]},
    "sol": {"provider": "openai", "model": "gpt-6.1-sol", "rates": [2., .10, 10.]},
    "astra": {"provider": "openai", "model": "gpt-6-astra", "rates": [10., 1., 50.]},
}
MAX_OUTPUT = 1024
CAP = 10.0


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    # JSON canonicalizes integer mapping keys; compare the persisted representation on resume.
    value = json.loads(json.dumps(value, allow_nan=False))
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        save(path, value)


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -np.asarray(x, dtype=float)))


def fit_logistic(rows, features, ridge=1.0):
    """Fixed L2 logistic MLE; sum loss, unpenalized intercept, training-only scaling."""
    x = np.array([[r[f] for f in features] for r in rows], dtype=float)
    y = np.array([r["goal"] for r in rows], dtype=float)
    if len(set(y)) != 2 or not np.isfinite(x).all():
        raise ValueError("Training needs finite features and both outcomes")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.diag([0.] + [ridge] * len(features))
    beta = np.zeros(z.shape[1])
    beta[0] = math.log(y.mean() / (1 - y.mean()))

    def objective(b):
        logits = z @ b
        return float((np.logaddexp(0, logits) - y * logits).sum() + .5 * b @ penalty @ b)

    for iteration in range(100):
        p = sigmoid(z @ beta)
        grad = z.T @ (p - y) + penalty @ beta
        if np.max(np.abs(grad)) < 1e-7:
            break
        hessian = (z.T * (p * (1 - p))) @ z + penalty
        step = np.linalg.solve(hessian, grad)
        if float(grad @ step) < 1e-14:
            break
        rate, loss = 1., objective(beta)
        while rate > 1e-10 and objective(beta - rate * step) > loss - 1e-4 * rate * (grad @ step):
            rate *= .5
        if rate <= 1e-10:
            if np.max(np.abs(grad)) < 1e-6:
                break
            raise ValueError("Logistic line search did not converge")
        beta -= rate * step
    else:
        raise ValueError("Logistic optimizer did not converge")
    return {"features": list(features), "mean": mean.tolist(), "scale": scale.tolist(),
            "beta": beta.tolist(), "ridge": ridge, "iterations": iteration + 1,
            "max_abs_gradient": float(np.max(np.abs(grad))),
            "train_matches": sorted({r["match_id"] for r in rows}), "train_shots": len(rows)}


def predict(model, rows):
    x = np.array([[r[f] for f in model["features"]] for r in rows], dtype=float)
    z = np.column_stack([np.ones(len(x)), (x - model["mean"]) / model["scale"]])
    return sigmoid(z @ model["beta"]).tolist()


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or not len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid evaluation probabilities")
    positive, negative = p[y == 1], p[y == 0]
    auc = float(((positive[:, None] > negative).sum() + .5 * (positive[:, None] == negative).sum()) /
                (len(positive) * len(negative))) if len(positive) and len(negative) else None
    # Numerical protection for log loss only; Brier and stored probabilities stay raw.
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return {"n": len(y), "goals": int(y.sum()), "goal_rate": float(y.mean()),
            "brier": float(np.mean((p - y) ** 2)), "logloss": float(-np.mean(y * np.log(q) + (1-y) * np.log1p(-q))),
            "auc": auc, "mean_xg": float(p.mean()), "mean_bias": float((p-y).mean())}


def token_cost(usage, rates, conservative=False):
    ni, no = usage["input_tokens"], usage["output_tokens"]
    if any(type(v) is not int or v < 0 for v in (ni, no)):
        raise ValueError("Missing or invalid token usage")
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    if type(cached) is not int or not 0 <= cached <= ni:
        raise ValueError("Invalid cache token usage")
    ir, cr, outr = rates
    return ((ni * ir * 1.25 if conservative else (ni-cached)*ir + cached*cr) + no*outr) / 1e6


def exposure(ledger, model, prior=0.):
    return prior + sum(e["charged_bound_usd"] for e in ledger if e["model"] == model)


def can_reserve(ledger, model, reserved, prior=0., cap=CAP):
    if not math.isfinite(reserved) or reserved < 0:
        raise ValueError("Invalid reservation")
    if any(e["state"] in ("reserved", "uncertain") for e in ledger):
        raise ValueError("Unreconciled request; refusing any further paid calls")
    return exposure(ledger, model, prior) + reserved <= cap


def credential():
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value
    values = set(re.findall(r"sk-(?!ant-)[A-Za-z0-9_-]+", (ROOT.parent / "keys.txt").read_text()))
    if len(values) != 1:
        raise ValueError("Configure one OpenAI credential; credentials are never printed")
    return values.pop()


def api(endpoint, key, data=None):
    request = urllib.request.Request("https://api.openai.com/v1/" + endpoint,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=150) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API HTTP {e.code}; no automatic retry") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("API network unavailable; no automatic retry") from None
