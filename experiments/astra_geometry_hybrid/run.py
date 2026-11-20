"""Matched Astra image-to-geometry xG pilot on the saved La Liga cohort."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import geometry_diagnostic as geometry
import hybrid_xg as hybrid

spec = importlib.util.spec_from_file_location(
    "direct_run", ROOT / "experiments/direct_model_search/run.py"
)
direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)

PRIOR = ROOT / "results/core_fixes_v1"
OUT = ROOT / "results/astra_geometry_hybrid_v1"
TRAIN = Path("/mnt/c/Users/ephou/Downloads/xG/output/research-loop/data")
MODEL = "gpt-6-astra"
ARM = {"model": MODEL, "effort": geometry.SETTINGS["reasoning_effort"]}
CAP, WORKERS = 3.0, 8
MAX_OUTPUT = geometry.SETTINGS["max_output_tokens"]
RATES = {"input": 10.0, "cached_input": 1.0, "output": 50.0}
COORDINATES = ["shooter_lateral", "shooter_depth", "goalkeeper_lateral", "goalkeeper_depth"]
METHODS = {
    "hybrid_xg": "Astra positions + logistic regression",
    "astra": "Astra, direct xG",
    "luna": "Luna, direct xG",
    "known_positions": "Same hybrid model, recorded shooter and keeper positions",
    "original_no_header": "Logistic baseline, recorded positions and defenders",
    "original_logistic": "Original logistic baseline, also receives header flag",
    "sb_xg": "StatsBomb",
}


def prepare():
    selection = PRIOR / "main/selection.json"
    predictions = PRIOR / "main/predictions.csv"
    main_protocol = PRIOR / "main/protocol.json"
    rows = pd.DataFrame(direct.read(selection))
    prior = pd.read_csv(predictions)
    if len(rows) != 150 or rows.shot_id.duplicated().any():
        raise ValueError("Expected the complete original 150-shot cohort")
    prior = prior.set_index("shot_id", verify_integrity=True).loc[rows.shot_id].reset_index()
    for field in ("shot_id", "match_id", "goal"):
        if not rows[field].equals(prior[field]):
            raise ValueError(f"Saved prediction/selection mismatch in {field}")

    shots = [job for job in direct.read(main_protocol)["schedule"] if job["arm"] == "astra"]
    if [job["shot_id"] for job in shots] != rows.shot_id.tolist():
        raise ValueError("Astra image schedule does not match the original cohort")
    images = [job["images"][0] for job in shots]
    for image in images:
        if direct.sha(ROOT / image["path"]) != image["sha256"]:
            raise ValueError("Original diagram hash differs")

    reference = direct.read(ROOT / "results/direct_scope_suite/data/match_split.json")
    training_frames = []
    training_hashes = {name: reference["training_source_hashes"][name] for name in ("wc2018", "wc2022")}
    for source, expected in training_hashes.items():
        file = TRAIN / f"{source}.jsonl"
        if direct.sha(file) != expected:
            raise ValueError(f"Training source hash differs: {source}")
        training_frames.append(pd.read_json(file, lines=True, precise_float=True))
    training = pd.concat(training_frames, ignore_index=True)
    if len(training) != reference["training_shots"] or training.shot_id.duplicated().any():
        raise ValueError("Unexpected training sample")
    if set(training.match_id) & set(rows.match_id):
        raise ValueError("Train/test matches overlap")
    model = hybrid.make_model().fit(hybrid.reference_features(training), training.goal)
    direct.freeze(OUT / "protocol.json", {
        "evaluation": "Retrospective matched pilot on all 150 previously scored La Liga 2016/17 shots",
        "training": "Fixed 2018 and 2022 World Cup sample, no La Liga outcomes used in fitting",
        "training_hashes": training_hashes,
        "training_shots": len(training),
        "training_matches": sorted(int(mid) for mid in training.match_id.unique()),
        "test_matches": sorted(int(mid) for mid in rows.match_id.unique()),
        "fitted_model": hybrid.fitted_audit(model),
        "inputs": {"selection": direct.sha(selection), "predictions": direct.sha(predictions),
                   "image_schedule": direct.sha(main_protocol)},
        "code": {path.name: direct.sha(path) for path in (
            Path(__file__), ROOT / "hybrid_xg.py", ROOT / "geometry_diagnostic.py", Path(direct.__file__)
        )},
        "model": MODEL, "settings": geometry.SETTINGS, "prompt": geometry.PROMPT,
        "schema": geometry.SCHEMA, "statistical_features": hybrid.FEATURES,
        "cap_usd": CAP, "workers": WORKERS, "retries": 0, "rates_per_million": RATES,
        "primary": "Hybrid minus direct Astra Brier, paired 95% whole-match bootstrap",
        "secondary": "Recorded-position control, existing baselines, ranking, log loss and coordinate errors",
        "schedule": [{"shot_id": sid, **image} for sid, image in zip(rows.shot_id, images)],
    })
    return rows, prior, model


def payload(job):
    data = direct.body(ARM, geometry.PROMPT, (ROOT / job["path"]).read_bytes())
    data["max_output_tokens"] = MAX_OUTPUT
    data["text"]["format"].update(name="shot_geometry", schema=geometry.SCHEMA)
    return data


def cost(usage, conservative=False):
    inputs, outputs = usage["input_tokens"], usage["output_tokens"]
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    input_cost = inputs * 12.50 if conservative else (
        (inputs - cached) * RATES["input"] + cached * RATES["cached_input"]
    )
    return (input_cost + outputs * RATES["output"]) / 1_000_000


def validate(raw, input_bound):
    returned = raw["model"]
    if returned != MODEL and not returned.startswith(MODEL + "-"):
        raise ValueError("Unexpected returned model")
    if raw["status"] != "completed" or raw["reasoning"]["effort"] != ARM["effort"]:
        raise ValueError("Incomplete response or unexpected reasoning effort")
    usage = raw["usage"]
    for field, maximum in (("input_tokens", input_bound), ("output_tokens", MAX_OUTPUT)):
        if type(usage[field]) is not int or not 0 <= usage[field] <= maximum:
            raise ValueError("Missing or out-of-bound token usage")
    texts = [part["text"] for item in raw["output"] if item["type"] == "message"
             for part in item["content"] if part["type"] == "output_text"]
    return geometry.validate_response(json.loads("".join(texts)))


def generate():
    prepare()
    schedule = direct.read(OUT / "protocol.json")["schedule"]
    ledger_path = OUT / "attempts.json"
    lock_path = OUT / "generation.lock"
    lock_path.touch(exist_ok=False)
    try:
        ledger = direct.read(ledger_path, [])
        if any(entry["state"] != "completed" for entry in ledger):
            raise ValueError("Unreconciled prior charge or incomplete response. No retry.")
        if [entry["shot_id"] for entry in ledger] != [job["shot_id"] for job in schedule[:len(ledger)]]:
            raise ValueError("Prior attempts differ from the fixed schedule")
        key = direct.credential()

        def count(data):
            usage = direct.api("responses/input_tokens", key, {
                field: data[field] for field in ("model", "input", "text")
            })
            tokens = usage["input_tokens"]
            if type(tokens) is not int or not 0 < tokens < 60000:
                raise ValueError("No reliable input token count")
            return tokens + 1024

        def request(entry, data):
            try:
                raw = direct.api("responses", key, data)
                direct.save(OUT / entry["response"], raw)
                validate(raw, entry["input_bound"])
                return {"state": "completed", "usage": raw["usage"]}
            except Exception as error:
                message = str(error) if isinstance(error, (RuntimeError, ValueError)) else type(error).__name__
                return {"state": "failed_or_uncertain", "error": message}

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while len(ledger) < len(schedule):
                jobs = schedule[len(ledger):len(ledger) + WORKERS]
                data = [payload(job) for job in jobs]
                bounds = list(pool.map(count, data))
                spent = sum(cost(entry["usage"], conservative=True) for entry in ledger)
                batch = []
                for job, body, bound in zip(jobs, data, bounds):
                    reservation = cost({"input_tokens": bound, "output_tokens": MAX_OUTPUT}, conservative=True)
                    if spent + reservation > CAP:
                        break
                    spent += reservation
                    batch.append({"shot_id": job["shot_id"], "input_bound": bound, "state": "started",
                                  "response": f"responses/{job['shot_id']}.json",
                                  "request_sha256": direct.digest(body)})
                if not batch:
                    raise ValueError("Budget cap reached before the full fixed cohort")
                ledger.extend(batch)
                direct.save(ledger_path, ledger)
                futures = [pool.submit(request, entry, body) for entry, body in zip(batch, data)]
                for entry, future in zip(batch, futures):
                    entry.update(future.result())
                    direct.save(ledger_path, ledger)
                failed = [entry["error"] for entry in batch if entry["state"] != "completed"]
                if failed:
                    raise RuntimeError(f"Stopped after failed or uncertain requests: {failed}")
                estimated = sum(cost(entry["usage"]) for entry in ledger)
                print(f"{len(ledger)}/{len(schedule)} geometry readings, estimated ${estimated:.3f}", flush=True)
    finally:
        lock_path.unlink()


def evaluate():
    rows, prior, model = prepare()
    protocol = direct.read(OUT / "protocol.json")
    ledger = direct.read(OUT / "attempts.json", [])
    if [entry["shot_id"] for entry in ledger] != rows.shot_id.tolist():
        raise ValueError("Full fixed cohort required for scoring")
    answers = []
    for job, entry in zip(protocol["schedule"], ledger):
        if entry["state"] != "completed" or entry["request_sha256"] != direct.digest(payload(job)):
            raise ValueError("Incomplete or mismatched attempt")
        raw = direct.read(OUT / entry["response"])
        if raw["usage"] != entry["usage"]:
            raise ValueError("Saved usage disagrees with original response")
        answers.append({"shot_id": entry["shot_id"], **validate(raw, entry["input_bound"])})
    readings = pd.DataFrame(answers)
    prior["hybrid_xg"] = model.predict_proba(hybrid.vlm_features(readings))[:, 1]
    prior["known_positions"] = model.predict_proba(hybrid.reference_features(rows))[:, 1]
    metrics = []
    for method, label in METHODS.items():
        values = prior[method]
        metrics.append({"method": method, "label": label, "brier": brier_score_loss(prior.goal, values),
                        "auc": roc_auc_score(prior.goal, values), "log_loss": log_loss(prior.goal, values),
                        "mean_xg": values.mean()})
    table = pd.DataFrame(metrics)
    comparisons = [hybrid.match_bootstrap(prior, "hybrid_xg", method, n_boot=10000)
                   for method in ("astra", "known_positions", "original_logistic")]
    recorded = pd.DataFrame([hybrid.reference_positions(row) for row in rows.itertuples()], columns=COORDINATES)
    errors = readings[COORDINATES].astype(float) - recorded
    diagnostics = {}
    for label, cols in (("shooter", COORDINATES[:2]), ("keeper", COORDINATES[2:])):
        distances = np.hypot(errors[cols[0]], errors[cols[1]]).dropna()
        diagnostics[label] = {"compared": len(distances), "mean": float(distances.mean()),
                              "median": float(distances.median()), "p90": float(distances.quantile(.90))}
    summary = {
        "shots": len(prior), "goals": int(prior.goal.sum()), "matches": int(prior.match_id.nunique()),
        "training_shots": protocol["training_shots"], "comparisons": comparisons,
        "position_error_units": diagnostics, "metrics": metrics,
        "estimated_usd": sum(cost(entry["usage"]) for entry in ledger),
        "conservative_usd": sum(cost(entry["usage"], conservative=True) for entry in ledger),
    }
    table.to_csv(OUT / "scores.csv", index=False)
    prior.join(readings.drop(columns="shot_id")).to_csv(OUT / "predictions.csv", index=False)
    direct.save(OUT / "summary.json", summary)
    lines = [
        "# Astra and logistic regression for xG", "",
        f"This matched pilot reused {len(prior)} La Liga 2016/17 shot diagrams across "
        f"{summary['matches']} matches. {summary['goals']} shots scored. Astra read the shooter "
        "and goalkeeper positions. Logistic regression fitted on 3,068 World Cup shots estimated "
        "xG from distance, goal angle and goalkeeper geometry. The same fitted model also ran "
        "on recorded positions to measure the effect of reading coordinates from the diagrams.", "",
        "| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |",
        "|---|---:|---:|---:|---:|",
        *[f"| {row.label} | {row.brier:.4f} | {row.auc:.3f} | {row.log_loss:.4f} | {row.mean_xg:.1%} |"
          for row in table.itertuples()], "",
        "### Paired Brier differences", "",
        "Differences are hybrid minus comparator. The 95% intervals resample whole matches "
        "with predictions fixed. Only the direct Astra comparison was primary.", "",
        *[f"- {METHODS[row['second']]} — {row['brier_difference']:+.4f} "
          f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]." for row in comparisons], "",
        "### Position measurements", "",
        *[f"- {label.title()} — mean error {values['mean']:.2f} normalized pitch units and "
          f"90th percentile {values['p90']:.2f} across {values['compared']} paired positions."
          for label, values in diagnostics.items()], "",
        "Astra received the original diagram and geometry instructions. Only reported positions "
        "entered the hybrid. The training fit supplied imputation values for missing positions. "
        "The recorded-position control is separate from the image-only hybrid. The existing "
        "statistical baselines use recorded defender positions and the original baseline also "
        "receives the header flag. The cohort informed the prior selection of Astra, so this is "
        "an exploratory comparison on diagrams. Confirmation requires unused matches.", "",
        f"Estimated Astra cost ${summary['estimated_usd']:.2f} for {len(ledger)} geometry calls.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "generate", "evaluate"])
    args = parser.parse_args()
    if args.action == "prepare":
        rows, _, _ = prepare()
        print(f"Froze {len(rows)} original Astra diagrams and the disjoint training fit.")
    elif args.action == "generate":
        generate()
    else:
        evaluate()

