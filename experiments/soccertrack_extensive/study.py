"""Fixed 306-shot SoccerTrack comparison using existing audited sources."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import base64
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import requests

SOURCE = Path(__file__).resolve().parents[1] / "soccertrack_comparison"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as c
spec = importlib.util.spec_from_file_location("soccertrack_engine", SOURCE / "run.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
OLD = c.OUT
OUT = c.ROOT / "results/soccertrack_extensive_v1"
CAP, WORKERS = 10., 8
MODELS = {name: c.MODELS[name] for name in ("luna", "terra", "sol")}
MODELS["opus"] = {"provider": "anthropic", "model": "claude-opus-5-5", "rates": [4., .2, 20.]}


def frame_jobs():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = c.read(c.ROOT / "results/soccertrack_sol_expansion_v1/plan.json")
    jobs = []
    for job in plan["cohort"]:
        if job["reuse_original"]:
            jobs.append(job)
        else:
            jobs.append({**job, "image": f"results/soccertrack_extensive_v1/frames/{job['shot_id']}.png"})
    if len(jobs) != 306 or len({j["shot_id"] for j in jobs}) != 306:
        raise ValueError("The audited full cohort must have 306 unique shots")
    return jobs


def validate_sources():
    import prepare
    rows = c.read(OLD / "shots.json")
    for previous in c.read(OLD / "source_audit.json"):
        new_rows, audit = prepare.extract_match(previous["match_id"])
        if json.loads(json.dumps(audit)) != previous:
            raise ValueError("Source audit differs")
        if new_rows != [row for row in rows if row["match_id"] == previous["match_id"]]:
            raise ValueError("Audited shot labels differ")
    stored_models = c.read(OLD / "lomo_models.json")
    predictions = c.read(OLD / "lomo_predictions.json")
    for mid, arms in stored_models.items():
        training = [r for r in rows if r["match_id"] != mid]
        selected = [r for r in predictions if r["match_id"] == mid]
        for label, model in arms.items():
            if mid in model["train_matches"]:
                raise ValueError("Baseline training contains its evaluated match")
            fitted = c.fit_logistic(training, model["features"])
            np.testing.assert_allclose(c.predict(fitted, selected), [r[label] for r in selected],
                                       rtol=0, atol=1e-8)
    return rows


def prepare():
    rows = validate_sources()
    cohort = c.read(OUT / "frames.json")
    if [{k: item[k] for k in frame_jobs()[0]} for item in cohort] != frame_jobs():
        raise ValueError("Frame manifest differs from the fixed complete cohort")
    for job in cohort:
        if c.sha(c.ROOT / job["image"]) != job["sha256"]:
            raise ValueError("Input frame changed")
    lookup = {job["shot_id"]: job for job in cohort}
    for reused in c.read(OLD / "reused_sol.json"):
        image = (c.ROOT / lookup[reused["shot_id"]]["image"]).read_bytes()
        fingerprint = c.sha(c.ROOT / lookup[reused["shot_id"]]["image"])
        from hashlib import sha256
        original = sha256((base64.b64encode(image).decode() + "\n" + c.PROMPT).encode()).hexdigest()
        if fingerprint != reused["image_sha256"] or original != reused["input_sha256"]:
            raise ValueError("Original Sol image or prompt fingerprint differs")
    prior = c.read(OLD / "protocol.json")["budget"]["prior_conservative_usd"]["gpt-6.1-sol"]
    for folder in (OLD, c.ROOT / "results/soccertrack_sol_expansion_v1"):
        ledger = folder / "ledger.json"
        if ledger.exists() and c.read(ledger):
            raise ValueError("Another SoccerTrack campaign has new spending to reconcile")
    protocol = {
        "dataset": "SoccerTrack v2 v1.2", "shots": 306, "goals": 44, "matches": 10,
        "cohort": cohort, "models": MODELS, "prompt": c.PROMPT,
        "settings": {"reasoning_effort": "low", "max_output_tokens": c.MAX_OUTPUT,
                     "openai_image_detail": "high"},
        "primary": "All 306 shots, paired with distance/angle logistic regression trained on other nine matches",
        "secondary": "Official 64-shot test set versus six-training-match logistic, new 276, original 30, per-match results",
        "statistics": "10,000 whole-match paired bootstrap resamples, 98.75% intervals for four model-baseline contrasts",
        "budget": {"combined_cap_usd": CAP, "prior_conservative_usd": prior,
                   "scope": "This run plus the original SoccerTrack Sol pilot. No account-wide balance claim.",
                   "workers": WORKERS, "retries": 0, "input_cost_margin": .25},
        "preparation": "One full-pitch frame 0.32 seconds before SHOT/GOAL, longest side 1536, no shooter marker",
        "reused_sol": c.sha(OLD / "reused_sol.json"),
        "source_hashes": {name: c.sha(OLD / name) for name in
                          ("shots.json", "source_audit.json", "lomo_predictions.json",
                           "lomo_models.json", "baseline_models.json")},
        "code_hashes": {p.name: c.sha(p) for p in
                        (Path(__file__), Path(__file__).with_name("frames.py"), SOURCE / "common.py", SOURCE / "run.py")},
        "limitations": c.read(OLD / "protocol.json")["scope_caveats"],
        "attribution": "Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim, Keisuke Fujii. SoccerTrack v2. CC BY 4.0.",
        "dataset_source": "https://huggingface.co/datasets/atomscott/soccertrack-v2",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "pricing_sources": ["https://developers.openai.com/api/docs/pricing",
                            "https://platform.claude.com/docs/en/about-claude/pricing"],
    }
    c.freeze(OUT / "protocol.json", protocol)
    print(f"Protocol frozen for {len(rows)} shots, four models, with $10 combined cap", flush=True)
    return protocol


def payload(arm, job):
    config = MODELS[arm]
    content = (c.ROOT / job["image"]).read_bytes()
    if c.sha(c.ROOT / job["image"]) != job["sha256"]:
        raise ValueError("Frame changed before request")
    if config["provider"] == "openai":
        return engine.request_body(config["model"], content)
    return {
        "model": config["model"], "max_tokens": c.MAX_OUTPUT, "inference_geo": "global",
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(content).decode()}},
            {"type": "text", "text": c.PROMPT}]}],
    }


def anthropic_api(endpoint, key, body=None):
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    with requests.Session() as session:
        session.trust_env = False
        response = session.request("GET" if body is None else "POST",
                                   "https://api.anthropic.com/v1/" + endpoint,
                                   headers=headers, json=body, timeout=(20, 150))
        if not response.ok:
            raise RuntimeError(f"Anthropic HTTP {response.status_code}")
        return response.json()


def provider_keys():
    matches = set(re.findall(r"sk-ant-[A-Za-z0-9_-]+", (c.ROOT.parent / "keys.txt").read_text()))
    if len(matches) != 1:
        raise ValueError("One Claude API credential is required")
    return {"openai": c.credential(), "anthropic": matches.pop()}


def opus_answer(raw):
    import xg_vlm_pilot
    returned = raw["model"]
    if not (returned == MODELS["opus"]["model"] or returned.startswith(MODELS["opus"]["model"] + "-")):
        raise ValueError("Unexpected Claude returned model")
    if raw["stop_reason"] != "end_turn":
        raise ValueError("Incomplete Claude answer")
    text = "".join(block["text"] for block in raw["content"] if block["type"] == "text")
    value, reason = xg_vlm_pilot.strict_prediction(text)
    return {"xg": value, "reason": reason}


def usage_cost(raw_usage, rates, claude=False):
    if claude:
        usage = {"input_tokens": raw_usage["input_tokens"] + raw_usage.get("cache_read_input_tokens", 0),
                 "output_tokens": raw_usage["output_tokens"],
                 "input_tokens_details": {"cached_tokens": raw_usage.get("cache_read_input_tokens", 0)}}
        if raw_usage.get("cache_creation_input_tokens", 0):
            raise ValueError("Unexpected Claude cache write")
    else:
        usage = raw_usage
    return usage, c.token_cost(usage, rates), c.token_cost(usage, rates, True)


def execute():
    sys.path.insert(0, str(c.ROOT))
    protocol = prepare()
    path = OUT / "ledger.json"
    shared_lock = OLD / "runner.lock"
    lock = shared_lock.open("x")
    try:
        lock.write(json.dumps({"pid": os.getpid(), "campaign": str(OUT)}))
        lock.flush()
        os.fsync(lock.fileno())
        ledger = c.read(path) if path.exists() else []
        if any(entry["state"] != "completed" for entry in ledger):
            raise ValueError("Unresolved charged request. Reconcile it before any new generation.")
        keys = provider_keys()
        for arm, config in MODELS.items():
            if config["provider"] == "openai":
                c.api("models/" + config["model"], keys["openai"])
            else:
                anthropic_api("models/" + config["model"], keys["anthropic"])
        reused = {r["shot_id"] for r in c.read(OLD / "reused_sol.json")}
        done = {r["job_id"] for r in ledger}
        jobs = [(arm, job) for job in protocol["cohort"] for arm in MODELS
                if not (arm == "sol" and job["shot_id"] in reused)
                and f"{arm}__{job['shot_id']}" not in done]
        prior = protocol["budget"]["prior_conservative_usd"]

        def preflight(item):
            arm, job = item
            data = payload(arm, job)
            if MODELS[arm]["provider"] == "openai":
                counted = c.api("responses/input_tokens", keys["openai"],
                               {k: data[k] for k in ("model", "input", "text")})["input_tokens"]
            else:
                counted = anthropic_api("messages/count_tokens", keys["anthropic"],
                                         {k: data[k] for k in ("model", "messages", "output_config")})["input_tokens"]
            if type(counted) is not int or not 0 < counted < 60000:
                raise ValueError("Unexpected input token count")
            bound = max(counted + 512, math.ceil(counted * 1.25))
            reserved = (bound * MODELS[arm]["rates"][0] * 1.25
                        + c.MAX_OUTPUT * MODELS[arm]["rates"][2]) / 1e6
            return arm, job, data, bound, reserved

        def generate(item, entry):
            arm, job, data, bound, _ = item
            config = MODELS[arm]
            try:
                claude = config["provider"] == "anthropic"
                raw = (anthropic_api("messages", keys["anthropic"], data) if claude
                       else c.api("responses", keys["openai"], data))
                response = OUT / "responses" / (entry["job_id"] + ".json")
                c.save(response, raw)
                usage, estimated, conservative = usage_cost(raw["usage"], config["rates"], claude)
                if (usage["input_tokens"] > bound or usage["output_tokens"] > c.MAX_OUTPUT
                        or conservative > entry["reserved_usd"]):
                    raise ValueError("Token use exceeded reservation")
                answer = opus_answer(raw) if claude else engine.validate_answer(raw, config["model"])
                return {"state": "completed", "usage": usage, "estimated_usd": estimated,
                        "charged_bound_usd": conservative, **answer, "response_sha256": c.sha(response)}
            except Exception as error:
                return {"state": "uncertain", "error": type(error).__name__ + ": " + str(error)}

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for offset in range(0, len(jobs), WORKERS):
                items = list(pool.map(preflight, jobs[offset:offset+WORKERS]))
                exposure = prior + sum(e["charged_bound_usd"] for e in ledger)
                if exposure + sum(item[4] for item in items) > CAP:
                    raise ValueError("Combined $10 cap reached. No further requests sent.")
                entries = []
                for arm, job, data, bound, reserved in items:
                    entry = {"job_id": f"{arm}__{job['shot_id']}", "arm": arm,
                             "shot_id": job["shot_id"], "model": MODELS[arm]["model"],
                             "state": "reserved", "input_bound": bound, "reserved_usd": reserved,
                             "charged_bound_usd": reserved, "request_sha256": c.digest(data),
                             "image_sha256": job["sha256"], "started_unix": time.time()}
                    ledger.append(entry)
                    entries.append(entry)
                c.save(path, ledger)
                futures = {pool.submit(generate, item, entry): entry for item, entry in zip(items, entries)}
                for future in as_completed(futures):
                    futures[future].update(future.result())
                    c.save(path, ledger)
                if any(entry["state"] != "completed" for entry in entries):
                    raise RuntimeError("A response failed or was incomplete. Dispatch stopped.")
                if (offset // WORKERS) % 5 == 0 or offset + WORKERS >= len(jobs):
                    counts = {arm: sum(e["arm"] == arm for e in ledger) + (30 if arm == "sol" else 0)
                              for arm in MODELS}
                    print(json.dumps({"predictions": counts,
                                      "new_estimated_usd": round(sum(e["estimated_usd"] for e in ledger), 4),
                                      "tracked_conservative_usd": round(prior + sum(e["charged_bound_usd"] for e in ledger), 4)}), flush=True)
    finally:
        lock.close()
        shared_lock.unlink()


def bootstrap(rows, first, second):
    groups = sorted({r["match_id"] for r in rows})
    differences = np.array([sum((r[first]-r["goal"])**2 - (r[second]-r["goal"])**2
                                for r in rows if r["match_id"] == mid) for mid in groups])
    counts = np.array([sum(r["match_id"] == mid for r in rows) for mid in groups])
    sampled = np.random.default_rng(20261112).integers(0, len(groups), (10000, len(groups)))
    values = differences[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {"brier_difference": float(differences.sum() / counts.sum()),
            "confidence_level": .9875, "lower": float(np.quantile(values, .00625)),
            "upper": float(np.quantile(values, .99375))}


def analyze():
    prepare()
    ledger = c.read(OUT / "ledger.json")
    if len(ledger) != 1194 or any(e["state"] != "completed" for e in ledger):
        raise ValueError("All four models must cover the complete fixed sample before main scoring.")
    maps = {arm: {} for arm in MODELS}
    for entry in ledger:
        raw = c.read(OUT / "responses" / (entry["job_id"] + ".json"))
        assert c.sha(OUT / "responses" / (entry["job_id"] + ".json")) == entry["response_sha256"]
        answer = opus_answer(raw) if entry["arm"] == "opus" else engine.validate_answer(raw, entry["model"])
        assert answer["xg"] == entry["xg"]
        maps[entry["arm"]][entry["shot_id"]] = entry["xg"]
    maps["sol"].update({r["shot_id"]: r["vlm_xg"] for r in c.read(OLD / "reused_sol.json")})
    rows = c.read(OLD / "lomo_predictions.json")
    for row in rows:
        for arm in MODELS:
            row[arm] = maps[arm][row["shot_id"]]
    methods = ["logistic_distance_angle", "logistic_distance_angle_header", "train_constant"] + list(MODELS)
    metrics = {name: c.metrics([r["goal"] for r in rows], [r[name] for r in rows]) for name in methods}
    contrasts = {arm: bootstrap(rows, arm, "logistic_distance_angle") for arm in MODELS}
    groups = {"official_test_64": [r for r in rows if r["match_id"] in c.TEST],
              "new_276": [r for r in rows if r["shot_id"] not in {j["shot_id"] for j in c.read(OLD/"reused_sol.json")}]}
    official = c.read(OLD / "baseline_models.json")["models"]
    subgroups = {}
    for name, selected in groups.items():
        subset = [dict(r) for r in selected]
        if name == "official_test_64":
            values = c.predict(official["logistic_distance_angle"], subset)
            for row, value in zip(subset, values):
                row["logistic_distance_angle"] = value
        subgroups[name] = {method: c.metrics([r["goal"] for r in subset], [r[method] for r in subset])
                          for method in ["logistic_distance_angle"] + list(MODELS)}
    by_match = {mid: {method: c.metrics([r["goal"] for r in rows if r["match_id"] == mid],
                                       [r[method] for r in rows if r["match_id"] == mid])
                     for method in ["logistic_distance_angle"] + list(MODELS)}
                for mid in sorted({r["match_id"] for r in rows})}
    summary = {"metrics": metrics, "contrasts": contrasts, "subgroups": subgroups, "per_match": by_match,
               "new_calls": len(ledger), "estimated_usd": sum(e["estimated_usd"] for e in ledger),
               "conservative_usd": c.read(OUT/"protocol.json")["budget"]["prior_conservative_usd"]
                                   + sum(e["charged_bound_usd"] for e in ledger)}
    c.save(OUT / "summary.json", summary)
    with (OUT / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = {"luna": "GPT-6 Luna", "terra": "GPT-6.1 Terra", "sol": "GPT-6.1 Sol",
              "opus": "Claude Opus 5.5", "logistic_distance_angle": "Logistic, distance + angle",
              "logistic_distance_angle_header": "Logistic, also receives header flag",
              "train_constant": "Training goal-rate constant"}
    lines = ["# SoccerTrack v2 model comparison", "",
             "All methods were scored on the same 306 shots across ten university matches, including 44 goals. "
             "Each VLM received one panoramic frame 0.32 seconds before the shot, resized to at most 1536 px "
             "on the long side. Shooters were unmarked. The original image-only prompt assumed an average "
             "professional finisher and requested a raw probability and a one-sentence reason.", "",
             "| Method | Shots | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in methods:
        m = metrics[name]
        lines.append(f"| {labels[name]} | {m['n']} | {m['brier']:.4f} | {m['auc']:.3f} | {m['logloss']:.4f} | {m['mean_xg']:.1%} |")
    lines += ["", "The main logistic regression uses shot distance and goal-opening angle from recorded "
              "coordinates. Its standardization and L2 fit exclude the complete evaluated match. "
              "The header model is a secondary comparator with additional event information.", "",
              "## Paired Brier differences", "",
              "Intervals resample whole matches 10,000 times and use 98.75% coverage for the four primary "
              "model-versus-baseline comparisons. They condition on the fitted predictions. Ten matches "
              "limit generalization.", ""]
    for arm, contrast in contrasts.items():
        lines.append(f"- {labels[arm]} minus logistic. {contrast['brier_difference']:+.4f}, interval "
                     f"[{contrast['lower']:+.4f}, {contrast['upper']:+.4f}].")
    lines += ["", "## Official test matches", "",
              "This subset contains 64 shots in two matches. Its logistic model is fitted on the six "
              "official training matches. It is descriptive because there are only two test matches.", "",
              "| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |",
              "|---|---:|---:|---:|---:|"]
    for name, m in subgroups["official_test_64"].items():
        lines.append(f"| {labels[name]} | {m['brier']:.4f} | {m['auc']:.3f} | {m['logloss']:.4f} | {m['mean_xg']:.1%} |")
    lines += ["", "## Cost and limits", "",
              f"Completed {len(ledger)} new generation calls. Estimated new spending was "
              f"${summary['estimated_usd']:.3f}. Conservative exposure including the prior Sol pilot was "
              f"${summary['conservative_usd']:.3f}, below the $10 combined cap.", "",
              "The 30 earlier Sol predictions were reused with matching image and prompt fingerprints. "
              "All probabilities remain raw. The complete cohort was already audited, and this is "
              "exploratory evaluation. BAS SHOT/GOAL labels can omit blocked or headed attempts classified "
              "under other actions, and unidentified penalties can remain. Video timing is aligned to "
              "within a few frames, the shooter is unmarked, and the panoramic view makes players small. "
              "The prompt's professional-finisher assumption differs from the university population.", "",
              "Source and attribution. SoccerTrack v2 by Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim "
              "and Keisuke Fujii. Frames were extracted and resized under CC BY 4.0. "
              "https://huggingface.co/datasets/atomscott/soccertrack-v2 "
              "https://creativecommons.org/licenses/by/4.0/", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "contrasts": contrasts,
                      "estimated_usd": summary["estimated_usd"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "analyze"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
    elif stage == "run":
        execute()
    else:
        analyze()

