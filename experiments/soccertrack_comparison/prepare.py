"""Prepare audited local SoccerTrack labels and a fixed, leak-free comparison protocol."""
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path

from common import (DATA, OUT, ROOT, PILOT, TRAIN, VALIDATION, TEST, MODELS,
                    MAX_OUTPUT, CAP, SEED, PROMPT, digest, freeze, read, save, sha, token_cost)

PERIOD = {"FIRST_HALF": 1, "SECOND_HALF": 2}
SHOT_TAGS = {"shot", "shotOnTarget", "shotMissed", "shotBlocked", "goal"}


def geometry(width_fraction, attacking_length_fraction):
    """Provider event coordinates: transverse x, attack-normalized longitudinal y."""
    x, y = float(width_fraction), float(attacking_length_fraction)
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Invalid normalized event coordinates")
    lateral, forward = (x - .5) * 68, (1 - y) * 105
    return math.hypot(lateral, forward), math.atan2(7.32 * forward, forward**2 + lateral**2 - 3.66**2)


def align(actions, nodes):
    """Recover raw-to-v1.2 half clock offsets from non-shot events only."""
    times = defaultdict(list)
    for node in nodes:
        half = PERIOD.get(node["event_period"])
        if half and node["player_id"]:
            times[half, node["player_id"]].append(int(node["event_time"]))
    for values in times.values():
        values.sort()
    candidates = defaultdict(list)
    for action in actions:
        if action["label"] in ("SHOT", "GOAL") or not action.get("player_id"):
            continue
        half = int(action["gameTime"].split(" - ")[0])
        values = times[half, str(action["player_id"])]
        if not values:
            continue
        position = int(action["position"])
        nominal = position + (half-1) * 2700000
        k = bisect_left(values, nominal)
        nearest = min(values[max(0, k-1):k+1], key=lambda t: abs(t-nominal))
        if abs(nearest-nominal) < 10000:
            candidates[half].append(nearest-position)
    result = {}
    for half in (1, 2):
        counts = Counter(candidates[half])
        offset, support = counts.most_common(1)[0]
        if support < 50 or support / len(candidates[half]) < .5:
            raise ValueError(f"Ambiguous raw/BAS alignment in half {half}")
        result[half] = {"offset_ms": offset, "support": support,
                        "candidates": len(candidates[half]), "method": "non-shot/non-goal actor/time mode"}
    return result


def extract_match(mid):
    bas_path = DATA / f"bas/{mid}/{mid}_12_class_events.json"
    node_path = DATA / f"raw/{mid}/{mid}_player_nodes.csv"
    actions = read(bas_path)["actions"]
    with node_path.open(encoding="utf-8-sig", newline="") as stream:
        nodes = list(csv.DictReader(stream))
    offsets = align(actions, nodes)
    index, by_time = defaultdict(list), defaultdict(list)
    for i, node in enumerate(nodes):
        if node["event_period"] in PERIOD and node["player_id"]:
            index[PERIOD[node["event_period"]], node["player_id"], int(node["event_time"])].append((i, node))
            if set(node["event_types"].split()).intersection(SHOT_TAGS):
                by_time[PERIOD[node["event_period"]], int(node["event_time"])].append((i, node))
    rows, excluded, duplicate, repairs = {}, [], [], []
    for action in actions:
        if action["label"] not in ("SHOT", "GOAL"):
            continue
        half = int(action["gameTime"].split(" - ")[0])
        sid = f"{mid}_{half}_{action['frame']}"
        target = int(action["position"]) + offsets[half]["offset_ms"]
        found = [pair for dt in (-1, 0, 1) for pair in index[half, str(action.get("player_id")), target+dt]]
        found = [(i, n) for i, n in found if set(n["event_types"].split()).intersection(SHOT_TAGS)]
        repaired = False
        if not found:
            found = [pair for dt in (-1, 0, 1) for pair in by_time[half, target+dt]]
            repaired = bool(found)
        positions = {(n["player_id"], n["x"], n["y"]) for _, n in found}
        if len(positions) != 1:
            excluded.append({"shot_id": sid, "reason": "no unique actor/time coordinate match", "matches": len(found)})
            continue
        i, node = found[0]
        tags = set(node["event_types"].split())
        if int("goal" in tags) != int(action["label"] == "GOAL"):
            raise ValueError("Raw primary-shot outcome disagrees with BAS")
        if repaired:
            repairs.append({"shot_id": sid, "bas_actor": action.get("player_id"),
                            "raw_shooter": node["player_id"], "method": "unique primary-shot node at aligned time"})
        distance, angle = geometry(node["x"], node["y"])
        lateral, forward = abs((float(node["x"])-.5)*68), (1-float(node["y"]))*105
        inside_tag = any(t in tags for t in ("shotInPenaltyArea", "shotOnTargetInPenaltyArea", "shotMissedInPenaltyArea", "shotBlockedInPenaltyArea", "goalInPenaltyArea"))
        zone_agrees = not inside_tag or (forward <= 17.5 and lateral <= 21.16)
        row = {"shot_id": sid, "match_id": mid, "half": half, "event_frame": action["frame"],
            "position_ms": int(action["position"]), "goal": int(action["label"] == "GOAL"),
            "distance_m": distance, "angle_rad": angle, "header": int("header" in tags),
            "raw_width_fraction": float(node["x"]), "raw_attacking_length_fraction": float(node["y"]),
            "raw_event_id": node["id"], "raw_csv_row": i+2, "raw_time_ms": int(node["event_time"]),
            "actor_id": node["player_id"], "bas_actor_id": str(action.get("player_id")),
            "actor_reconciled": repaired, "penalty_area_tag": inside_tag,
            "coordinate_zone_agrees": zone_agrees,
            "split": "train" if mid in TRAIN else "validation" if mid in VALIDATION else "test"}
        if sid in rows:
            old = rows[sid]
            if any(old[f] != row[f] for f in ("distance_m", "angle_rad", "actor_id")):
                raise ValueError("Conflicting same-frame shot coordinates")
            duplicate.append(sid)
            old["goal"] = max(old["goal"], row["goal"])
        else:
            rows[sid] = row
    rows = sorted(rows.values(), key=lambda r: (r["half"], r["event_frame"]))
    if any(not r["coordinate_zone_agrees"] for r in rows):
        raise ValueError("Provider coordinate convention failed penalty-area tag audit")
    return rows, {"match_id": mid, "bas_sha256": sha(bas_path), "nodes_sha256": sha(node_path),
                  "clock_offsets": offsets, "bas_shot_goal_events": sum(a["label"] in ("SHOT", "GOAL") for a in actions),
                  "unique_shots": len(rows), "goals": sum(r["goal"] for r in rows),
                  "duplicate_shot_ids": duplicate, "actor_repairs": repairs, "excluded": excluded}


def main():
    rows, audits = [], []
    for mid in sorted(TRAIN + VALIDATION + TEST):
        shots, audit = extract_match(mid)
        rows.extend(shots)
        audits.append(audit)
        print(mid, len(shots), "shots", sum(r["goal"] for r in shots), "goals; excluded", len(audit["excluded"]), flush=True)
    # This source-level check supports (but does not independently prove) the raw event coordinate interpretation.
    freeze(OUT / "shots.json", rows)
    freeze(OUT / "source_audit.json", audits)
    pilot = read(PILOT)
    by_id = {r["shot_id"]: r for r in rows}
    cohort = []
    for item in pilot["shots"]:
        path = PILOT.parent / item["frame"]
        if sha(path) != item["sha256"] or item["frame_index"] >= item["event_frame"]:
            raise ValueError("Pilot frame integrity or past-only check failed")
        if item["shot_id"] not in by_id or by_id[item["shot_id"]]["goal"] != item["goal"]:
            raise ValueError("Pilot shot could not be reconciled with local BAS/raw labels")
        cohort.append({"shot_id": item["shot_id"], "image": str(path.relative_to(ROOT)).replace("\\", "/"),
                       "sha256": item["sha256"], "frame_index": item["frame_index"], "event_frame": item["event_frame"]})
    cohort.sort(key=lambda r: r["shot_id"])
    # Verify exact original prompt and image fingerprints before reusing Sol's pilot calls.
    import base64
    cache = [__import__("json").loads(line) for line in (ROOT / "results/soccertrack_gpt_6.1_sol/vlm_cache.jsonl").read_text().splitlines()]
    cache_by_id = {r["shot_id"]: r for r in cache}
    reused = []
    for item in cohort:
        cached = cache_by_id[item["shot_id"]]
        b64 = base64.b64encode((ROOT / item["image"]).read_bytes()).decode()
        original_hash = __import__("hashlib").sha256((b64 + "\n" + PROMPT).encode()).hexdigest()
        if original_hash != cached["input_sha256"]:
            raise ValueError("Original Sol request fingerprint differs")
        if type(cached["vlm_xg"]) not in (int, float) or not 0 <= cached["vlm_xg"] <= 1:
            raise ValueError("Invalid cached raw Sol probability")
        reused.append({**cached, "model": MODELS["sol"]["model"], "arm": "sol", "condition": "original",
                       "source": "verified_original_pilot", "image_sha256": item["sha256"]})
    prior = sum(token_cost({"input_tokens": c["input_tokens"], "output_tokens": c["output_tokens"]}, MODELS["sol"]["rates"], True) for c in cache)
    protocol = {"dataset": "SoccerTrack v2 v1.2", "seed": SEED,
        "train_matches": list(TRAIN), "validation_matches": list(VALIDATION), "test_matches": list(TEST),
        "primary_baseline": "logistic_distance_angle", "secondary_baselines": ["train_constant", "logistic_distance", "logistic_distance_angle_header"],
        "baseline_settings": {"ridge": 1., "standardize": "training only", "tuning": "none", "refit_with_validation": False},
        "primary_metric": "raw Brier", "secondary_metrics": ["log_loss", "AUC", "mean_bias", "reliability_bins", "per_match", "distance_bins", "headers"],
        "inference": "Two test matches only: descriptive paired differences, no population CI or significance/superiority claim.",
        "cv": "Leave-one-match-out across all 10 matches is secondary baseline-only robustness; each fold fits its scaler and coefficients on the other 9.",
        "cohort": cohort, "models": MODELS, "prompt": PROMPT, "image_detail": "high", "reasoning_effort": "low", "max_output_tokens": MAX_OUTPUT,
        "budget": {"per_model_cap_usd": CAP, "scope": "this SoccerTrack comparison plus the verified earlier SoccerTrack Sol pilot; other historical project/account spend is not an account balance",
                   "prior_conservative_usd": {"gpt-6.1-sol": prior}, "input_margin": .25, "automatic_retries": 0,
                   "uncertain_request": "retain full reservation and stop; never automatically retry", "concurrency": 1},
        "consistency": {"shot_ids": [r["shot_id"] for r in cohort[:10]], "condition": "exact_repeat", "aggregate_predictions": False},
        "novelty": "Retrospective extension of an inspected 30-shot pilot; not an untouched test set. Baseline specification fixed before fitting in this extension.",
        "coordinate_caveat": "Raw event x is pitch-width fraction, y is attacking-direction pitch-length fraction. Checked against raw penalty-area tags, without tracking/video verification. Baseline uses at-event annotation; VLM sees a frame 0.32 seconds earlier.",
        "scope_caveats": ["Unidentified penalties may be included; do not call this a non-penalty cohort.", "BAS SHOT/GOAL protocol may omit blocked/header attempts assigned other event classes.", "Professional-finisher instruction on university-level matches is retained for an exact pilot comparison.", "No calibration, hybrid prior, fine-tuning or probability averaging.", "Header baseline has explicit body-part information absent from image-only prompts."],
        "shots_sha256": sha(OUT / "shots.json"), "sources_sha256": sha(OUT / "source_audit.json"), "pilot_manifest_sha256": sha(PILOT),
        "pricing_source": "https://developers.openai.com/api/docs/pricing", "pricing_checked_utc": "2026-11-13"}
    freeze(OUT / "protocol.json", protocol)
    freeze(OUT / "reused_sol.json", reused)
    print("Prepared", len(rows), "audited shots;", len(cohort), "verified paired images; no API calls", flush=True)


if __name__ == "__main__":
    main()
