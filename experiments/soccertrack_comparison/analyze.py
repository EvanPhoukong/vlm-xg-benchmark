"""Offline matched-cohort scoring; never dispatches API calls or alters VLM probabilities."""
from collections import defaultdict
import csv
import json

from common import (OUT, ROOT, TRAIN, VALIDATION, TEST, MODELS, CAP,
                    fit_logistic, predict, metrics, read, freeze, save, sha, exposure)
from run import check_protocol, schedule

FEATURES = {
    "logistic_distance": ["distance_m"],
    "logistic_distance_angle": ["distance_m", "angle_rad"],
    "logistic_distance_angle_header": ["distance_m", "angle_rad", "header"],
}


def score(rows, columns):
    return {name: metrics([r["goal"] for r in rows], values) for name, values in columns.items()}


def slices(rows, p):
    bins = [(0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .7), (.7, 1.00000001)]
    reliability = []
    for lower, upper in bins:
        indexes = [i for i, value in enumerate(p) if lower <= value < upper]
        if indexes:
            reliability.append({"bin": f"{lower:g}-{min(upper, 1):g}",
                **metrics([rows[i]["goal"] for i in indexes], [p[i] for i in indexes])})
    result = {"reliability": reliability}
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups["match:" + r["match_id"]].append(i)
        groups["body:" + ("header" if r["header"] else "not_header")].append(i)
        label = "0-10m" if r["distance_m"] < 10 else "10-20m" if r["distance_m"] < 20 else "20m+"
        groups["distance:" + label].append(i)
    result["groups"] = {name: metrics([rows[i]["goal"] for i in ids], [p[i] for i in ids]) for name, ids in groups.items()}
    return result


def markdown_table(scores):
    lines = ["| Method | Shots | Brier ↓ | Log loss ↓ | AUC ↑ | Mean xG |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in scores.items():
        auc = "—" if m["auc"] is None else f"{m['auc']:.3f}"
        lines.append(f"| {name} | {m['n']} | {m['brier']:.5f} | {m['logloss']:.4f} | {auc} | {m['mean_xg']:.3f} |")
    return "\n".join(lines)


def main():
    protocol = check_protocol()
    rows = read(OUT / "shots.json")
    by_id = {r["shot_id"]: r for r in rows}
    training = [r for r in rows if r["match_id"] in TRAIN]
    validation = [r for r in rows if r["match_id"] in VALIDATION]
    test = [r for r in rows if r["match_id"] in TEST]
    train_rate = sum(r["goal"] for r in training) / len(training)
    models = {name: fit_logistic(training, features) for name, features in FEATURES.items()}
    freeze(OUT / "baseline_models.json", {"constant": train_rate, "models": models,
        "training_shot_ids": [r["shot_id"] for r in training], "protocol_sha256": sha(OUT / "protocol.json")})
    baseline_rows = {}
    all_pred = {name: predict(model, rows) for name, model in models.items()}
    all_pred["train_constant"] = [train_rate] * len(rows)
    for i, r in enumerate(rows):
        baseline_rows[r["shot_id"]] = {name: p[i] for name, p in all_pred.items()}
    official = {}
    for label, selected in (("validation", validation), ("test", test)):
        columns = {name: [baseline_rows[r["shot_id"]][name] for r in selected] for name in all_pred}
        official[label] = score(selected, columns)
    # Every LOMO row uses a completely disjoint fitting/scaling fold.
    cv_rows, cv_models = [], {}
    for mid in sorted({r["match_id"] for r in rows}):
        fit = [r for r in rows if r["match_id"] != mid]
        held = [r for r in rows if r["match_id"] == mid]
        cv_models[mid] = {name: fit_logistic(fit, features) for name, features in FEATURES.items()}
        predictions = {name: predict(model, held) for name, model in cv_models[mid].items()}
        predictions["train_constant"] = [sum(r["goal"] for r in fit) / len(fit)] * len(held)
        for i, row in enumerate(held):
            cv_rows.append({**row, **{name: p[i] for name, p in predictions.items()}})
    save(OUT / "lomo_models.json", cv_models)
    save(OUT / "lomo_predictions.json", cv_rows)
    cv_scores = score(cv_rows, {name: [r[name] for r in cv_rows] for name in all_pred})
    reused = read(OUT / "reused_sol.json")
    ledger = read(OUT / "ledger.json") if (OUT / "ledger.json").exists() else []
    predictions, repeats = defaultdict(dict), defaultdict(dict)
    for entry in reused:
        predictions[entry["arm"]][entry["shot_id"]] = entry["vlm_xg"]
    for entry in ledger:
        if entry["state"] == "completed":
            path = OUT / f"responses/{entry['job_id']}.json"
            if sha(path) != entry["response_sha256"]:
                raise ValueError("Raw response hash changed")
            destination = repeats if entry["condition"] == "exact_repeat" else predictions
            destination[entry["arm"]][entry["shot_id"]] = entry["xg"]
    image_rows = [by_id[item["shot_id"]] for item in protocol["cohort"]]
    image_ids = [r["shot_id"] for r in image_rows]
    columns = {name: [baseline_rows[sid][name] for sid in image_ids] for name in all_pred}
    availability, pairwise, consistency = {}, {}, {}
    for arm, config in MODELS.items():
        coverage = sum(sid in predictions[arm] for sid in image_ids)
        availability[arm] = {"model": config["model"], "original_completed": coverage, "original_expected": len(image_ids),
            "repeat_completed": len(repeats[arm]), "repeat_expected": len(protocol["consistency"]["shot_ids"]),
            "source": "verified earlier pilot" if arm == "sol" else "new campaign",
            "conservative_exposure_usd": exposure(ledger, config["model"], protocol["budget"]["prior_conservative_usd"].get(config["model"], 0.))}
        if coverage == len(image_ids):
            columns[config["model"]] = [predictions[arm][sid] for sid in image_ids]
        # Partial methods are evaluated only on their explicitly matched subset, never in the main leaderboard.
        selected = [r for r in image_rows if r["shot_id"] in predictions[arm]]
        if selected:
            p = [predictions[arm][r["shot_id"]] for r in selected]
            b = [baseline_rows[r["shot_id"]][protocol["primary_baseline"]] for r in selected]
            sm, bm = metrics([r["goal"] for r in selected], p), metrics([r["goal"] for r in selected], b)
            pairwise[arm] = {"model": config["model"], "complete_cohort": coverage == len(image_ids),
                "shot_ids": [r["shot_id"] for r in selected], "model_scores": sm, "baseline_scores": bm,
                "brier_difference_model_minus_baseline": sm["brier"]-bm["brier"],
                "per_match": {mid: {
                    "n": sum(r["match_id"] == mid for r in selected),
                    "brier_difference": sum((pi-r["goal"])**2 - (bi-r["goal"])**2 for r, pi, bi in zip(selected, p, b) if r["match_id"] == mid) /
                        sum(r["match_id"] == mid for r in selected)} for mid in sorted({r["match_id"] for r in selected})},
                "uncertainty": "Descriptive only: two test match clusters; no CI or population superiority claim."}
        pairs = [(predictions[arm][sid], p) for sid, p in repeats[arm].items() if sid in predictions[arm]]
        if pairs:
            consistency[arm] = {"n": len(pairs), "mean_absolute_change_pp": sum(abs(a-b)*100 for a,b in pairs)/len(pairs),
                "max_absolute_change_pp": max(abs(a-b)*100 for a,b in pairs), "averaged_for_accuracy": False}
    image_scores = score(image_rows, columns)
    save(OUT / "subgroups.json", {name: slices(image_rows, p) for name, p in columns.items()})
    with (OUT / "paired_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["shot_id", "match_id", "goal", "distance_m", "angle_rad", "header", "actor_reconciled"] + list(columns)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(image_rows):
            writer.writerow({**{k: row[k] for k in fields if k in row}, **{name: p[i] for name, p in columns.items()}})
    audits = read(OUT / "source_audit.json")
    source_summary = {"unique_shots": len(rows), "goals": sum(r["goal"] for r in rows),
        "bas_shot_goal_events": sum(a["bas_shot_goal_events"] for a in audits),
        "duplicates_removed": sum(len(a["duplicate_shot_ids"]) for a in audits),
        "excluded_events": sum(len(a["excluded"]) for a in audits),
        "unique_repaired_shots": sum(r["actor_reconciled"] for r in rows),
        "training_shots": len(training), "validation_shots": len(validation), "test_shots": len(test)}
    checks = {
        "unique_shot_ids": len(by_id) == len(rows),
        "train_validation_test_disjoint": not (set(TRAIN)&set(TEST) or set(TRAIN)&set(VALIDATION) or set(VALIDATION)&set(TEST)),
        "all_baseline_fits_training_only": all(set(m["train_matches"]) == set(TRAIN) for m in models.values()),
        "all_lomo_folds_disjoint": all(mid not in m["train_matches"] for mid, ms in cv_models.items() for m in ms.values()),
        "all_probabilities_valid": all(0 <= v <= 1 for p in columns.values() for v in p),
        "all_source_zone_checks": all(r["coordinate_zone_agrees"] for r in rows),
        "pilot_label_agreement": all(r["goal"] == next(s["goal"] for s in read(ROOT / "results/soccertrack_sol_v1/manifest.json")["shots"] if s["shot_id"] == r["shot_id"]) for r in image_rows),
        "all_model_caps_respected": all(a["conservative_exposure_usd"] <= CAP for a in availability.values()),
        "all_scheduled_jobs_only": all(e["job_id"] in {f"{j['arm']}__{j['condition']}__{j['shot_id']}" for j in schedule(protocol, list(MODELS))} for e in ledger),
        "no_uncertain_requests": all(e["state"] == "completed" for e in ledger),
    }
    if not all(checks.values()):
        raise ValueError("Analysis audit failed: " + str(checks))
    save(OUT / "audit.json", checks)
    summary = {"status": "complete" if all(a["original_completed"] == a["original_expected"] and a["repeat_completed"] == a["repeat_expected"] for a in availability.values()) else "partial_offline_analysis",
        "data": source_summary, "availability": availability, "official_split_baselines": official,
        "lomo_baselines": cv_scores, "paired_image_cohort": image_scores,
        "paired_contrasts": pairwise, "consistency": consistency,
        "new_generation_calls": len(ledger), "new_estimated_usd": sum(e.get("estimated_usd", 0.) for e in ledger),
        "protocol_sha256": sha(OUT / "protocol.json")}
    save(OUT / "summary.json", summary)
    comparison = pairwise.get("sol")
    lines = ["# SoccerTrack v2: expanded comparison", "",
        "Status: " + summary["status"] + ". No results are imputed for unrun models.", "",
        f"Audited {len(rows)} unique SHOT/GOAL situations from 10 matches ({source_summary['goals']} goals). "
        f"The official six training matches supply {len(training)} shots; validation has {len(validation)}; "
        f"the two official test matches have {len(test)}. The paired image cohort remains the earlier 30-shot pilot, with 7 goals.", "",
        "## Existing paired image results", "", markdown_table(image_scores), "",
        "Every row above uses exactly the same 30 shots. Sol predictions are reused only after verifying the original image and prompt fingerprints. "
        "The primary baseline is distance + goal-opening angle logistic regression fitted on the six official training matches. "
        "The header baseline has additional explicit body-part information. All coefficients and standardization use training data only; no hyperparameter search or validation refit.", ""]
    if comparison:
        lines.extend([f"Sol minus the primary logistic baseline Brier difference: {comparison['brier_difference_model_minus_baseline']:+.5f}. "
            "Lower is better. This is a point comparison on two matches, not evidence of population superiority or equivalence.", ""])
    lines.extend(["## Model coverage and spending", "", "| Model | Original predictions | Exact repeats | Conservative tracked exposure |",
        "|---|---:|---:|---:|"])
    for a in availability.values():
        lines.append(f"| {a['model']} | {a['original_completed']}/{a['original_expected']} | {a['repeat_completed']}/{a['repeat_expected']} | ${a['conservative_exposure_usd']:.4f} / $10 |")
    lines.extend(["", f"New paid calls in this extension: {len(ledger)}; estimated new spending ${summary['new_estimated_usd']:.4f}. "
        "Exposure includes the earlier verified SoccerTrack Sol pilot. This is experiment accounting, not an account-wide API balance. "
        "The runner reserves input plus maximum output (including reasoning) before dispatch, keeps uncertain requests charged at the full reservation, and has no automatic retries. "
        "One single-writer lock prevents concurrent budget races.", "",
        "## Larger baseline-only checks", "", "Official test matches (all reconciled SHOT/GOAL events):", "", markdown_table(official["test"]), "",
        "Secondary leave-one-match-out results across all 10 matches:", "", markdown_table(cv_scores), "",
        "The cross-validation fits both scaling and coefficients on the other nine matches. These are baseline robustness results; "
        "the VLM was not evaluated on all 10 matches. No confidence intervals treating correlated shots as independent are reported.", "",
        "## Data and interpretation limits", "",
        f"- Removed {source_summary['duplicates_removed']} duplicate SHOT/GOAL annotations; {source_summary['excluded_events']} events excluded for an unresolved source match.",
        f"- Reconciled {source_summary['unique_repaired_shots']} shot actor IDs using the unique raw primary-shot record at the aligned timestamp. The original labels/images remain unchanged. See source_audit.json.",
        "- Raw provider x runs across the pitch width and y along the attacking direction. Converted with a 68 × 105 m pitch; penalty-area tag checks passed. This is an internally checked interpretation, without an independent tracking/video geometry audit.",
        "- Geometry is recorded at contact; images are 0.32 seconds earlier. The image does not identify the shooter with a marker. These input differences can affect the comparison.",
        "- The SHOT/GOAL event definition may omit blocked/header attempts placed in other BAS classes. Penalties have no dedicated label and are not silently excluded.",
        "- The prompt assumes an average professional finisher while these are university-level matches; it is retained for an exact comparison with the pilot.",
        "- These test images and earlier Sol results were already inspected. This is an exploratory extension, not a new independent confirmation study.",
        "- No VLM probability is calibrated, blended, clipped, averaged, or supplied with the baseline estimate. Reliability curves are diagnostics only.", "",
        "## Execution", "",
        ("All scheduled model conditions have completed." if summary["status"] == "complete" else
         "Model coverage is incomplete. The preparation session could read local files but could not resolve external API/download hosts. "
         "See runner_status.json for the most recent execution state. The broader model/repeat study remains pending where the coverage table is incomplete. "
         "The local video/tracking files are Git LFS pointers; no additional real frames were invented or downloaded."), "",
        "From the repository root, in an environment with NumPy and network access:", "", "```bash",
        "python experiments/soccertrack_comparison/prepare.py",
        "python -m unittest discover -s experiments/soccertrack_comparison -p 'test_*.py' -v",
        "python experiments/soccertrack_comparison/run.py --preflight",
        "python experiments/soccertrack_comparison/run.py --execute",
        "python experiments/soccertrack_comparison/analyze.py", "```", "",
        "The original pilot is cached; 130 new calls are scheduled: 90 original predictions (Luna/Terra/Astra) plus 40 exact repeats (all four models). "
        "The common test cohort and prompt are frozen; no test-set prompt selection or spending to a target score. "
        "A larger VLM study needs new pre-shot images from additional match groups and a separate fixed protocol, not repeated optimization of these two matches.", "",
        "Full numeric output: summary.json, paired_predictions.csv, subgroups.json, baseline_models.json, lomo_predictions.json, audit.json.", "",
        "Pricing checked against https://developers.openai.com/api/docs/pricing on 2026-11-13 UTC. "
        "Dataset provenance: local SoccerTrack v2 README.md, CHANGES_v1_1.md, CHANGES_v1_2.md, and hashed BAS/player_nodes files.", ""])
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"data": source_summary, "paired": image_scores, "status": summary["status"], "audit": checks}, indent=2))


if __name__ == "__main__":
    main()
