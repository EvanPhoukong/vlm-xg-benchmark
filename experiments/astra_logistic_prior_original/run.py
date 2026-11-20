"""Apply the La Liga Astra hybrid to the original 298-shot benchmark."""
import argparse
import csv
import importlib.util
from pathlib import Path
from statistics import mean

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "experiments/astra_logistic_prior/run.py"
spec = importlib.util.spec_from_file_location("hybrid_runner", ENGINE)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
direct = engine.direct
original = engine.transfer.original

OUT = ROOT / "results/astra_logistic_prior_original_v1"
ARCHIVES = {
    "gemma": ROOT / "shots_gemma_4_26b.csv",
    "sonnet": ROOT / "shots_sonnet_5_5.csv",
    "opus": ROOT / "shots_opus5_5.csv",
    "gemini": ROOT / "shots_gemini_3_8_flash.csv",
}
IMAGE_MANIFEST = ROOT / "results/claude_fable_5_1/inputs.json"
LA_LIGA = ROOT / "results/astra_logistic_prior_v1"
CAP = 4.00
LABELS = {
    "hybrid": "Astra + logistic xG",
    "gemma": "Gemma 4 (26B)",
    "sonnet": "Claude Sonnet 5.5",
    "opus": "Claude Opus 5.5",
    "gemini": "Gemini 3.8 Flash",
    "baseline_xg": "Original logistic baseline",
    "sb_xg": "StatsBomb xG",
}

# The previous run's generation and validation code are frozen by its protocol.
# Bind its runner to this study's separate artifact directory and budget.
engine.OUT = OUT
engine.BASELINE = ARCHIVES["gemini"]
engine.CAP = CAP


def prepare():
    tables = {}
    for method, path in ARCHIVES.items():
        frame = pd.read_csv(path, float_precision="round_trip").set_index("shot_id")
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate original shot IDs in {path.name}")
        tables[method] = frame.loc[frame.vlm_xg.notna()]

    common = set.intersection(*(set(frame.index) for frame in tables.values()))
    reference = tables["gemini"].loc[lambda frame: frame.index.isin(common)]
    columns = ["match_id", "goal", "baseline_xg", "sb_xg"]
    if (len(reference), reference.goal.sum(), reference.match_id.nunique()) != (298, 42, 63):
        raise ValueError("The archive does not reproduce the published cohort")
    for method, frame in tables.items():
        if not reference[columns].equals(frame.loc[reference.index, columns]):
            raise ValueError(f"Original baseline or outcome mismatch in {method}")

    image_manifest = direct.read(IMAGE_MANIFEST)
    la_liga_protocol = direct.read(LA_LIGA / "protocol.json")
    prompt = direct.original_prompt()
    if image_manifest["prompt"] != prompt or la_liga_protocol["original_prompt"] != prompt:
        raise ValueError("The original image prompt differs")
    if la_liga_protocol["hybrid_addition"] != engine.PRIOR:
        raise ValueError("The tested hybrid prompt differs")
    schedule = []
    for sid, row in reference.iterrows():
        path = ROOT / "results/images" / f"{sid}.png"
        image_hash = direct.sha(path)
        if image_hash != image_manifest["image_sha256_by_shot"][sid]:
            raise ValueError("The archived diagram differs from the image manifest")
        schedule.append({
            "shot_id": sid, "arm": "hybrid", "image": str(path.relative_to(ROOT)),
            "image_sha256": image_hash,
            "logistic_probability": f"{row.baseline_xg:.6f}",
        })
    selection = reference[columns].copy()
    for method, frame in tables.items():
        selection[method] = frame.loc[selection.index, "vlm_xg"]
    rows = selection.reset_index().to_dict("records")
    direct.freeze(OUT / "selection.json", rows)

    eligible = set.union(*(set(frame.index) for frame in tables.values()))
    direct.freeze(OUT / "match_plan.json", {
        "competition": "2022 FIFA World Cup, StatsBomb",
        "rule": "Intersection of valid predictions in the four original model archives",
        "archives": {str(path.relative_to(ROOT)): direct.sha(path) for path in ARCHIVES.values()},
        "archive_sizes": {method: len(frame) for method, frame in tables.items()},
        "excluded_shots": sorted(eligible - common),
        "match_ids": sorted(reference.match_id.unique().tolist()),
    })
    direct.freeze(OUT / "protocol.json", {
        "design": "Retrospective comparison on the exact published 298-shot common set.",
        "baseline": "Archived out-of-fold predictions from the original five-fold, match-grouped logistic fit. No refit.",
        "primary": "Paired hybrid minus logistic Brier difference, 95% whole-match interval.",
        "secondary": "Reproduce the original scoring convention, including 95% shot bootstrap against logistic, for comparison with the paper.",
        "model": la_liga_protocol["model"],
        "original_prompt": prompt, "hybrid_addition": engine.PRIOR,
        "schema": la_liga_protocol["schema"], "image_detail": "high",
        "max_output_tokens": engine.MAX_OUTPUT, "cap_usd": CAP,
        "workers": engine.WORKERS, "retries": 0, "rates_per_million": list(engine.RATES),
        "baseline_sha256": direct.sha(ARCHIVES["gemini"]),
        "selection_sha256": direct.sha(OUT / "selection.json"),
        "match_plan_sha256": direct.sha(OUT / "match_plan.json"),
        "code": {str(p.relative_to(ROOT)): direct.sha(p) for p in (
            Path(__file__), ENGINE, Path(direct.__file__), ROOT / "xg_vlm_pilot.py",
            IMAGE_MANIFEST, LA_LIGA / "protocol.json", *ARCHIVES.values()
        )},
        "schedule": schedule,
    })

    archived_scores = {method: original.scores(selection.goal, selection[method])
                       for method in LABELS if method != "hybrid"}
    direct.save(OUT / "original_scores.json", archived_scores)
    attempts = direct.read(LA_LIGA / "attempts.json")
    usage = [e["usage"] for e in attempts if e["job"]["arm"] == "hybrid"]
    forecast = len(rows) * mean(engine.cost(u) for u in usage)
    print(f"Prepared {len(rows)} archived diagrams from {len(set(reference.match_id))} matches. "
          f"Estimated API cost ${forecast:.2f}, based on the prior hybrid run. No requests sent.")
    for method, scores in archived_scores.items():
        print(f"{LABELS[method]:<30} {scores['auc']:.3f} AUC  {scores['brier']:.5f} Brier")


def evaluate():
    study = engine.protocol()
    schedule = study["schedule"]
    ledger = direct.read(OUT / "attempts.json", [])
    rows = direct.read(OUT / "selection.json")
    if len(ledger) != len(rows) or len(schedule) != len(rows):
        raise ValueError("The full published cohort is required")
    ids = set()
    model_ids = set()
    for job, row, entry in zip(schedule, rows, ledger):
        if (job["shot_id"] != row["shot_id"] or entry["job"] != job
                or job["logistic_probability"] != f"{row['baseline_xg']:.6f}"
                or entry["state"] != "completed"
                or entry["request_sha256"] != direct.digest(engine.payload(job, study))):
            raise ValueError("Mismatched or incomplete hybrid request")
        raw = direct.read(OUT / entry["response"])
        if raw["usage"] != entry["usage"] or raw["id"] in ids:
            raise ValueError("Usage mismatch or duplicate response")
        ids.add(raw["id"])
        model_ids.add(raw["model"])
        answer = engine.validate(raw, entry["input_bound"])
        row["hybrid"], row["hybrid_reason"] = answer["xg"], answer["reason"]
        row["supplied_xg"] = float(job["logistic_probability"])
    if len(model_ids) != 1:
        raise ValueError("The returned model changed during the run")

    frame = pd.DataFrame(rows)
    scores = {method: original.scores(frame.goal, frame[method]) for method in LABELS}
    primary = direct.comparison(rows, "hybrid", "baseline_xg", .95)
    shot_bootstrap = {}
    for method in LABELS:
        if method != "baseline_xg":
            delta, ci = original.bootstrap_brier_diff(frame.goal, frame[method], frame.baseline_xg)
            shot_bootstrap[method] = {"difference": delta, "interval": ci.tolist()}
    summary = {
        "shots": len(rows), "goals": int(frame.goal.sum()), "matches": frame.match_id.nunique(),
        "scores": scores, "hybrid_raw_scores": direct.metrics(frame.goal, frame.hybrid),
        "primary": primary, "paper_style_comparisons": shot_bootstrap,
        "model": model_ids.pop(), "calls": len(ledger),
        "estimated_usd": sum(engine.cost(e["usage"]) for e in ledger),
        "conservative_usd": sum(engine.cost(e["usage"], conservative=True) for e in ledger),
    }
    if summary["conservative_usd"] > CAP:
        raise ValueError("Run exceeded the frozen spending limit")
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    direct.save(OUT / "summary.json", summary)
    report(summary)


def report(s):
    lines = [
        "# Astra hybrid on the original World Cup benchmark", "",
        f"The evaluation uses the original common set of {s['shots']} shots, "
        f"with {s['goals']} goals from {s['matches']} matches. Astra saw the archived "
        "diagram and the original match-grouped out-of-fold logistic estimate. "
        "The fixed hybrid instructions carry over from La Liga.", "",
        "| Method | AUC ↑ | Brier ↓ | Mean xG | Brier vs logistic [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in s["scores"].items():
        comparison = s["paper_style_comparisons"].get(method)
        interval = "—"
        if comparison:
            lo, hi = comparison["interval"]
            interval = f"{comparison['difference']:+.5f} [{lo:+.5f}, {hi:+.5f}]"
        lines.append(f"| {LABELS[method]} | {metrics['auc']:.3f} | {metrics['brier']:.5f} | "
                     f"{metrics['mean_pred']:.1%} | {interval} |")

    primary = s["primary"]
    lo, hi = primary["interval"]
    lines += [
        "", f"The primary whole-match bootstrap gives a Brier difference of "
        f"{primary['difference']:+.5f} [{lo:+.5f}, {hi:+.5f}] for hybrid minus logistic. "
        "The intervals in the table resample shots to match the original analysis.", "",
        "The original table clips probabilities to [0.001, 0.999] when computing "
        "point scores. The raw hybrid answers are preserved in the prediction file "
        "and the primary comparison uses raw Brier errors. The logistic estimate "
        "indirectly supplies the header flag. The comparison reuses the published "
        "test cohort and archived model predictions, with a fresh hybrid response "
        "for each diagram.", "",
        f"All {s['calls']} hybrid calls completed, with estimated spend "
        f"${s['estimated_usd']:.2f} and conservative accounting of ${s['conservative_usd']:.2f}.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate", "evaluate"))
    action = parser.parse_args().action
    {"prepare": prepare, "generate": engine.generate, "evaluate": evaluate}[action]()
