"""Run the frozen, paired Astra + logistic xG study on unused La Liga matches."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import importlib.util
import math
from pathlib import Path
import random
import statistics
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "transfer_prepare", ROOT / "experiments/dataset_transfer/prepare.py"
)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
direct = transfer.base

OUT = ROOT / "results/astra_logistic_prior_v1"
REGISTRY = ROOT / "results/core_fixes_v1/data/match_split.json"
BASELINE = ROOT / "results/direct_scope_suite/data/match_split.json"
MANIFEST = ROOT / "results/direct_model_search_v1/data/matches_11_27.json"
MODEL = {"model": "gpt-6-astra", "effort": "low"}
RATES = (10.0, 1.0, 50.0)
CAP, MAX_OUTPUT, WORKERS = 2.90, 1024, 8
SHOTS, MATCH_POOL, SEED = 100, 40, 2026110501
PRIOR = """
A logistic regression trained on past shots estimates this shot's xG at {probability}.
It uses shot distance, angle to goal, a header indicator and the number of opponents
in the shooting cone. Treat that estimate as your starting point. Inspect the diagram
for relevant visual evidence the regression may miss, such as the goalkeeper's
position or where defenders are positioned within the shooting lane. Return your
own final goal probability. Adjust the starting estimate when the image justifies
a change, and keep it if the image gives no clear reason to adjust. In your reason,
name the visual evidence for your adjustment or why you kept the estimate.
""".strip()
METHODS = {
    "hybrid": "Astra + logistic estimate",
    "astra": "Astra, diagram only",
    "original_logistic": "Logistic regression (supplied estimate)",
    "original_no_header": "Logistic regression (without header)",
    "sb_xg": "StatsBomb xG",
}


def prepare():
    registry = direct.read(REGISTRY)
    used = set(registry["previous_match_ids"] + registry["selected_match_ids"])
    eligible = sorted({m["match_id"] for m in direct.read(MANIFEST)} - used)
    if len(used) != 522 or len(eligible) < MATCH_POOL:
        raise ValueError("Unexpected prior match registry or match availability")
    matches = sorted(random.Random(SEED).sample(eligible, MATCH_POOL))
    direct.freeze(OUT / "match_plan.json", {
        "competition": "La Liga 2015/16",
        "match_ids": matches,
        "prior_match_count": len(used),
        "manifest_sha256": direct.sha(MANIFEST),
        "prior_registry_sha256": direct.sha(REGISTRY),
        "seed": SEED,
        "rule": "Sample 40 unused match groups, then 100 eligible non-penalty shots uniformly from their pooled freeze frames.",
    })

    def load_match(mid):
        path = OUT / "data" / f"events_{mid}.json"
        if not path.exists():
            with urlopen(direct.BASE + f"events/{mid}.json", timeout=90) as response:
                import json
                direct.save(path, json.load(response))
        rows, _ = transfer.sb_rows(path, mid)
        return rows, direct.sha(path)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        sources = list(pool.map(load_match, matches))
    eligible_shots = sorted(
        (r for rows, _ in sources for r in rows), key=lambda r: r["shot_id"]
    )
    rows = random.Random(SEED + 1).sample(eligible_shots, SHOTS)
    if len({r["shot_id"] for r in rows}) != SHOTS:
        raise ValueError("Duplicate shots in evaluation cohort")
    transfer.predict_frozen(rows)

    jobs = []
    rng = random.Random(SEED + 2)
    for row in rows:
        sid = row["shot_id"]
        image = OUT / "images" / f"{sid}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        if not image.exists():
            transfer.original.render(pd.Series(row), image)
        arms = ["hybrid", "astra"]
        rng.shuffle(arms)
        for arm in arms:
            job = {"shot_id": sid, "arm": arm, "image": str(image.relative_to(ROOT)),
                   "image_sha256": direct.sha(image)}
            if arm == "hybrid":
                job["logistic_probability"] = f"{row['original_logistic']:.6f}"
            jobs.append(job)

    direct.freeze(OUT / "selection.json", rows)
    direct.freeze(OUT / "protocol.json", {
        "design": "Paired, prospective comparison on match groups outside the recorded development and evaluation sets. All 100 shots and both prompts frozen before inference.",
        "baseline": "Frozen four-feature logistic fit from 3,068 World Cup 2018/2022 shots.",
        "primary": "Two paired Brier differences, hybrid minus Astra and hybrid minus the supplied logistic regression. Use 97.5% whole-match bootstrap intervals for each.",
        "secondary": "Log loss, AUC, mean xG, and shifts from the logistic starting point. Other logistic and StatsBomb are descriptive references.",
        "model": MODEL, "original_prompt": direct.original_prompt(), "hybrid_addition": PRIOR,
        "schema": direct.SCHEMA, "image_detail": "high", "max_output_tokens": MAX_OUTPUT,
        "cap_usd": CAP, "workers": WORKERS, "retries": 0, "rates_per_million": list(RATES),
        "sources": {str(mid): sha for mid, (_, sha) in zip(matches, sources)},
        "eligible_shots": len(eligible_shots),
        "baseline_sha256": direct.sha(BASELINE),
        "selection_sha256": direct.sha(OUT / "selection.json"),
        "match_plan_sha256": direct.sha(OUT / "match_plan.json"),
        "code": {str(p.relative_to(ROOT)): direct.sha(p) for p in
                 (Path(__file__), Path(direct.__file__), Path(transfer.__file__),
                  ROOT / "xg_vlm_pilot.py")},
        "schedule": jobs,
    })
    print(f"Frozen {len(rows)} shots across {len({r['match_id'] for r in rows})} unused matches.", flush=True)


def protocol():
    saved = direct.read(OUT / "protocol.json")
    if not saved:
        raise ValueError("Prepare the fixed cohort before running inference")
    hashes = {**saved["code"], str(BASELINE.relative_to(ROOT)): saved["baseline_sha256"],
              str((OUT / "selection.json").relative_to(ROOT)): saved["selection_sha256"],
              str((OUT / "match_plan.json").relative_to(ROOT)): saved["match_plan_sha256"]}
    if any(direct.sha(ROOT / file) != sha for file, sha in hashes.items()):
        raise ValueError("Frozen code or study input changed")
    return saved


def payload(job, study):
    image = ROOT / job["image"]
    if direct.sha(image) != job["image_sha256"]:
        raise ValueError("Frozen diagram changed")
    prompt = study["original_prompt"]
    if job["arm"] == "hybrid":
        prompt += "\n\n" + study["hybrid_addition"].format(probability=job["logistic_probability"])
    data = direct.body(study["model"], prompt, image.read_bytes())
    data["max_output_tokens"] = study["max_output_tokens"]
    return data


def cost(usage, conservative=False):
    input_rate, cached_rate, output_rate = RATES
    inputs, outputs = usage["input_tokens"], usage["output_tokens"]
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    if not 0 <= cached <= inputs:
        raise ValueError("Invalid cached token count")
    input_cost = inputs * input_rate * 1.25 if conservative else (
        (inputs - cached) * input_rate + cached * cached_rate
    )
    return (input_cost + outputs * output_rate) / 1_000_000


def validate(raw, bound):
    answer = direct.validate(raw, MODEL)
    usage = raw["usage"]
    if usage["input_tokens"] > bound or usage["output_tokens"] > MAX_OUTPUT:
        raise ValueError("Token usage exceeded the reservation")
    cost(usage)
    return answer


def generate():
    study = protocol()
    schedule = study["schedule"]
    ledger_path = OUT / "attempts.json"
    lock = OUT / "generation.lock"
    lock.touch(exist_ok=False)
    try:
        ledger = direct.read(ledger_path, [])
        for job, entry in zip(schedule, ledger):
            if entry["state"] == "started" and (OUT / entry["response"]).exists():
                raw = direct.read(OUT / entry["response"])
                validate(raw, entry["input_bound"])
                entry.update(state="completed", usage=raw["usage"])
            if (entry["state"] != "completed" or entry["job"] != job
                    or entry["request_sha256"] != direct.digest(payload(job, study))):
                raise ValueError("Unreconciled prior request or schedule mismatch")
        if len(ledger) > len(schedule):
            raise ValueError("Ledger exceeds the fixed schedule")
        direct.save(ledger_path, ledger)
        if len(ledger) == len(schedule):
            print("All scheduled predictions are already complete.", flush=True)
            return
        key = direct.credential()

        def count(data):
            response = direct.api("responses/input_tokens", key, {
                field: data[field] for field in ("model", "input", "text")
            })
            tokens = response.get("input_tokens")
            if type(tokens) is not int or not 0 < tokens < 60000:
                raise ValueError("Input token count unavailable")
            return tokens + 1024

        def request(entry, data):
            try:
                raw = direct.api("responses", key, data)
                direct.save(OUT / entry["response"], raw)
                validate(raw, entry["input_bound"])
                return {"state": "completed", "usage": raw["usage"]}
            except (RuntimeError, ValueError) as error:
                return {"state": "failed_or_uncertain", "error": str(error)}

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while len(ledger) < len(schedule):
                offset = len(ledger)
                jobs = schedule[offset:offset + WORKERS]
                bodies = [payload(job, study) for job in jobs]
                bounds = list(pool.map(count, bodies))
                exposure = sum(cost(e["usage"], conservative=True) for e in ledger)
                batch = []
                for index, (job, body, bound) in enumerate(zip(jobs, bodies, bounds), offset + 1):
                    reserve = cost({"input_tokens": bound, "output_tokens": MAX_OUTPUT}, conservative=True)
                    if exposure + reserve > CAP:
                        break
                    exposure += reserve
                    batch.append({
                        "job": job, "input_bound": bound, "state": "started",
                        "response": f"responses/{index:04d}.json",
                        "request_sha256": direct.digest(body),
                    })
                if not batch:
                    raise ValueError("Spending cap reached before the complete fixed cohort")
                ledger.extend(batch)
                direct.save(ledger_path, ledger)
                futures = [pool.submit(request, entry, body) for entry, body in zip(batch, bodies)]
                for entry, future in zip(batch, futures):
                    entry.update(future.result())
                    direct.save(ledger_path, ledger)
                failed = [entry["error"] for entry in batch if entry["state"] != "completed"]
                if failed:
                    raise RuntimeError(f"Stopped after failed or uncertain requests: {failed}")
                estimated = sum(cost(e["usage"]) for e in ledger)
                print(f"{len(ledger)}/{len(schedule)} Astra predictions, estimated ${estimated:.3f}", flush=True)
    finally:
        lock.unlink()


def evaluate():
    study = protocol()
    sources = direct.read(OUT / "selection.json")
    schedule = study["schedule"]
    ledger = direct.read(OUT / "attempts.json", [])
    if len(ledger) != len(schedule):
        raise ValueError("The full fixed cohort is required before evaluation")

    baseline = {row["shot_id"]: row["original_logistic"] for row in sources}
    predictions, ids, model_ids = {}, set(), set()
    for job, entry in zip(schedule, ledger):
        if (entry["job"] != job or entry["state"] != "completed"
                or entry["request_sha256"] != direct.digest(payload(job, study))):
            raise ValueError("Incomplete or mismatched request")
        if job["arm"] == "hybrid" and job["logistic_probability"] != f"{baseline[job['shot_id']]:.6f}":
            raise ValueError("The supplied prior differs from the frozen logistic estimate")
        raw = direct.read(OUT / entry["response"])
        if raw["usage"] != entry["usage"] or raw["id"] in ids:
            raise ValueError("Usage mismatch or duplicate response")
        ids.add(raw["id"])
        model_ids.add(raw["model"])
        predictions[job["shot_id"], job["arm"]] = validate(raw, entry["input_bound"])

    if len(predictions) != 2 * SHOTS or len(model_ids) != 1:
        raise ValueError("The paired design requires distinct responses from one model version")
    columns = ("shot_id", "match_id", "goal", "original_logistic", "original_no_header", "sb_xg")
    rows = []
    for source in sources:
        row = {key: source[key] for key in columns}
        for arm in ("hybrid", "astra"):
            answer = predictions[row["shot_id"], arm]
            row[arm], row[arm + "_reason"] = answer["xg"], answer["reason"]
        row["supplied_xg"] = float(f"{row['original_logistic']:.6f}")
        row["hybrid_minus_supplied"] = row["hybrid"] - row["supplied_xg"]
        rows.append(row)
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    goals = [r["goal"] for r in rows]
    moves = [r["hybrid_minus_supplied"] for r in rows]
    scores = {method: direct.metrics(goals, [r[method] for r in rows]) for method in METHODS}
    comparisons = [direct.comparison(rows, "hybrid", method) for method in ("astra", "original_logistic")]
    summary = {
        "shots": len(rows), "goals": sum(goals), "matches": len({r["match_id"] for r in rows}),
        "scores": scores, "primary": comparisons,
        "adjustments": {"mean_absolute": statistics.mean(map(abs, moves)),
                        "median_absolute": statistics.median(map(abs, moves)),
                        "up_at_least_one_point": sum(d >= .01 for d in moves),
                        "down_at_least_one_point": sum(d <= -.01 for d in moves)},
        "returned_model": model_ids.pop(), "completed_calls": len(ledger),
        "estimated_usd": sum(cost(e["usage"]) for e in ledger),
        "conservative_usd": sum(cost(e["usage"], conservative=True) for e in ledger),
    }
    if not math.isfinite(summary["conservative_usd"]) or summary["conservative_usd"] > CAP:
        raise ValueError("Run exceeded the configured spending cap")
    direct.save(OUT / "summary.json", summary)
    report(summary)


def report(s):
    lines = [
        "# Astra with a logistic xG estimate", "",
        f"The fixed test covered {s['shots']} non-penalty shots with freeze frames "
        f"from {s['matches']} previously unused La Liga 2015/16 matches. "
        f"{s['goals']} shots scored ({s['goals'] / s['shots']:.1%}). "
        "The same diagrams went to Astra in two independent calls per shot. "
        "The hybrid prompt supplied the frozen logistic estimate as a starting point "
        "and asked Astra to adjust it when the image gave a reason. Astra's raw answer was scored.", "",
        "| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |",
        "|---|---:|---:|---:|---:|",
        *[f"| {METHODS[m]} | {v['brier']:.5f} | {v['auc']:.3f} | {v['log_loss']:.5f} | {v['mean_xg']:.1%} |"
          for m, v in s["scores"].items()], "",
        "## Paired comparisons", "",
        "Each 97.5% interval resamples whole matches. Negative Brier differences favor the hybrid.", "",
    ]
    for item in s["primary"]:
        lo, hi = item["interval"]
        difference = item["difference"]
        finding = (
            "The interval supports lower Brier error for the hybrid." if hi < 0 else
            "The interval supports higher Brier error for the hybrid." if lo > 0 else
            "The interval includes zero, leaving the difference uncertain."
        )
        lines.append(f"- Hybrid minus {METHODS[item['second']]} was {difference:+.5f} "
                     f"[{lo:+.5f}, {hi:+.5f}]. {finding}")
    a = s["adjustments"]
    lines += [
        "", "## How Astra used the starting estimate", "",
        f"The mean absolute adjustment was {a['mean_absolute'] * 100:.1f} percentage points "
        f"and the median was {a['median_absolute'] * 100:.1f}. Astra raised the estimate "
        f"by at least one point on {a['up_at_least_one_point']} shots and lowered it "
        f"by at least one point on {a['down_at_least_one_point']}.", "",
        "The regression was fitted on 3,068 World Cup shots. It uses recorded distance, goal "
        "angle, a header flag and the opponent count in the shooting cone. The header flag "
        "is absent from the diagram and reaches the hybrid indirectly through the estimate. "
        "This tests the complete hybrid prompt, including the instruction to check for "
        "visual evidence beyond those features. Shot outcomes and StatsBomb xG were used "
        "for scoring and reference only. The results cover one league and season, with one "
        "model response per arm per shot.", "",
        f"All {s['completed_calls']} calls completed. Estimated API usage cost "
        f"${s['estimated_usd']:.2f} using standard rates. Conservative accounting was "
        f"${s['conservative_usd']:.2f} against the ${CAP:.2f} cap. "
        "[OpenAI pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).", "",
        "Shot and freeze-frame data from [StatsBomb Open Data](https://github.com/hudl/open-data).", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate", "evaluate"))
    action = parser.parse_args().action
    {"prepare": prepare, "generate": generate, "evaluate": evaluate}[action]()
