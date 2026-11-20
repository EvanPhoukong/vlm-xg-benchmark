"""Player-position diagrams plus visual ICL. Preparation is entirely offline."""
import argparse
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "results/soccertrack_spatial_v1"
OUT = ROOT / "results/soccertrack_spatial_icl_v1"
spec = importlib.util.spec_from_file_location("spatial_experiment", HERE.parent / "soccertrack_spatial/experiment.py")
spatial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spatial)
c = spatial.c
SEED = 2026112011
ARMS = ["zero", "outcome32", "shuffled_outcome32", "teacher32", "shuffled_teacher32"]
COUNT = 32
OUTCOME_INTRO = """Historical examples follow on numbered sheets. Each cell shows a DIFFERENT
past shot as a player-position diagram, paired with its single observed result:
GOAL or NO GOAL. These binary outcomes are noisy observations, not true xG values.
A goal does not mean true xG is 1, and a miss does not mean true xG is 0.
Use the situations and their outcomes as in-context examples when estimating the
new shot's scoring probability. All examples are before contact and come from
matches other than the new query's match. Read each sheet left to right, then
top to bottom. Only predict the NEW QUERY following all example sheets."""
TEACHER_INTRO = """Historical examples follow on numbered sheets. Each cell shows a DIFFERENT
past shot as a player-position diagram, paired with a teacher xG probability.
The teacher is a logistic model using shot distance, angle, goalkeeper position
and defender geometry, fitted on other matches excluding the new query's entire
match. Its training-example predictions are not ground-truth probabilities or
observed outcomes. Use the situations and probability labels as in-context
examples when estimating the new shot's scoring probability. All examples are
before contact and come from matches other than the new query's match. Read each
sheet left to right, then top to bottom. Only predict the NEW QUERY following
all example sheets. No teacher prediction is provided for the new query."""


def examples_for(rows, mid, i, model):
    training = sorted([r for r in rows if r["match_id"] != mid], key=lambda r: r["shot_id"])
    selected = random.Random(SEED + 100 + i).sample(training, COUNT)
    labels = [r["goal"] for r in selected]
    shuffled = labels.copy()
    random.Random(SEED + 200 + i).shuffle(shuffled)
    if shuffled == labels and len(set(labels)) > 1:
        shuffled = shuffled[1:] + shuffled[:1]
    teacher = [round(float(p), 4) for p in c.predict(model, selected)]
    shuffled_teacher = teacher.copy()
    random.Random(SEED + 300 + i).shuffle(shuffled_teacher)
    return [{"shot_id": r["shot_id"], "match_id": r["match_id"], "image": r["images"]["diagram"],
             "goal": r["goal"], "shuffled_goal": y, "teacher": p, "shuffled_teacher": q}
            for r, y, p, q in zip(selected, shuffled, teacher, shuffled_teacher)]


@lru_cache(maxsize=304)
def thumbnail(relative):
    with Image.open(ROOT / relative) as im:
        return im.convert("RGB").resize((320, 480), Image.Resampling.LANCZOS)


def make_pages(examples, mid, arm, destination=OUT):
    labels = {"outcome32": "goal", "shuffled_outcome32": "shuffled_goal",
              "teacher32": "teacher", "shuffled_teacher32": "shuffled_teacher"}
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default(size=17)
    result = []
    for start in range(0, COUNT, 4):
        canvas = Image.new("RGB", (640, 1008), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(examples[start:start + 4]):
            x, y = i % 2 * 320, i // 2 * 504
            canvas.paste(thumbnail(row["image"]["path"]), (x, y))
            v = row[labels[arm]]
            label = ("GOAL" if v else "NO GOAL") if "outcome" in arm else f"teacher xG = {v:.4f}"
            draw.text((x + 8, y + 482), f"{start+i+1}. {label}", fill="black", font=font)
        path = destination / "examples" / f"{mid}_{arm}_{start//4+1:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94, subsampling=0)
        result.append({"path": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_relative_to(ROOT) else str(path),
                       "sha256": c.sha(path), "size": [640, 1008]})
    return result


def ordered_pool(rows):
    """Natural random order within each match; rounds cover matches before repeats."""
    mids = sorted({r["match_id"] for r in rows})
    random.Random(SEED + 1).shuffle(mids)
    groups = {}
    for i, mid in enumerate(mids):
        group = sorted([r for r in rows if r["match_id"] == mid], key=lambda r: r["shot_id"])
        random.Random(SEED + 10 + i).shuffle(group)
        groups[mid] = group
    return [groups[mid][j] for j in range(max(map(len, groups.values()))) for mid in mids if j < len(groups[mid])]


def prepare():
    if (OUT / "protocol.json").exists():
        raise ValueError("Study already frozen")
    rows = c.read(SOURCE / "eligible.json")
    source_audit = c.read(SOURCE / "audit.json")
    assert source_audit["passed"] and len(rows) == 304
    source_models = c.read(SOURCE / "baseline_models.json")
    source_p = c.read(SOURCE / "protocol.json")
    for key, name in [("eligible_sha256", "eligible.json"), ("baseline_models_sha256", "baseline_models.json")]:
        assert c.sha(SOURCE / name) == source_p[key]
    for row in rows:
        for kind in ("diagram", "raw"):
            assert c.sha(ROOT / row["images"][kind]["path"]) == row["images"][kind]["sha256"]
    pages, examples, models = {}, {}, {}
    for i, mid in enumerate(sorted({r["match_id"] for r in rows})):
        training = [r for r in rows if r["match_id"] != mid]
        model = c.fit_logistic(training, spatial.FEATURES)
        np.testing.assert_allclose(c.predict(model, rows), c.predict(source_models[mid]["spatial"], rows), atol=1e-10, rtol=0)
        models[mid] = {"model": model, "training_ids": [r["shot_id"] for r in training],
                       "training_matches": sorted({r["match_id"] for r in training})}
        examples[mid] = examples_for(rows, mid, i, model)
        pages[mid] = {"zero": []}
        for arm in ARMS[1:]:
            pages[mid][arm] = make_pages(examples[mid], mid, arm)
        print(f"Prepared {mid}: 32 other-match examples, {sum(r['goal'] for r in examples[mid])} goals", flush=True)
    pool = ordered_pool(rows)
    c.freeze(OUT / "pool.json", pool)
    c.freeze(OUT / "pages.json", pages)
    c.freeze(OUT / "examples.json", examples)
    c.freeze(OUT / "teacher_models.json", models)
    dependencies = [HERE / "experiment.py", HERE / "run.mjs", HERE.parent / "soccertrack_spatial/experiment.py",
                    HERE.parent / "soccertrack_spatial/common.py", HERE.parent / "soccertrack_spatial/run.mjs",
                    HERE.parent / "soccertrack_comparison/common.py"]
    sources = [SOURCE / n for n in ("protocol.json", "audit.json", "eligible.json", "source_audit.json", "baseline_models.json", "baseline_predictions.json", "attempts.json")]
    p = {"dataset": "SoccerTrack v2", "model": "claude-sonnet-5-5", "seed": SEED, "arms": ARMS,
         "example_count": COUNT, "prompt": source_p["prompt"], "outcome_intro": OUTCOME_INTRO, "teacher_intro": TEACHER_INTRO,
         "cap_usd": 10.0, "rates_per_million": {"input": 2., "cache_write": 2.5, "cache_read": .1, "output": 10.},
         "conservative_rates_per_million": {"input": 2.5, "cache_write": 2.5, "cache_read": .125, "output": 10.},
         "pricing_source": "https://platform.claude.com/docs/en/models/sonnet-5-5/overview",
         "max_output_tokens": 1024, "effort": "low", "workers": 6, "retries": 0, "global_inference": True,
         "cache": "Explicit ephemeral 5-minute breakpoint on the final example-sheet image only. Cache accounting includes writes and reads; no cache saving assumed in pre-dispatch reservations.",
         "pool": "All304 previously source-audited eligible shots. Natural random within-match order, round-robin across a seeded match order. No outcome balancing or accuracy selection.",
         "stopping_rule": "Before accuracy inspection, evaluate a prefix of the fixed pool, up to304 shots. At each block choose the largest next1..10 complete shots whose full five-arm maximum uncached/write-cost reservations fit the remaining$10. Reserve all block requests durably before any dispatch. Finish every reserved complete-shot block. Stop before the next complete shot would exceed cap. Cache savings may allow a larger prefix but never change arms/examples. Final cohort is usage-selected, not fixed-N external confirmation.",
         "request_bounds": "Count largest-byte query image in each match/arm before generation; use maximum count plus1024 as every request's input bound. Reservation=input_bound*2.5+1024*10 per million. Validate actual usage against bound; stop on unresolved requests, never retry automatically.",
         "scheduling": "Selected blocks use fixed pool order. Within each block sort by match ID to encourage prefix caching; rotate five-arm order by pool index. No dispatch choice uses answer accuracy.",
         "examples": "32 natural random source diagrams from other9 matches for every excluded query match. Binary shuffled controls preserve goal counts; teacher shuffled controls preserve complete rounded probability distribution. Same images and order within each correct/shuffled pair.",
         "teacher": "Spatial logistic distance,angle,goalkeeper-line-distance,goalkeeper-lateral-error,defenders-in-cone,nearest-defender. Fitted/scaled on other9 complete matches, excluding query match. Training-example probabilities rounded4 decimals; no query teacher probability supplied.",
         "images": "Original640x960 pre-shot query diagram unchanged. Example diagrams resized320x480; four cells on640x1008 JPEG94 sheets with17px labels, eight sheets per condition. No photo-only/marked-photo input enters new requests.",
         "statistics": "Four primary Brier contrasts:outcome32-zero,outcome32-shuffled_outcome32,teacher32-zero,teacher32-shuffled_teacher32.98.75% whole-match percentile intervals,20000 draws,Bonferroni family5%. Other references and AUC descriptive; overlapping fold fits/example sets held fixed.",
         "scope": "Annotation-assisted exploratory evaluation on10 previously studied matches. No independent external cohort. No query-match examples or fitted teacher training, no auxiliary ball track, no post-event input, outcome/header/coordinate values/identities/reference xG in queries. Raw VLM probabilities; no calibration,averaging or weight updates.",
         "source_caveats": "Original source audit and two geometry exclusions retained. Supplied roles/coordinates are not independently validated automatic detections. Full large-file hashes were not verified. Professional-finisher assumption retained on university footage.",
         "historical_controls": "Previous304-shot raw and diagram Sonnet predictions are matched descriptive references only; fresh diagram-zero predictions are the primary ICL control. Do not compare absolute scores with previous80-shot Sol ICL study.",
         "frozen": {str(f.relative_to(ROOT)).replace("\\", "/"): c.sha(f) for f in dependencies + sources + [OUT / n for n in ("pool.json", "pages.json", "examples.json", "teacher_models.json")]}}
    c.freeze(OUT / "protocol.json", p)
    print(json.dumps({"pool": len(pool), "matches": len(examples), "conditions": len(ARMS), "maximum_calls": len(pool)*len(ARMS)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare"])
    globals()[parser.parse_args().action]()
