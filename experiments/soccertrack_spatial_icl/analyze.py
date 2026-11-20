"""Score only after the resource-defined run is terminal. No API calls."""
import csv
import json

import numpy as np
from PIL import Image, ImageDraw

import experiment as e

c, ROOT, OUT = e.c, e.ROOT, e.OUT
PAIRS = [("outcome32", "zero"), ("outcome32", "shuffled_outcome32"),
         ("teacher32", "zero"), ("teacher32", "shuffled_teacher32")]


def interval(rows, p, q, seed):
    y = np.array([r["goal"] for r in rows])
    delta = (np.array(p) - y)**2 - (np.array(q) - y)**2
    mids = sorted({r["match_id"] for r in rows})
    sums = np.array([sum(delta[i] for i, r in enumerate(rows) if r["match_id"] == mid) for mid in mids])
    counts = np.array([sum(r["match_id"] == mid for r in rows) for mid in mids])
    draws = np.random.default_rng(seed).integers(len(mids), size=(20000, len(mids)))
    diffs = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"difference": float(delta.mean()), "confidence": .9875,
            "interval": np.quantile(diffs, [.00625, .99375]).tolist(), "matches": len(mids)}


def load_complete():
    terminal = c.read(OUT / "complete.json")
    assert not (OUT / "generation.lock").exists(), "Wait for paid runner completion"
    pool, ledger, blocks = [c.read(OUT / n) for n in ("pool.json", "attempts.json", "blocks.json")]
    n = terminal["selected_n"]
    assert n == sum(b["shots"] for b in blocks) and all(b["state"] == "completed" for b in blocks)
    assert len(ledger) == n * len(e.ARMS) and all(r["state"] == "completed" for r in ledger)
    rows = pool[:n]
    assert {(r["shot_id"], a) for r in rows for a in e.ARMS} == {(r["job"]["shot_id"], r["job"]["arm"]) for r in ledger}
    return terminal, rows, ledger


def values_for(rows, ledger):
    answers = {(r["job"]["shot_id"], r["job"]["arm"]): r["answer"]["xg"] for r in ledger}
    values = {a: [answers[(r["shot_id"], a)] for r in rows] for a in e.ARMS}
    historical = {(r["job"]["shot_id"], r["job"]["arm"]): r["answer"]["xg"]
                  for r in c.read(e.SOURCE / "attempts.json")}
    for arm in ("raw", "diagram", "photo_diagram"):
        values["historical_" + arm] = [historical[(r["shot_id"], arm)] for r in rows]
    baseline = {r["shot_id"]: r for r in c.read(e.SOURCE / "baseline_predictions.json")}
    for arm in ("basic", "spatial", "constant"):
        values["logistic_" + arm if arm != "constant" else "training_constant"] = [baseline[r["shot_id"]][arm] for r in rows]
    return values


def preview(rows):
    row = min(rows, key=lambda r: r["shot_id"])
    pages = c.read(OUT / "pages.json")
    canvas = Image.new("RGB", (1200, 710), "white")
    draw = ImageDraw.Draw(canvas)
    choices = [(row["images"]["diagram"]["path"], "New query: outcome unknown"),
               (pages[row["match_id"]]["outcome32"][0]["path"], "Outcome examples: sheet 1 of 8"),
               (pages[row["match_id"]]["teacher32"][0]["path"], "Teacher examples: sheet 1 of 8")]
    for i, (path, title) in enumerate(choices):
        with Image.open(ROOT / path) as im:
            im = im.convert("RGB")
            im.thumbnail((390, 650), Image.Resampling.LANCZOS)
            canvas.paste(im, (i * 400 + (400-im.width)//2, 40))
        draw.text((i*400+10, 12), title, fill="black")
    canvas.save(OUT / "input_preview.png")


def main():
    terminal, rows, ledger = load_complete()
    values = values_for(rows, ledger)
    y = [r["goal"] for r in rows]
    metrics = {a: c.metrics(y, v) for a, v in values.items()}
    contrasts = {a + "_minus_" + b: interval(rows, values[a], values[b], e.SEED+500+i)
                 for i, (a, b) in enumerate(PAIRS)}
    example_sets = c.read(OUT / "examples.json")
    usage = {name: sum(r["usage"].get(name, 0) for r in ledger)
             for name in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    summary = {"dataset": "SoccerTrack v2", "model": "claude-sonnet-5-5", "shots": len(rows), "goals": sum(y),
               "matches": len({r["match_id"] for r in rows}), "metrics": metrics, "primary": contrasts,
               "example_count": 32, "example_goal_counts": {m: sum(r["goal"] for r in ex) for m, ex in example_sets.items()},
               "calls": len(ledger), "estimated_usd": sum(r["estimated_usd"] for r in ledger),
               "conservative_usd": sum(r["conservative_usd"] for r in ledger), "cap_usd": 10,
               "usage": usage, "terminal": terminal,
               "scope": "Exploratory annotation-assisted evaluation on previously studied matches; resource-selected prefix; fresh diagram-zero control; historical photo/diagram references descriptive."}
    assert summary["conservative_usd"] <= 10
    c.save(OUT / "summary.json", summary)
    with (OUT / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["shot_id", "match_id", "goal"] + list(values))
        writer.writeheader()
        writer.writerows({"shot_id": r["shot_id"], "match_id": r["match_id"], "goal": r["goal"],
                          **{a: v[i] for a, v in values.items()}} for i, r in enumerate(rows))
    preview(rows)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
