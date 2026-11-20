"""Create the comparison figure and diagnostic tables from completed predictions."""
from collections import defaultdict
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import study as s

LABELS = {"logistic_distance_angle": "Logistic regression", "luna": "GPT-6 Luna",
          "terra": "GPT-6.1 Terra", "sol": "GPT-6.1 Sol", "opus": "Claude Opus 5.5"}


def main():
    summary = s.c.read(s.OUT / "summary.json")
    with (s.OUT / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("goal", "distance_m", "header", *LABELS):
            row[key] = float(row[key])
    counts = defaultdict(list)
    for row in rows:
        distance = row["distance_m"]
        group = "<12 m" if distance < 12 else "12–18 m" if distance < 18 else "18+ m"
        counts[group].append(row)
    diagnostic = {}
    for group, subset in counts.items():
        diagnostic[group] = {
            name: s.c.metrics([r["goal"] for r in subset], [r[name] for r in subset]) for name in LABELS
        }
    s.c.save(s.OUT / "distance_metrics.json", diagnostic)
    ledger = s.c.read(s.OUT / "ledger.json")
    costs = {arm: {"new_calls": sum(e["arm"] == arm for e in ledger),
                   "estimated_usd": sum(e["estimated_usd"] for e in ledger if e["arm"] == arm),
                   "conservative_usd": sum(e["charged_bound_usd"] for e in ledger if e["arm"] == arm)}
             for arm in s.MODELS}
    s.c.save(s.OUT / "costs.json", costs)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    colors = ["#475569", "#d97706", "#0891b2", "#2563eb", "#9333ea"]
    names = list(LABELS)
    values = [summary["metrics"][name]["brier"] for name in names]
    bars = axes[0].barh([LABELS[name] for name in names], values, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Brier score · lower is better")
    axes[0].set_title("Probability error on 306 shots")
    axes[0].set_xlim(0, max(values) * 1.22)
    for bar, value in zip(bars, values):
        axes[0].text(value + .002, bar.get_y() + bar.get_height()/2, f"{value:.4f}", va="center")
    axes[1].plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1)
    reliability = {}
    for name, color in zip(names, colors):
        groups = defaultdict(list)
        for row in rows:
            groups[min(4, int(row[name] * 5))].append(row)
        points = [{"n": len(group), "mean_xg": np.mean([r[name] for r in group]),
                   "goal_rate": np.mean([r["goal"] for r in group])}
                  for _, group in sorted(groups.items())]
        reliability[name] = points
        axes[1].plot([p["mean_xg"] for p in points], [p["goal_rate"] for p in points],
                     "o-", label=LABELS[name], color=color, markersize=5)
    axes[1].set(xlabel="Mean predicted xG", ylabel="Observed goal rate",
                xlim=(0, 1), ylim=(0, 1), title="Reliability · five fixed probability bins")
    axes[1].legend(fontsize=8, loc="lower right")
    figure.suptitle("SoccerTrack v2 · 306 shots · 44 goals · 10 matches", fontsize=14)
    figure.savefig(s.OUT / "comparison.png", dpi=160)
    plt.close(figure)
    s.c.save(s.OUT / "reliability_bins.json", reliability)
    lines = ["", "## Distance groups", "",
             "| Distance | Shots | Goal rate | Logistic Brier | Luna Brier | Terra Brier | Sol Brier | Opus Brier |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("<12 m", "12–18 m", "18+ m"):
        group_metrics = diagnostic[group]
        base = group_metrics["logistic_distance_angle"]
        errors = " | ".join(f"{group_metrics[name]['brier']:.4f}" for name in LABELS)
        lines.append(f"| {group} | {base['n']} | {base['goal_rate']:.1%} | {errors} |")
    lines += ["", "## New API spending", "",
              "| Model | New calls | Estimated spend |",
              "|---|---:|---:|"]
    for arm, values in costs.items():
        lines.append(f"| {LABELS[arm]} | {values['new_calls']} | ${values['estimated_usd']:.3f} |")
    lines += ["", "![Model comparison](comparison.png)", ""]
    with (s.OUT / "REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print("Saved comparison figure, distance groups, reliability bins, and model cost table.", flush=True)


if __name__ == "__main__":
    main()

