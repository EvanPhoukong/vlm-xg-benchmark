"""Evaluate probability calibration on saved VLM predictions, without API calls.

Each correction is fitted using other matches, never the match being scored.
The two fixed experiments are an intercept-only correction and a monotonic
sigmoid correction on log odds. No settings are selected using test performance.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


PROBABILITY_COLUMNS = ["vlm_xg", "baseline_xg", "sb_xg"]
METHODS = {
    "vlm_xg": "VLM, original",
    "offset_xg": "VLM, mean correction",
    "sigmoid_xg": "VLM, sigmoid correction",
    "baseline_xg": "Logistic baseline",
    "sb_xg": "StatsBomb",
    "training_rate_xg": "Training goal-rate reference",
}


def validate_predictions(frame):
    required = {"shot_id", "match_id", "goal", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame[["shot_id", "match_id"]].isna().any().any():
        raise ValueError("Shot and match IDs must be present.")
    if frame.shot_id.duplicated().any():
        raise ValueError("Each shot_id must occur exactly once.")
    if set(frame.goal.unique()) != {0, 1}:
        raise ValueError("The evaluation needs both goals (1) and non-goals (0).")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must be finite probabilities between 0 and 1.")


def fit_correction(probabilities, outcomes, method):
    z = logit(np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6))
    y = np.asarray(outcomes, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Every calibration training fold must contain goals and non-goals.")
    if method == "offset":
        intercept = brentq(lambda a: expit(z + a).mean() - y.mean(), -50, 50)
        return float(intercept), 1.0
    if method != "sigmoid":
        raise ValueError(f"Unknown calibration method: {method}")

    def objective(parameters):
        a, b = parameters
        eta = a + b * z
        residual = expit(eta) - y
        # Fixed weak shrinkage toward slope=1 limits flexibility on small data.
        loss = np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * (b - 1) ** 2
        gradient = np.array([residual.sum(), np.dot(residual, z) + b - 1])
        return float(loss), gradient

    fitted = minimize(objective, [0.0, 1.0], jac=True, method="L-BFGS-B",
                      bounds=[(None, None), (0.0, None)], options={"maxiter": 1000})
    if not fitted.success:
        raise RuntimeError(f"Calibration fit failed: {fitted.message}")
    return tuple(float(value) for value in fitted.x)


def apply_correction(probabilities, parameters):
    z = logit(np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6))
    intercept, slope = parameters
    return expit(intercept + slope * z)


def make_folds(frame, n_splits=5, seed=7):
    if not 2 <= n_splits <= frame.match_id.nunique():
        raise ValueError("Choose at least two folds and no more folds than distinct matches.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(frame)), frame.goal, groups=frame.match_id))


def cross_calibrate(frame, folds):
    validate_predictions(frame)
    evaluated = frame.copy().reset_index(drop=True)
    for column in ["offset_xg", "sigmoid_xg", "training_rate_xg"]:
        evaluated[column] = np.nan
    evaluated["calibration_fold"] = -1
    assigned = np.zeros(len(frame), dtype=int)
    audit = []
    for fold_id, (training, testing) in enumerate(folds):
        train = evaluated.iloc[training]
        test = evaluated.iloc[testing]
        train_matches = set(train.match_id)
        test_matches = set(test.match_id)
        if train_matches.intersection(test_matches):
            raise ValueError("A match appears in both calibration training and testing.")
        assigned[testing] += 1
        evaluated.loc[testing, "calibration_fold"] = fold_id
        evaluated.loc[testing, "training_rate_xg"] = train.goal.mean()
        for method in ["offset", "sigmoid"]:
            parameters = fit_correction(train.vlm_xg, train.goal, method)
            evaluated.loc[testing, f"{method}_xg"] = apply_correction(test.vlm_xg, parameters)
            audit.append({
                "fold": fold_id, "method": method,
                "intercept": parameters[0], "slope": parameters[1],
                "training_shots": len(train), "training_goals": int(train.goal.sum()),
                "testing_shots": len(test), "testing_goals": int(test.goal.sum()),
                "training_match_ids": sorted(str(value) for value in train_matches),
                "testing_match_ids": sorted(str(value) for value in test_matches),
            })
    if not np.all(assigned == 1):
        raise ValueError("Every shot must be tested exactly once.")
    return evaluated, audit


def calculate_metrics(frame):
    y = frame.goal.to_numpy()
    rows = []
    for column, label in METHODS.items():
        p = np.clip(frame[column].to_numpy(), 1e-6, 1 - 1e-6)
        rows.append({"prediction_column": column, "model": label,
                     "brier": float(brier_score_loss(y, p)),
                     "auc": float(roc_auc_score(y, p)),
                     "logloss": float(log_loss(y, p, labels=[0, 1])),
                     "mean_xg": float(p.mean())})
    return pd.DataFrame(rows)


def match_bootstrap(frame, first, second, n_boot=4000, seed=7):
    errors = pd.DataFrame({
        "match_id": frame.match_id,
        "difference": (frame[first] - frame.goal) ** 2 - (frame[second] - frame.goal) ** 2})
    by_match = errors.groupby("match_id", sort=True).difference.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(by_match), size=(n_boot, len(by_match)))
    differences = by_match["sum"].to_numpy()[sampled].sum(axis=1)
    differences /= by_match["count"].to_numpy()[sampled].sum(axis=1)
    return {"first": first, "second": second,
            "brier_difference": float(errors.difference.mean()),
            "ci_low": float(np.quantile(differences, 0.025)),
            "ci_high": float(np.quantile(differences, 0.975))}


def save_plot(frame, metrics, path):
    plotted = metrics[metrics.prediction_column != "training_rate_xg"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#727c8e", "#e69f00", "#b65b22", "#2674b8", "#33835d"]
    axes[0].barh(plotted.model, plotted.brier, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Brier score — lower is better")
    for i, value in enumerate(plotted.brier):
        axes[0].text(value + 0.001, i, f"{value:.4f}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(plotted.brier) * 1.25)
    axes[1].plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
    for color, (column, label) in zip(colors, list(METHODS.items())[:5]):
        groups = pd.qcut(frame[column], q=5, duplicates="drop")
        points = frame.groupby(groups, observed=True).agg(pred=(column, "mean"), obs=("goal", "mean"))
        axes[1].plot(points.pred, points.obs, "o-", label=label, color=color, markersize=4)
    axes[1].set(xlabel="Average predicted scoring probability", ylabel="Observed scoring rate",
                xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Exploratory calibration test: {len(frame)} shots, {int(frame.goal.sum())} goals")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(frame, metrics, comparisons, output, input_path, n_splits):
    scores = metrics.set_index("prediction_column")
    raw = scores.loc["vlm_xg", "brier"]
    sigmoid = scores.loc["sigmoid_xg", "brier"]
    offset = scores.loc["offset_xg", "brier"]
    baseline = scores.loc["baseline_xg", "brier"]
    main_comparison = next(item for item in comparisons
                           if item["first"] == "sigmoid_xg" and item["second"] == "vlm_xg")
    lines = [
        "# Calibration test report", "",
        f"Input: `{input_path}`", "",
        f"**{len(frame)} shots; {int(frame.goal.sum())} goals; {frame.match_id.nunique()} matches.** "
        f"Observed scoring rate: {frame.goal.mean():.1%}.", "",
        "## What happened", "",
        f"- Original VLM Brier: **{raw:.4f}**.",
        f"- Mean correction Brier: **{offset:.4f}** ({'better' if offset < raw else 'worse'} than original).",
        f"- Sigmoid correction Brier: **{sigmoid:.4f}** ({'better' if sigmoid < raw else 'worse'} than original).",
        f"- Logistic baseline Brier: **{baseline:.4f}**. The sigmoid-corrected VLM was "
        f"**{'better' if sigmoid < baseline else 'worse'} by point estimate**.",
        f"- Sigmoid minus original Brier difference: **{main_comparison['brier_difference']:+.4f}**; "
        f"approximate 95% match-bootstrap interval **[{main_comparison['ci_low']:+.4f}, "
        f"{main_comparison['ci_high']:+.4f}]**. Negative favors calibration.", "",
        "## Scores", "",
        "| Model | Brier (lower better) | AUC | Mean xG | Log loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples():
        lines.append(f"| {row.model} | {row.brier:.4f} | {row.auc:.3f} | {row.mean_xg:.1%} | {row.logloss:.4f} |")
    lines += ["", "## How it was tested", "",
              f"- {n_splits}-fold cross-validation, grouping whole matches and using fixed seed 7 by default.",
              "- Each correction was fitted only on the other folds, with no goals from the tested matches.",
              "- Mean correction fits one intercept on log odds. Sigmoid correction fits an intercept and a nonnegative slope, with fixed weak shrinkage toward slope 1.",
              "- Both methods were fixed before evaluating results; there was no search for the best settings.",
              "- Baselines are the existing saved predictions, rather than newly fitted models.",
              "- The training-rate reference predicts the goal rate of the other folds; it does not use the tested matches' outcomes.",
              "- Bootstrap resamples whole matches. Its intervals condition on the fitted cross-validation predictions and do not include refitting uncertainty.", "",
              "## Limits", "",
              "- This is an exploratory cross-validation result, not validation on a new competition.",
              "- With few goals, fitted corrections and score differences can be unstable. Independent matches are needed to confirm any improvement.",
              "- Calibration is monotonic within each fold. Pooled AUC may change because different folds use different corrections; this does not demonstrate improved within-model ranking.",
              "- Calibrated predictions include a supervised training step and should be reported separately from zero-shot results.", "",
              "![Calibration comparison](calibration_comparison.png)", ""]
    report = "\n".join(lines)
    (output / "report.md").write_text(report, encoding="utf-8")
    return report


def evaluate(input_path, output, n_splits=5, seed=7, n_boot=4000):
    frame = pd.read_csv(input_path)
    validate_predictions(frame)
    frame = frame.sort_values(["match_id", "shot_id"]).reset_index(drop=True)
    folds = make_folds(frame, n_splits, seed)
    evaluated, audit = cross_calibrate(frame, folds)
    metrics = calculate_metrics(evaluated)
    comparisons = [match_bootstrap(evaluated, first, second, n_boot, seed)
                   for first, second in [("sigmoid_xg", "vlm_xg"), ("offset_xg", "vlm_xg"),
                                         ("sigmoid_xg", "baseline_xg"), ("sigmoid_xg", "sb_xg")]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    (output / "fold_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metadata = {"input": str(Path(input_path).resolve()), "shots": len(evaluated),
                "goals": int(evaluated.goal.sum()), "matches": int(evaluated.match_id.nunique()),
                "folds": n_splits, "seed": seed, "bootstrap_resamples": n_boot,
                "sigmoid_slope_penalty": 1.0, "match_overlap": False}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_plot(evaluated, metrics, output / "calibration_comparison.png")
    return write_report(evaluated, metrics, comparisons, output, input_path, n_splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="A saved benchmark shots.csv")
    parser.add_argument("--output", type=Path, help="Derived output folder; raw predictions are unchanged")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap", type=int, default=4000)
    args = parser.parse_args()
    output = args.output or args.input.parent / "calibration_test"
    print(evaluate(args.input, output, args.folds, args.seed, args.bootstrap))


if __name__ == "__main__":
    main()
