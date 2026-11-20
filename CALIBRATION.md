# Testing probability calibration

The calibration evaluator uses saved shot predictions. It does not make API calls
or change the original predictions.

From the repository folder, with the virtual environment active:

```bash
python calibrate_xg.py --input results/gpt_6_luna/shots.csv --output results/calibration_experiment/image_only
python calibrate_xg.py --input results/gpt_6_luna_context/shots.csv --output results/calibration_experiment/body_part
```

The output folder contains a short `report.md`, `calibration_comparison.png`, the
scores in `metrics.csv`, held-out predictions in `predictions.csv`, paired Brier
comparisons in `comparisons.csv`, and the split audit in `fold_audit.json`.

Two fixed corrections are evaluated:

- **Mean correction:** adjusts one intercept on log odds to correct an overall
  scoring-rate bias.
- **Sigmoid correction:** adjusts the intercept and a nonnegative slope on log
  odds. A fixed penalty toward slope 1 limits flexibility on small datasets.

Five folds are used by default, with whole matches kept together. For every tested
shot, the correction is learned only from the other folds. Parameters are fixed in
advance; the script does not search for whichever settings give the best result.
The original logistic and StatsBomb predictions are scored on the same rows.

The paired confidence intervals resample whole matches. They condition on the
fitted cross-validation predictions and do not include uncertainty from refitting.
Pooled cross-validation AUC can change because the folds use different corrections;
each fitted correction remains monotonic within its fold.

These are exploratory calibration results. Confirmation requires independent
matches, and ideally another competition. A corrected VLM includes supervised
calibration and must be reported separately from the original zero-shot VLM.

To generate a larger image-only prediction set before evaluating calibration:

```bash
python xg_vlm_pilot.py --backend openai --model gpt-6-luna --n 300 --seed 7
```

This command uses the API key exported in your terminal and makes API requests for
shots that are not already cached. It replaces that run's derived summary and CSV
with the larger sample. The experiment performed in this session preserved the
original 50-shot files in `results/calibration_experiment/before_expansion/`.

Run the offline checks:

```bash
python -m unittest discover -s tests -v
```
