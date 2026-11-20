# Training with imperfect player positions

One fixed offline test changes the statistical model training inputs, while
preserving the earlier image-only spline model and settings. It reuses 200
completed Euro 2024 VLM readings and makes no API calls. These are retrospective
results, not an independent confirmation. The partial Euro 2020 run is separate.

`spatial_models.py` is an unchanged snapshot of the earlier spatial study.
World Cup outcomes train the probability model; they never enter geometry inputs.
Whole-match development folds exclude held-out matches from both training and
the noise-donor pool. No noise setting is selected from Euro results. Sixteen
versions of each original shot retain total training weight one.

From this folder, with the repository virtual environment active:

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python robust_xg.py \
  --data-root /mnt/c/Users/ephou/Downloads/xG/output/research-loop \
  --geometry-results ../../results/hybrid_gpt_6_luna_300 \
  --output ../../results/noise_robustness_v1
```

The data root is the earlier spatial study's saved training tournaments, Euro
selection, reading cache/images, and frozen models. No credentials are needed.
The experiment freezes its protocol and hashes before fitting or loading external
outcomes. A changed input or implementation requires a new output folder.

Outputs include the report and figure, scores, per-shot predictions, whole-match
bootstrap intervals, fold audit, training-error donors, frozen models, and input
integrity checks. The current hybrid must reproduce saved predictions to 1e-7.
Defender jitter/dropout are fixed assumptions, not measured training errors.
