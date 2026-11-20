# Testing the image-to-geometry xG hybrid

The hybrid has three steps:

1. GPT-6 Luna reads shooter and opposing goalkeeper locations from the original
   half-pitch image, using the existing fixed geometry prompt.
2. Code calculates geometric features from those reported locations.
3. A logistic regression model converts the features into xG.

The VLM is not fine-tuned. The statistical model learns from the known coordinates
and outcomes in StatsBomb training matches. At evaluation time, the hybrid uses
VLM-reported positions. It never substitutes the true test-shot coordinates.

## Run the study

From the repository folder, with the virtual environment active and
`OPENAI_API_KEY` set:

```bash
python hybrid_xg.py --n 300 --seed 7 --output results/hybrid_gpt_6_luna_300
```

The study uses the saved 300-shot cohort from `results/gpt_6_luna/shots.csv`.
It reuses the 50 saved extractions from `results/geometry_gpt_6_luna_50` after
checking the prompt/model signature, source-data hash, and image hashes.
Only the remaining 250 images require new requests. Requests run sequentially
and responses are saved as they complete. Earlier xG results are unchanged.

To prepare the reference data and save the fixed protocol without OpenAI calls:

```bash
python hybrid_xg.py --prepare-only
```

To rerun the evaluation using only completed extractions:

```bash
python hybrid_xg.py --offline
```

A change to the source data, implementation, prompt, model, selected shots,
or model settings requires a new output folder.

## Fixed design

The probability model uses five features:

- Distance from the shooter to goal center.
- Goal-opening angle between the shooter-to-post lines, calculated in radians.
- Goalkeeper depth from the goal line.
- Goalkeeper perpendicular distance from the shooter-to-goal-center line.
- Whether the goalkeeper is inside the triangle formed by shooter and goalposts.

Header/body part and other defender positions are not supplied to the hybrid.
The reported distance and angle numbers from the VLM are also not used; code
recalculates both from its reported shooter coordinates.

There is no feature or hyperparameter search. The pipeline uses training-fold
median imputation with missingness indicators, training-fold standardization,
and L2 logistic regression with C=1 and max_iter=2000. It does not balance classes.

The reference pool is the 1,430 eligible World Cup 2022 shots. GroupKFold makes
five folds over the full pool. Every tested match, including its other shots,
is excluded from training that predicts it. Imputation and scaling are fitted
only on training matches. All 300 selected shots are retained; missing VLM
geometry is imputed from the training fold, not replaced with the true location.

Controls include the identical fitted model evaluated with known coordinates,
a distance-and-angle model with the same preprocessing and folds, the saved
four-feature logistic baseline, and StatsBomb xG. The original baseline is
refitted to verify the same data and folds reproduce its saved predictions.

Paired Brier intervals resample whole matches 4,000 times. They condition on the
fitted predictions and do not refit the models or adjust for multiple comparisons.
The evaluation set has already been studied in earlier experiments; results
are exploratory and need confirmation on new matches or another competition.

The original logistic baseline uses header and opponent-count information absent
from this hybrid. The known-geometry control helps assess extraction quality,
while the other baselines provide practical references rather than equal-input
comparisons.

Outputs include reports, a comparison figure, all per-shot xG values and VLM-derived
features, score tables, paired intervals, the fixed protocol, and a fold audit with
training/test match IDs and model parameters. API credentials are never saved.

```bash
python -m unittest discover -s tests -v
```
