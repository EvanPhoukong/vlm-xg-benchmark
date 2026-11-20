# Direct-image xG experiment log

Started 29 October 2026 after the user authorized autonomous, logical experiments within the original project scope. User account budget approximately $10; this campaign enforces a lower $2 working cap, inclusive of the earlier $0.0063094 pilot. Published standard Luna rates are recorded in campaign_protocol.json. Actual account balance cannot be checked by this runner.

## Question

Can an unchanged general-purpose VLM estimate goal probability directly from a shot-situation image, and how does it compare with logistic regression? All new model outputs are raw probabilities. No recalibration, probability adjustment, statistical head, query-specific numeric answers, outcome prompts or model fine-tuning.

## Fixed study sequence

1. Reasoning study: 50 naturally sampled fresh AFCON 2023 shots, paired low/medium/high Luna effort. Same prompt, original renderer and uniform 4096-token output allowance. The earlier 12-shot pilot used a 2048-token allowance; its scores are separate development evidence.
2. Marker-visibility study: 40 previously studied World Cup development shots, original filled star versus a hollow star at the same location and size. This is a narrow renderer intervention, distinct from previous grid, zoom and separated-panel tests. It is exploratory and must earn an independent confirmation.
3. Confirmation: 150 AFCON shots in wholly separate match groups, reserved before development inference. Freeze the selected effort/renderer first. Two primary comparisons: candidate versus the four-feature logistic baseline and candidate versus original low-effort prediction, with 97.5% whole-match intervals. No confirmation-driven tuning or sample extension.
4. Consistency diagnostic: 20 development shots, compare a horizontal mirror against ordinary exact-repeat variation. These are not additional independent goal-labelled shots, and outputs are not averaged or corrected.

## Data and comparators

Public StatsBomb source: https://github.com/hudl/open-data
The preparation excluded 264 prior match IDs from World Cups 2018/2022, Euros 2020/2024 and the Leverkusen Bundesliga fixture pool. The new competition has 52 matches; a seeded match split assigns 16 eligible match groups to development and the rest to confirmation. Seeded shot sampling uses eligibility and IDs, not goals, model errors or reference probabilities.

The primary logistic comparator uses distance, angle, header and opponent count in the shooting cone. It is fitted once on 3,068 World Cup 2018/2022 shots. A three-feature body-part-free model is reported separately on fresh data. Both receive recorded coordinates; the VLM only receives rendered images. No test-match outcomes fit these models. Development rendering comparisons use existing match-grouped out-of-fold baseline predictions rather than in-sample fitted values.

## Interpretation rules

Lower Brier and log loss are better; higher AUC is better. Better means must be accompanied by uncertainty and sample sizes. An interval crossing zero does not demonstrate equivalence. A lower average probability or a mean matching the scoring rate is not sufficient evidence of better individual predictions. Development winners are not confirmed by the data used to select them.

Public source files, hashes, selected IDs, frozen protocols, randomized schedules, complete API responses and input/output/reasoning token usage are saved. Unknown charges, exhausted credits, invalid or incomplete responses stop paid generation; no difficult shots are replaced. All request batches contain at most ten sequential attempts.

## Current state

Consult status.json and attempts.json for live completion and cost. No stage accuracy is scored before its full frozen cohort finishes. The final report will include every completed, negative, inconclusive or interrupted experiment.
