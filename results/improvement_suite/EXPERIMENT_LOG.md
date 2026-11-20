# Research record

All scores below are Brier scores: lower is better. No negative experiments have
been removed from this record. Body-part context is kept separate from image-only
conditions. New models are not selected by repeatedly rescoring the original 300
World Cup shots.

| Experiment | Result | Interpretation |
|---|---|---|
| Original Luna, 300 World Cup 2022 shots | 0.1124; original baseline 0.1070 | Below the existing baseline by point estimate. |
| Mean calibration of Luna, same 300 | 0.1098 | Small, uncertain improvement; still below baseline. |
| First shooter/keeper hybrid, same 300 | 0.1150; known-coordinate control 0.1107 | No improvement; rare extraction errors and limited spatial features. |
| Richer spatial model development | World Cups 2018 + 2022: 3,068 shots, 287 goals, 128 matches | Candidate selection only; these scores are not independent confirmation. |
| Image-only spatial candidates | Linear 0.07204; spline 0.07190; boosted trees 0.07345 / 0.07411; baseline 0.07573 | Spline selected by development Brier before external evaluation. |
| Spatial candidates with header context | Linear 0.07137; spline 0.07104; boosted trees 0.07285 / 0.07327; baseline 0.07401 | Context spline selected independently within this input condition. |
| Fresh Euro 2024, known coordinates | 1,304 shots, 98 goals, 51 matches. Image spline 0.06424 vs image baseline 0.06583; context spline 0.06395 vs context baseline 0.06492 | Point improvements, but match-bootstrap intervals include zero. StatsBomb remains better at 0.06154. |
| Fresh Euro 2024, VLM pipeline | 200 shots, 16 goals. Image hybrid 0.07756; image baseline 0.07534; context hybrid 0.07869; context baseline 0.07570 | Failed to beat either baseline. Direct position-first Luna 0.07973; known-coordinate image control 0.07297. |
| Renderer diagnosis after Euro 2024 | Largest errors included nearby red defenders partly or fully hidden behind the gold shooter star | Measurement mismatch: the probability model knew blocker positions that the VLM could not fully see in the rendering. |
| New Euro 2020 reference | Frozen models, no retraining. 1,234 shots, 122 goals. Image spatial model 0.07343 vs image baseline 0.07965; context spatial 0.07362 vs context baseline 0.07770 | Supports trying the unchanged probability model with clearer images. StatsBomb remains better at 0.07030. |
| Role-separated Euro 2020 VLM test | Fixed random sample of 200, seed 7. Stopped at 62 responses at the user's request. | No final performance result. The run remains separate from the new noise-training experiment. |
| Noise-trained image hybrid, 2026-10-22 | Same 200 saved Euro 2024 shots: Brier 0.075009 vs original hybrid 0.077561. Difference -0.002552; approximate 95% match-bootstrap interval [-0.005269, -0.000490]. | Promising exploratory improvement over the current hybrid. Simple baseline 0.075335; new-minus-baseline interval [-0.007787, +0.006377] includes no improvement. StatsBomb 0.074505. Previously studied cohort, not independent confirmation. |

The fresh VLM experiment also records Luna's own xG after reading the structured
geometry. This is a new prompt condition, not a rerun of the original zero-shot
prompt. No shot-specific numeric labels, outcomes, commercial xG, or body-part
information are shown to the VLM. Context is supplied only as a separate input
to the context statistical model and matched comparator.

The original 300-shot results, the old geometry cache, and the first hybrid were
preserved. Full development results, fixed protocols, selected models, and
external score tables are saved alongside this log.

Euro 2024 informed the rendering revision and is not reused as independent
confirmation of that revision. Euro 2020 was reserved for the next test. The
probability models and their World Cup training data remain unchanged.

## Noise-training experiment, 2026-10-22

The user requested trying the proposed improvements in order of potential and
stopping after each completed experiment for review. The first idea is complete;
do not start the next idea until that review. Next proposed idea: multiple image
readings with disagreement checks.

The new statistical head uses the same spline family/settings and 3,068 World
Cup training shots. One clean plus fifteen noisy scenes per original shot retain
total training weight one. Shooter/keeper errors come from 300 training-only
World Cup readings; defender jitter 0.25 units and omission probability 5% are
fixed assumptions. Five-fold development keeps matches and noise donors out of
their test folds. No external tuning or new API requests were made.

The current hybrid reproduced its saved predictions within 9.9e-17. All 200
test shots were retained and protected inputs remained unchanged. Six new checks
passed. Code and results are saved in the actual WSL repository under
experiments/noise_robustness and results/noise_robustness_v1; an accessible report
copy is at C:/Users/ephou/Downloads/xG/output/noise-robustness-v1/report.md.

The old frozen models are preserved. The new head is a separate experimental
artifact. The incomplete Euro 2020 run remains stopped at 62/200 responses.
Fresh matches are required before claiming reliable improvement.

## Five-idea suite and fresh checks, 2026-10-22

The user subsequently authorized continuing all proposed ideas while they
napped and delivering one final report. All five 200-shot Euro 2024 screening
trials are complete. The extra best-VLM fresh confirmation is incomplete because
the API credit balance was exhausted; no partial-cohort accuracy scores are used.

| New trial | Same 200-shot Euro 2024 Brier | Interpretation |
|---|---|---|
| Multiple readings | Current head 0.074502; noise-trained head 0.072746 | Strongest screening image-only VLM direction. Comparison with the simple baseline remains uncertain. |
| Four worked training examples | Current head 0.077860; noise head 0.075194 | Failed to improve over their respective original-reading controls. |
| Pixel marker reader | Current head 0.074287; noise head 0.072552 | Separate computer-vision control; not a VLM success. |
| Additional pre-shot context | Current head 0.079042; noise head 0.076327 | AUC improved, but primary Brier worsened on this cohort. |
| Training-rate sanity reference | Constant 287/3068 probability; Brier 0.073784 | Added after screening and frozen before fresh results; no test-rate fitting. |

Fresh 200-shot check: 19 goals, all 34 publicly available Leverkusen Bundesliga
2023/24 fixtures represented. Original hybrid 0.080113; noise-trained hybrid
0.080040; simple known-coordinate baseline 0.082472; pixel/noise 0.080576;
StatsBomb 0.082537. Two predeclared primary comparisons have 97.5% match-bootstrap
intervals including zero. The earlier noise-training gain was not confirmed.

The strongest VLM candidate, multiple readings plus the noise head, was selected
and frozen before fresh scores. Its extra confirmation stopped at 46/200 after
46 successful second readings and 9 third checks, on credit_balance_exhausted.
Responses and the unchanged protocol are saved for resuming after credits are
available. A network-stalled worker was restarted earlier, reusing saved work.

Secondary full-pool controls use 908 known-coordinate shots, 102 goals and 34
matches, including the 200-shot VLM subset. Rich geometry Brier 0.085348 vs simple
0.089096; unadjusted paired interval [-0.007033,-0.000271]. StatsBomb 0.082558.
This analysis was added after the first fresh 200-shot scores were viewed;
it is not another independent cohort or a VLM result.

Sixteen targeted checks and 23 provenance/integrity checks passed. Successful new
API requests total 707. Requests ran sequentially. No API workers remain active.
No original model or prediction cache was replaced. The old Euro 2020 rendering
trial remains stopped at 62/200. New code is in experiments/improvement_suite;
artifacts are in results/improvement_suite. The accessible report is
C:/Users/ephou/Downloads/xG/output/improvement-suite/final_report.md.
