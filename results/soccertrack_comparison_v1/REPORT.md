# SoccerTrack v2: expanded comparison

Status: partial_offline_analysis. No results are imputed for unrun models.

Audited 306 unique SHOT/GOAL situations from 10 matches (44 goals). The official six training matches supply 187 shots; validation has 55; the two official test matches have 64. The paired image cohort remains the earlier 30-shot pilot, with 7 goals.

## Existing paired image results

| Method | Shots | Brier ↓ | Log loss ↓ | AUC ↑ | Mean xG |
|---|---:|---:|---:|---:|---:|
| logistic_distance | 30 | 0.18688 | 0.6230 | 0.677 | 0.106 |
| logistic_distance_angle | 30 | 0.18952 | 0.6073 | 0.708 | 0.128 |
| logistic_distance_angle_header | 30 | 0.17961 | 0.5759 | 0.733 | 0.122 |
| train_constant | 30 | 0.18778 | 0.5751 | 0.500 | 0.139 |
| gpt-6.1-sol | 30 | 0.19091 | 0.6001 | 0.550 | 0.115 |

Every row above uses exactly the same 30 shots. Sol predictions are reused only after verifying the original image and prompt fingerprints. The primary baseline is distance + goal-opening angle logistic regression fitted on the six official training matches. The header baseline has additional explicit body-part information. All coefficients and standardization use training data only; no hyperparameter search or validation refit.

Sol minus the primary logistic baseline Brier difference: +0.00138. Lower is better. This is a point comparison on two matches, not evidence of population superiority or equivalence.

## Model coverage and spending

| Model | Original predictions | Exact repeats | Conservative tracked exposure |
|---|---:|---:|---:|
| gpt-6-luna | 0/30 | 0/10 | $0.0000 / $10 |
| gpt-6.1-terra | 0/30 | 0/10 | $0.0000 / $10 |
| gpt-6.1-sol | 30/30 | 0/10 | $0.1309 / $10 |
| gpt-6-astra | 0/30 | 0/10 | $0.0000 / $10 |

New paid calls in this extension: 0; estimated new spending $0.0000. Exposure includes the earlier verified SoccerTrack Sol pilot. This is experiment accounting, not an account-wide API balance. The runner reserves input plus maximum output (including reasoning) before dispatch, keeps uncertain requests charged at the full reservation, and has no automatic retries. One single-writer lock prevents concurrent budget races.

## Larger baseline-only checks

Official test matches (all reconciled SHOT/GOAL events):

| Method | Shots | Brier ↓ | Log loss ↓ | AUC ↑ | Mean xG |
|---|---:|---:|---:|---:|---:|
| logistic_distance | 64 | 0.11914 | 0.4023 | 0.756 | 0.109 |
| logistic_distance_angle | 64 | 0.11913 | 0.3923 | 0.778 | 0.116 |
| logistic_distance_angle_header | 64 | 0.11834 | 0.3897 | 0.780 | 0.117 |
| train_constant | 64 | 0.13213 | 0.4346 | 0.500 | 0.139 |

Secondary leave-one-match-out results across all 10 matches:

| Method | Shots | Brier ↓ | Log loss ↓ | AUC ↑ | Mean xG |
|---|---:|---:|---:|---:|---:|
| logistic_distance | 306 | 0.11345 | 0.3807 | 0.703 | 0.144 |
| logistic_distance_angle | 306 | 0.10719 | 0.3654 | 0.700 | 0.144 |
| logistic_distance_angle_header | 306 | 0.10442 | 0.3596 | 0.698 | 0.144 |
| train_constant | 306 | 0.12325 | 0.4123 | 0.445 | 0.144 |

The cross-validation fits both scaling and coefficients on the other nine matches. These are baseline robustness results; the VLM was not evaluated on all 10 matches. No confidence intervals treating correlated shots as independent are reported.

## Data and interpretation limits

- Removed 4 duplicate SHOT/GOAL annotations; 0 events excluded for an unresolved source match.
- Reconciled 7 shot actor IDs using the unique raw primary-shot record at the aligned timestamp. The original labels/images remain unchanged. See source_audit.json.
- Raw provider x runs across the pitch width and y along the attacking direction. Converted with a 68 × 105 m pitch; penalty-area tag checks passed. This is an internally checked interpretation, without an independent tracking/video geometry audit.
- Geometry is recorded at contact; images are 0.32 seconds earlier. The image does not identify the shooter with a marker. These input differences can affect the comparison.
- The SHOT/GOAL event definition may omit blocked/header attempts placed in other BAS classes. Penalties have no dedicated label and are not silently excluded.
- The prompt assumes an average professional finisher while these are university-level matches; it is retained for an exact comparison with the pilot.
- These test images and earlier Sol results were already inspected. This is an exploratory extension, not a new independent confirmation study.
- No VLM probability is calibrated, blended, clipped, averaged, or supplied with the baseline estimate. Reliability curves are diagnostics only.

## Execution

Model coverage is incomplete. The preparation session could read local files but could not resolve external API/download hosts. See runner_status.json for the most recent execution state. The broader model/repeat study remains pending where the coverage table is incomplete. The local video/tracking files are Git LFS pointers; no additional real frames were invented or downloaded.

From the repository root, in an environment with NumPy and network access:

```bash
python experiments/soccertrack_comparison/prepare.py
python -m unittest discover -s experiments/soccertrack_comparison -p 'test_*.py' -v
python experiments/soccertrack_comparison/run.py --preflight
python experiments/soccertrack_comparison/run.py --execute
python experiments/soccertrack_comparison/analyze.py
```

The original pilot is cached; 130 new calls are scheduled: 90 original predictions (Luna/Terra/Astra) plus 40 exact repeats (all four models). The common test cohort and prompt are frozen; no test-set prompt selection or spending to a target score. A larger VLM study needs new pre-shot images from additional match groups and a separate fixed protocol, not repeated optimization of these two matches.

Full numeric output: summary.json, paired_predictions.csv, subgroups.json, baseline_models.json, lomo_predictions.json, audit.json.

Pricing checked against https://developers.openai.com/api/docs/pricing on 2026-11-13 UTC. Dataset provenance: local SoccerTrack v2 README.md, CHANGES_v1_1.md, CHANGES_v1_2.md, and hashed BAS/player_nodes files.
