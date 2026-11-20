# Original study context

Source: user-provided two-page PDF, read and visually inspected on 29 October 2026.
Local source: C:/Users/ephou/.codex/attachments/50ef2c3a-1fc6-403a-8207-3897eed55b10/mit_sloan-1.pdf
Title: Can Vision-Language Models Estimate Expected Goals? A Benchmark Against Statistical xG

This is source material describing the study, not authorization to run experiments. The user's current direction is to preserve the original direct image-to-VLM-to-xG question. Paid work remains subject to approval.

## Original question and protocol

Estimate goal probability directly from a shot-situation image, and identify systematic differences from statistical models. The primary study uses a standardized zero-shot prompt and requests a probability plus a one-sentence rationale.

Source pool: StatsBomb 2022 World Cup, 1,430 non-penalty shots with freeze frames, including 152 goals. Inputs are half-pitch diagrams of shooter, teammates, opponents and goalkeeper; no body-part or play-context metadata is supplied to the VLM.

The PDF reports a common evaluation cohort of 298 shots and 42 goals. Later repository reports often use 300 shots and 42 goals. Do not silently merge these cohorts or their metrics. Reconcile shot IDs and exclusions before regenerating original-study results; the reason for the difference has not been established in this context review.

The original logistic comparator uses distance, goal-opening angle, header and opponent count in the shooting cone, with match-grouped cross-validation. Its header feature supplies information absent from the default image-only condition; preserve and describe this distinction.

## Initial results, as reported in PDF Table 1

| Method | AUC | Brier | Mean xG |
|---|---:|---:|---:|
| Gemma 4, 26B | 0.597 | 0.190 | 0.379 |
| Claude Sonnet 5.5 | 0.683 | 0.121 | 0.085 |
| Claude Opus 5.5 | 0.721 | 0.111 | 0.109 |
| Gemini 3.8 Flash | 0.721 | 0.116 | 0.167 |
| Logistic baseline | 0.768 | 0.106 | 0.108 |
| StatsBomb xG | 0.792 | 0.100 | 0.100 |

Opus minus logistic Brier difference: +0.005, 95% interval [-0.002, +0.013]. This is an inconclusive difference, not a demonstration of equivalence. The PDF reports all four VLMs as significantly worse than StatsBomb under its analysis.

The observed evaluation goal rate is approximately 14.1%, compared with approximately 10.6% in the full eligible pool. Mean predicted probability alone is not individual prediction accuracy.

Figure 1 and the discussion emphasize model-specific short-range errors. In the 6-12 distance bin (77 shots), observed conversion is approximately 18%; StatsBomb averages approximately 0.17, versus 0.15 for Sonnet, 0.32 for Gemini and 0.45 for Gemma. Opus and Gemini have similar ranking scores but different probability scales. Rationales sometimes misdescribe distance or angle; they suggest a diagnostic hypothesis without proving a causal mechanism.

The PDF labels distances in yards. Later geometry reports use normalized StatsBomb pitch units. Avoid presenting these coordinate units as verified physical measurements without reconciling conventions.

## Implications for continuation

Success includes reliable measurement of VLM performance, failure patterns and generalization; an overall baseline win is desirable but is not the only result that answers the original question.

Prioritize a concrete direct-VLM experiment linked to observed short-range or geometric interpretation errors. Audit candidate prompts and rendering changes against earlier geometry-first, worked-example, multiple-view and incomplete clearer-image experiments before claiming novelty.

Retain original results as historical evidence; evaluate revised methods prospectively on untouched matches for confirmation. Calibrators, hybrid probability heads and numerical baseline fusion remain separate follow-up research, not the primary direct-VLM method. The source PDF mentions hybrids, recalibration and broadcast video as future directions; this does not override the user's current scope or grant execution permission.
