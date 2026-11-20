# Research continuation — original project question

Date: 29 October 2026 (America/Los_Angeles).
Status: scope revised following the user's decision; experiments still require approval. The previous hybrid proposal is preserved in RESEARCH_PLAN_2026-10-29_HYBRID_SUPERSEDED.md and is superseded.

## Primary research question

Can a general-purpose vision-language model directly estimate the probability of a goal from an image of the shot situation, and how does its performance compare with logistic regression and StatsBomb xG?

Primary prediction path: shot image + generic instructions -> VLM -> xG probability.

Preserve the original zero-shot benchmark as the primary reference. Do not silently change the study into baseline-plus-vision, image-to-geometry plus a statistical head, supervised distillation, or a fine-tuned vision model. Any future training or worked-example condition must be explicitly distinguished from the original zero-shot question and separately approved.

## Prediction inputs and comparators

The VLM may receive an outcome-blind shot image, a generic marker legend, and fixed instructions. For the primary image-only condition, do not provide query-specific coordinates, numeric baseline predictions, StatsBomb xG, shot outcomes or extra event metadata.

The VLM itself returns the probability. No trained downstream model, probability calibrator, baseline blend or external geometry calculation supplies or revises its primary prediction.

Logistic regression and StatsBomb remain separate comparators. Retain the original four-feature logistic baseline in the historical results, and clearly identify its header-information advantage over the image-only VLM. Include an image-observable-feature baseline where appropriate; do not quietly weaken or rename the original comparator. Baseline training must exclude all matches being evaluated.

## What changes are in scope

Candidate directions include an outcome-blind improvement to image readability, a different direct probability-elicitation instruction, or a comparison using real pre-shot imagery if access and shot outcomes are suitable. These are directions to assess, not claims that they are untried or likely to win.

Before selecting a paid experiment, compare its exact rendering and prompt with prior runs. Geometry-first direct xG, worked examples and multiple-view outputs have already been studied. Clearer or role-separated rendering also has an incomplete prior run; do not call it new or restart it automatically.

The next proposal should select one concrete, non-duplicate direct-VLM test and explain its intended mechanism and comparison. No experiment is selected or started by this scope revision.

## Existing work

Preserve all original benchmark results, calibration studies, geometry diagnostics, hybrid models, noise training, multiple readings, pixel controls, event-context fusion and specialist-reader results. They inform the research but do not establish a direct image-only VLM improvement merely because a hybrid score improved.

The earlier best-hybrid confirmation and clearer-image run remain stopped. No restarts are authorized.

## Budget and evidence

Use gpt-6-luna for proposed paid calls, with no automatic model fallback. The experiment model setting is separate from the model serving this chat.

After approval of a concrete protocol, the initial pilot remains capped at 40 request attempts including retries, in batches of at most 10, with saved responses and explicit token limits. No broad model or prompt sweep. Reuse compatible cached outputs for offline development checks where possible.

A small pilot checks feasibility. Previously studied matches remain exploratory. A confirmatory comparison requires a frozen method and untouched matches, which can come from the same StatsBomb source rather than a new dataset. Evaluate every eligible shot, report missing/invalid predictions, and use a prespecified failure policy rather than dropping difficult shots.

Primary evaluation is Brier score against actual goal outcomes, with whole-match uncertainty; report log loss, calibration and ranking as secondary measures. Treat StatsBomb xG as a comparison, not ground truth. Report inconclusive or negative results honestly.

Approval of the research scope does not authorize paid inference, downloads, model training or experiment restarts.

## Original project write-up

The user's original two-page Sloan write-up was read and visually inspected on 29 October 2026. Its protocol, initial results, limitations and implications are recorded in ORIGINAL_STUDY_CONTEXT.md. It reports 298 common evaluation shots; later 300-shot reports must remain distinct until cohort membership is reconciled. The original question includes identifying systematic VLM errors, particularly at short range, as well as overall predictive performance. Its mention of hybrids is future-work context and does not change the user's chosen primary scope.

## Completed direct-VLM reasoning pilot — 29 October 2026

The user authorized comparing reasoning settings while keeping API spending small. Completed 12 development shots and 24 successful gpt-6-luna requests, comparing low and medium effort with original image bytes and the original zero-shot prompt unchanged. Estimated inference cost: $0.0063094. Medium Brier 0.14697 versus low 0.18878; logistic reference 0.15696. The medium-minus-low exploratory interval includes zero; there are only two goals and one scored shot explains approximately 62% of the net improvement. This is not a confirmed baseline win. Full results, raw responses and integrity checks: results/reasoning_effort_v1/. No further model requests are authorized by this completion record.

## Completion record — autonomous campaign

The user subsequently authorized autonomous experiments within the original project scope and an approximately $10 API limit. The expanded frozen campaign is complete: 50-shot low/medium/high reasoning, 40-shot marker visibility, a match-disjoint 150-shot confirmation, and a 20-scene repeat/mirror diagnostic. No improvement earned a reliable baseline-win claim. Estimated total cost including the earlier pilot was $0.1004494. The chosen method remained the original diagram with low reasoning. See CURRENT_RESEARCH_STATUS.md and results/direct_scope_suite/FINAL_REPORT.md; no API runner remains active.
