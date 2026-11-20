# Proposed next experiment: direct VLM xG, low versus medium reasoning

Status: user authorized the low-versus-medium pilot on 29 October 2026; all 24 requests are complete. Estimated API cost $0.0063094. Results: results/reasoning_effort_v1/report.md. No automatic expansion or additional experiment is authorized.

## Scope

Preserve image + generic instructions -> gpt-6-luna -> raw xG probability and a short rationale. Keep the original zero-shot prompt and original image bytes identical between conditions. Logistic regression is a separate comparator. No baseline predictions, outcome labels, query-specific coordinates or StatsBomb xG enter model inputs. No downstream probability correction, averaging, statistical head or calibration is applied.

## Audit of novelty

The inspected original OpenAI backend, geometry diagnostic, rich-geometry/direct-xG runner, worked-example/multiple-view runner, specialist runner and saved protocols all specify low reasoning effort. No completed low-versus-medium reasoning comparison was found in the inspected files. This statement is limited to the available code and records.

Already tested: explicit scale instructions, numeric grids, geometry-first prompting with secondary direct xG, enlarged views, worked geometry examples, repeated readings, statistical calibration and specialist readers. These must not be presented as new experiments.

A role-separated clearer-image run is incomplete. It is not a completed negative result and must not be restarted automatically. Hollow or smaller markers could address known rendering overlap, but a renderer change would be a separate proposal with a separate novelty audit.

## Hypothesis

Additional model reasoning may improve its use of the visual evidence when estimating scoring probability, even with unchanged image and prompt. This is plausible, not established; it may increase cost without improving xG.

## Small development pilot

Select 12 development shots using a recorded seed and prespecified distance strata, independently of outcomes, prior errors and model probabilities. Retain ordinary and crowded scenes rather than selecting only known failures. Do not describe normalized pitch units as verified physical yards.

For each shot, request one low-effort and one medium-effort response with the same model, prompt, image, output schema, image detail and maximum output budget. Randomize arm order within each shot and interleave requests to reduce time-order effects. Use fresh paired controls rather than assuming an older response is identical in model/service conditions.

Maximum 24 new attempts, including failed requests and retries. Process at most six attempts per batch. Disable hidden SDK retries, save each result immediately and verify the returned model identity. Preserve the existing total limits of 120,000 input tokens and 24,000 output/reasoning tokens; use the same per-request maximum of 2,048 output tokens for both arms. Preflight conservatively; stop before exceeding a limit. If failed-request usage cannot be reconciled, stop. Do not raise budgets automatically to complete the cohort.

Intermediate inspection checks response validity, credit availability and usage only. Do not change the prompt or select a winner after viewing partial xG scores. Interrupted or truncated responses remain recorded; a cohort stopped by the cap is incomplete, not a smaller finished experiment.

## Evaluation and decision

This pilot checks feasibility and produces exploratory paired results. With only 12 shots, it cannot establish improved probability accuracy. Report raw xG, Brier, log loss, response validity and reasoning-token usage. Keep sample IDs, actual image bytes, prompts, settings and timestamps. Check explicit factual statements in rationales against source geometry, with the reviewer blinded to effort and outcome where feasible; do not treat plausible explanations as proof of the model's internal process.

If a larger study is justified, propose its sample size and cost for separate approval. Freeze the method and use untouched match groups from the same data source for confirmation. Retain the original logistic comparator, clearly disclose its header-information advantage, and report a matched image-observable comparator separately. Do not silently weaken a comparator to obtain a win.

## Other in-scope directions, not part of this pilot

1. A controlled renderer change focused on marker occlusion, preserving source positions, football content and prompt. A general grid/zoom/clearer-image rerun is not novel; compare exact implementations before proposing it. Evaluate visual measurement and raw direct xG separately.
2. A consistency diagnostic using repeated identical inputs and mirrored scenes. Compare transformation-induced probability changes against ordinary repeat variability. Mirroring should have little effect under the position-only, unknown-body-part representation, but exact equality is not assumed. These are diagnostics, not independent goal-labelled shots or a new accuracy claim. Do not average the outputs or enforce a preferred probability.

Source for model capability: https://developers.openai.com/api/docs/models/gpt-6-luna
General visual-presentation motivation, not football/xG validation: https://arxiv.org/abs/2608.21170

