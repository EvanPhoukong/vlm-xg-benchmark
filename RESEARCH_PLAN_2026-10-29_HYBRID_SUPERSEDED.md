# Proposed xG research continuation — approval required

Date: 29 October 2026 (America/Los_Angeles).
Status: PLAN ONLY. No experiments, model requests, data downloads, or restarts authorized by this document. The user will approve the next stage.

## Recommendation

Test one new question: **Can Luna extract useful body orientation and shot context from real pre-shot imagery, and does that information improve a supervised xG model beyond a conventional baseline?**

Prior work mainly used rendered dots on a pitch. Real imagery can supply information those diagrams omit. A successful annotated-feature model would motivate testing visual extraction; it would not itself demonstrate VLM success.

Use `gpt-6-luna` for every proposed paid request, with an explicit allowlist and no automatic model fallback. Haiku 5.5 is not needed for this plan. The active chat model has not been switched or verified; selecting Luna for the chat is separate from configuring experiment requests.

## Existing experiments to preserve and avoid repeating

Reviewed the local reports and the chats “Assess multimodal approaches” and “Propose new research improvements.” This is a report audit, not a fresh numerical reproduction.

| Already tested | Evidence and implication |
|---|---|
| Direct image-only xG with several VLMs | Original project benchmark includes Sonnet, Opus, Gemini and Gemma; later work adds Luna. Do not repeat a broad model sweep. |
| Luna calibration | On 300 World Cup shots, raw Brier 0.1124; mean correction 0.1098; sigmoid correction 0.1106; logistic baseline 0.1070. Neither correction established improvement over raw Luna. |
| Geometry extraction and code calculation | Luna located most markers accurately; code improved angle calculation. This demonstrated measurement ability, not better xG. |
| Basic geometry hybrid | On those 300 shots, hybrid Brier 0.1150 versus direct Luna 0.1124 and baseline 0.1070. No established advantage. |
| Noise training, multiple readings, four worked examples, pixel extraction, event context | The best VLM screen was multiple readings plus noise training: 0.07275 versus original 0.07756 on reused Euro shots. Fresh primary comparisons were inconclusive. |
| Unfinished confirmations | Best-candidate confirmation stopped at 46/200 shots after credits ran out. A clearer-image Euro 2020 attempt stopped at 62/200. Preserve caches; do not restart either under this plan. |
| Specialist readers and metadata fusion | Latest study used 600 Luna calls on 200 previously studied Bundesliga shots. Specialists scored 0.08069 versus previous pipeline 0.08004. Primary intervals included zero; metadata did not improve the point estimate. |

World Cup, Euro and Bundesliga scores come from different cohorts and must not be ranked as if they were the same test. The evidence does not yet establish a reliable VLM xG improvement over the relevant baselines.

Local evidence:
- [Original benchmark](README.md)
- [Calibration](results/calibration_experiment/REPORT.md)
- [Geometry](results/geometry_gpt_6_luna_50/summary.md)
- [Basic hybrid](results/hybrid_gpt_6_luna_300/summary.md)
- [Improvement suite](results/improvement_suite/final_report.md)
- [Specialists and fusion](results/task_division_v1/report.md)

## Stage 1 — data and information-value check; zero model API calls

1. Inventory xGHub's available annotations, actual shot/goal/match counts, sampling rules, missingness, match identifiers and shot timing. Verify whether restricted footage is already accessible and its terms permit the proposed API processing. Do not submit an access request or accept an NDA on the user's behalf.
2. Verify that inputs can stop at or before shot contact, before ball flight or outcome is visible. Mask scoreboards and remove outcome fields, result-coded filenames and identifying match metadata from model payloads. Confirm that annotations describe information available at that cutoff.
3. Compare match identities with every prior training and evaluation cohort. Keep entire matches, duplicate shots and alternate clips together. Reserve a sealed confirmation set before model development; use only development matches for these checks.
4. Fit a fixed regularized logistic baseline using distance, angle, body part and available conventional pressure/occlusion features. Compare the same model family plus a small prespecified orientation/technique feature block. Encode angles cyclically. Use match-grouped development cross-validation, training-only preprocessing and no hyperparameter sweep.
5. Report paired Brier differences and uncertainty, log loss, calibration and goal counts. Available annotations are reference measurements, not perfect ground truth. Added-feature improvement would demonstrate information value, not that Luna can extract it.

Gate: proceed to a paid pilot only if usable pre-shot images are available, input timing is defensible, match separation is possible, and the annotated-feature check is promising. A development interval excluding zero is not required; weak or harmful results should stop the current proposal rather than trigger a search through many variants.

## Stage 2 — one small Luna feasibility pilot

Maximum **40 new API attempts in total**, including failures and retries, in batches of **at most 10 attempts**. One request per shot with one fixed pre-shot image or fixed small frame set; choose the representation from development-only timing inspection before inference. No specialist ensemble, repeated readings, worked-example sweep or model comparison.

Use development matches only. The first 10 shots check parsing, visibility and extraction feasibility; up to 30 additional shots check consistency. Stop early for credit exhaustion, invalid outputs, unusable visibility or unreliable timing. Any material prompt change becomes a new protocol and requires another review, rather than consuming the remaining budget opportunistically.

Return compact structured attributes with an explicit unknown option, such as body part, coarse torso orientation and shot technique. Avoid unvalidated precise 3D claims. Code computes geometry and the fixed supervised head estimates xG. Keep a conventional baseline with the same shared metadata, plus an annotation-reference variant to diagnose extraction losses. Evaluate missing readings using training-only imputation and missingness indicators; report every eligible shot and coverage.

Hard usage limits: at most 600 output tokens per attempt; stop before another request would exceed 120,000 cumulative input tokens or 24,000 output tokens. Count image tokens and reasoning/output usage, record cached tokens separately, disable hidden SDK retries, save every response and inspect the returned model identifier. Preflight each payload and track failed attempts; if usage for a failed request is unknown, stop for reconciliation. No paid requests have been made in preparing this plan.

Deliver a short pilot report with extraction agreement/errors against available references, missingness, example failures, actual token usage and a go/no-go recommendation. Forty shots cannot establish reliable xG superiority.

## Stage 3 — independent confirmation; separately approved

After a successful pilot, use development-only paired losses to estimate the number of independent matches and goals required for a worthwhile Brier reduction. Propose the required inference budget and sample size before making further calls. A large number of bootstrap resamples is offline computation, not additional VLM inference.

Freeze the feature definitions, prompt, model version, preprocessing, fitted probability head, missing-data policy, one primary comparison and evaluation cohort before opening confirmation outcomes. Evaluate on naturally sampled untouched matches, without selecting goals or easy images. Compare baseline versus baseline plus Luna-derived visual attributes on identical shots and shared metadata. Report the annotation-reference control and training-rate constant separately.

Primary evidence: paired Brier improvement with a whole-match 95% interval wholly below zero, against a prespecified practically worthwhile effect. Also report log loss, calibration, discrimination, coverage and failures. Keep one primary hypothesis; adjust or clearly label additional tests. Small or uncertain results remain inconclusive. A repeat on another competition would strengthen generalization; no positive outcome is guaranteed.

## Combining datasets

Use xGHub alone first to isolate the new information. Consider StatsBomb plus xGHub only after units, feature meanings, shot eligibility and match overlaps have been audited. Use only shared fields for pooled training, retain source labels, group overlapping matches together, and evaluate each source separately. Body pose missing from StatsBomb must not be silently invented. A pooled model is a later proposed experiment, not part of the 40-call pilot.

If xGHub video is unavailable, complete the annotation-only feasibility report and report the access dependency. Do not substitute rendered xGHub diagrams and claim broadcast-image validation, or automatically launch a different dataset experiment.

## Sources and interpretation

- [xGHub official repository](https://github.com/mguti97/xGHub): documents shot outcome, geometry, body part, technique, pressure, occlusion, orientation and match identifiers.
- [xGHub dataset card](https://huggingface.co/datasets/mguti97/xGHub): restricted footage requires institutional access and NDA compliance; annotation and video permissions differ.
- [Once Upon a Goal, original paper](https://www.sciencedirect.com/science/article/pii/S1077314226000998): reports orientation information as useful for specific subsets, with limited contribution elsewhere. This motivates a test rather than establishing general-purpose VLM superiority. The abstract was available; full quantitative tables were not verified in this planning pass.

Approval requested: Stage 1 and, only if its gates pass, Stage 2 within the stated limits. Stage 3, dataset pooling, model changes and additional prompt experiments require a new concrete proposal.
