# Proposed next step: replicate the reasoning-effort result

Status: superseded by the user-authorized autonomous campaign, now completed. The original proposal below is preserved as history. Actual protocols/results: results/direct_scope_suite/; readable summary: RESEARCH_SUMMARY.pdf; current status: CURRENT_RESEARCH_STATUS.md.

## Reason for the next step

The completed 12-shot pilot favored medium effort on Brier, but its interval included zero and one scored shot contributed approximately 62% of the net gain. Medium's AUC remained below the logistic reference. We should test whether the benefit repeats before changing the image, prompt or reasoning setting again.

## Proposed comparison

Use the same original image-only zero-shot prompt, original renderer, GPT-6 Luna, high image detail and 2,048-token per-request output allowance. The VLM returns raw xG directly. Compare one low-effort and one medium-effort request for each of 50 new shots, with paired request order randomized. Logistic regression remains a separately fitted comparator; never send its prediction or any query-specific numerical answers to the VLM. Do not recalibrate, average or replace VLM outputs.

## Data eligibility and current limitation

The saved 300-shot cohort contains 288 shots outside the pilot and 237 shots outside the pilot's 11 matches, but every one of those saved shots has been studied previously. These counts do not establish unused confirmation matches.

Find eligible matches in the same StatsBomb open-data source, excluding all match groups previously used in training, development, evaluation or prompt/rendering diagnosis. Known used cohorts include World Cups 2018/2022, Euros 2020/2024 and the 34 Leverkusen 2023/24 fixtures. Audit stored match IDs, source provenance, shot eligibility and duplicate identities before claiming a new cohort is untouched. The exact 50 shot IDs are not yet frozen because a fresh eligible source pool has not been verified.

After verifying the candidate pool, freeze match and shot IDs, source/image/code hashes, prompts, settings, request order and evaluation protocol before inference. Select without looking at goals, baseline probabilities, earlier errors or prior VLM scores. Use a natural shot sample rather than enforcing the pilot's equal distance strata. Exclude penalties and require the same shot-snapshot eligibility as the original study. Reserve whole matches from baseline training.

## Budget

At most 100 generation attempts including failures and retries: two per shot, in batches of at most 10. No hidden retries or automatic replacements. Estimated cost approximately $0.03 using the completed pilot's usage rate; this is not a guaranteed bill. Hard cumulative limits: $0.10 usage-derived conservative cost reservation, 100,000 input tokens, 100,000 output/reasoning tokens and 2,048 output tokens per request. Preflight image tokens and reserve the full next response before every generation. Stop on any cap, invalid/incomplete response, credit exhaustion, unexpected model or uncertain charge. No automatic expansion if the cohort cannot be completed within those limits.

## Evaluation

Prespecify medium minus low paired Brier difference as the primary comparison. Report whole-match uncertainty, log loss, AUC, raw calibration, failures and cost. Report medium versus the original logistic comparator as a secondary comparison and disclose its extra header information; keep any image-observable-feature comparator separate and do not silently weaken the original baseline. Keep pilot and replication results separate.

Fifty shots are a practical replication step, not a statistically powered proof of superiority. A convincing baseline win may require many more independent matches and goal events. Decide on any larger confirmation only after reviewing this fixed result; do not keep sampling until a favorable interval appears.

