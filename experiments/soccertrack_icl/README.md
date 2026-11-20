# SoccerTrack v2 in-context learning

Completed GPT-6.2 Sol comparison: 80 unique shots, 10 goals, 10 matches, six conditions per shot. This is SoccerTrack v2, the dataset previously called SoccerNet v2 in project requests.

Final outputs: `results/soccertrack_icl_completed_v1/REPORT.md`, `summary.json`, `audit.json`, and `predictions.csv`.

No-example Brier/AUC was 0.108515/0.6329. Eight outcome examples scored 0.112684/0.4300; sixteen scored 0.112716/0.4443. Sixteen logistic-labelled examples scored 0.109574/0.5579. Distance/angle logistic regression scored 0.087125/0.7471. All four prespecified 98.75% match-bootstrap contrasts crossed zero: no ICL improvement was established.

Every query's 16 example frames and logistic fitting data exclude its entire match. Eight examples form a nested prefix. Controls permute outcomes or teacher probabilities across identical images. Examples are before-shot panoramas letterboxed at most 768×384 into JPEG sheets; original query PNGs remain unchanged. Teacher labels come from other-match logistic predictions, not StatsBomb. No weights, calibration or output averaging are used.

The original 306-shot pool and six conditions were frozen before inference. Server token counts selected an 80-shot natural prefix covering all ten matches under a combined $10 cap. All source matches were previously studied, so this is exploratory evaluation.

## Preparation and execution

Run `experiment.py prepare`, import `run.mjs` in a native Node runtime, call `countPreview()`, then run `experiment.py choose`. Preparation and token counting make no generation calls. Credentials are read internally from the existing project credential file and never printed or saved in outputs.

The original `run.mjs` completed 96 predictions before the OpenAI account exhausted its credits. The six rejected requests and one bounded diagnostic rejection are retained. A separately declared Sonnet fallback completed four startup predictions. The user then reported adding $5 and authorized continuing the original Sol study. No Sonnet prediction is used in the reported comparisons.

The remaining 384 Sol predictions are recorded in `results/soccertrack_icl_resume_v1/`. `resume.mjs`, `resume_fast.mjs`, and `resume_robust.mjs` preserve successive implementation versions and protocols. The final active runner is `resume_robust.mjs`; import it and call `runBatch('evaluation', 24)` to inspect completion. All scheduled jobs are already complete, so the same call does not generate replacements. The original runner refuses to bypass its retained rejected-request records.

The fast version reuses the global maximum of 60 previously server-counted largest-canvas match/condition previews plus the same 1,024-token margin. Actual response usage is still checked. The robust version retries only transient filesystem rename operations. No model inputs or settings changed. One save failed before a new batch was dispatched; all existing paid responses were intact and none was repeated.

The Sonnet fallback is superseded. Its old protocol predates the replenished-credit Sol continuation and must not be resumed without updating combined accounting for any newly authorized work.

## Audit and accounting

`finalize.py` audits all 480 Sol request hashes, source annotations, match exclusions, label permutations, image pixels, raw outputs and token costs. It separately verifies the four Sonnet startup requests and retains the maximum reservations for all seven rejected requests. The completed audit passed 29 checks.

Sol successful-call estimate: $5.4419356. Sonnet startup: $0.046076. Combined estimate: $5.4880116. Conservative combined accounting, including all rejection reservations: $7.065415, below $10. There are no active paid runners.

Source: SoccerTrack v2 by Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim and Keisuke Fujii, under CC BY 4.0. https://huggingface.co/datasets/atomscott/soccertrack-v2
