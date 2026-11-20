# StatsBomb visual in-context learning

Examples in the inference prompt, with observed goal/no-goal labels; no weight updates.
Compare GPT-6.2 Sol with 0, 8, 32 and 64 natural random training examples, plus shuffled-label controls.
Use separate whole-match development and confirmation groups, raw xG probabilities, and a combined $10 cap.

`experiment.py prepare` freezes the study and makes contact sheets without network access.
Import `run.mjs` in a native Node runtime with fetch access; `countPreview()` counts input tokens without generation.
Call `runBatch('development', 30)` until complete, then `experiment.py select`.
Call `runBatch('confirmation', 30)` until complete, then `experiment.py analyze`.
No generation starts on import. Credentials are read internally from the existing project credential file and are never saved in outputs.

Outputs: `results/statsbomb_icl_v1/`. Retain StatsBomb Open Data attribution.

Completed investigation: 1,100 new calls, estimated $6.19321, conservative $7.62215 under one combined $10 cap. The final report is `results/statsbomb_icl_v1/FINAL_REPORT.md`.

The outcome-labelled study tested 8/32/64 examples on 40 development shots and selected 64. On 200 match-held-out shots, correct outcomes scored Brier 0.103870 versus zero-shot 0.105254; the difference was inconclusive.

A separate study used the same 64 training diagrams with StatsBomb xG labels, explicitly treating these as teacher predictions. It was declared before confirmation accuracy was inspected. On 100 shots with fresh zero-shot controls, Brier was 0.081542 versus 0.093860, but shuffled teacher labels scored 0.082607. The correct-versus-shuffled comparison was inconclusive. The teacher gain does not establish learning from correct image–probability pairings.

Secondary workflow: `teacher.py declare`, then after the original 800 calls `teacher.py prepare`; import `teacher_run.mjs` and call `countPreview()`, then `teacher.py choose`; run `runBatch('evaluation', 30)` to completion, then `teacher.py analyze` and `finalize.py`. The secondary runner includes the immutable original ledger in every budget reservation. Its outputs are in `results/statsbomb_icl_teacher_v1/`.

Run `test_budget.mjs` for offline budget checks. `audit.py` verifies the original 800 requests; `finalize.py` also verifies the secondary 300, reconstructs every request hash, and writes the combined report. Imports do not dispatch paid calls. Existing frozen protocols are preserved on rerun; successful cached jobs are not regenerated.
