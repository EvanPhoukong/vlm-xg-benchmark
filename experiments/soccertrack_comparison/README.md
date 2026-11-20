# SoccerTrack comparison

## Larger Sol evaluation

`expand_sol.py` freezes all **306 unique shots, 44 goals, 10 matches**, retains the 30 original Sol probabilities and schedules 276 new predictions with the unchanged image-only method. Primary comparator: leave-one-match-out distance/angle logistic regression, fitted on the other nine matches. The official 64-shot test comparison uses the original six-training-match baseline. The original 30 and additional 276 are reported separately. More test data can change the score in either direction; it does not train the model.

```bash
python experiments/soccertrack_comparison/expand_sol.py prepare
python -m unittest discover -s experiments/soccertrack_comparison -p 'test_expansion.py' -v
python experiments/soccertrack_comparison/expand_sol.py frames
python experiments/soccertrack_comparison/expand_sol.py run           # access preflight only
python experiments/soccertrack_comparison/expand_sol.py run --execute
python experiments/soccertrack_comparison/expand_sol.py analyze
```

Frame extraction additionally requires `av` (PyAV) and Pillow. It seeks individual pre-shot frames from the pinned dataset revision, accounting for a nonzero first decoded timestamp; it does not bulk-download all 77 GiB. Local video LFS pointers are not usable footage. Original PNGs remain unchanged; new PNGs use LANCZOS resizing, since the original resizing implementation is not established. Avoid interpreting the retained/new subgroup contrast as a controlled experiment in sample size.

The largest allowed input/output reservation for all 276 new calls is **$5.087232**, plus **$0.13086** already tracked Sol usage, under the $10 cap. Oversized input stops before generation. Uncertain requests retain their reservation and stop the run. The existing comparison and expansion share a campaign lock. Run one campaign at a time; after any legacy ledger change, reconcile budget/protocol before continuing.

**Current state: prepared, not executed.** PyAV is missing and external footage/API access is blocked in this session. No new Sol accuracy score exists. Outputs are under `results/soccertrack_sol_expansion_v1/`. An incomplete analysis deliberately reports no expanded accuracy metrics.

## Original 30-shot comparison

Offline preparation and analysis are complete. The paid comparison is pending network access.

The fixed experiment compares GPT-6 Luna, GPT-6.1 Terra, GPT-6.1 Sol and GPT-6 Astra on the same 30 verified pilot images. Sol's earlier original predictions are reused after checking exact image/prompt fingerprints. Ten exact-repeat diagnostics per model remain separate from accuracy predictions. No recalibration or prediction averaging.

Primary comparator: L2 logistic regression on distance and opening angle, trained on the six official training matches (187 shots). Secondary references use distance alone, distance/angle/header, and the training goal rate. The 306 reconciled shots across all ten matches also support a separate baseline-only leave-one-match-out analysis. The two official test matches are too few for a population superiority claim.

Each model has a separate $10 cap. Reservations cover input and maximum output, including reasoning. Accounting includes the earlier SoccerTrack Sol pilot ($0.13086 conservative), and does not represent all historical project/account spending. The runner is sequential, uses a single-writer lock, saves reservations before dispatch, stops on uncertain requests, and never automatically retries.

From the repository root:

```bash
python experiments/soccertrack_comparison/prepare.py
python -m unittest discover -s experiments/soccertrack_comparison -p 'test_*.py' -v
python experiments/soccertrack_comparison/run.py             # offline schedule only
python experiments/soccertrack_comparison/run.py --preflight # model access; no generation
python experiments/soccertrack_comparison/run.py --execute
python experiments/soccertrack_comparison/analyze.py
```

Requires Python 3.10+ and NumPy. Uses the standard-library HTTP client and the existing credential file or OPENAI_API_KEY. No import, preparation, analysis, or test starts paid inference.

Results: `results/soccertrack_comparison_v1/REPORT.md`, `summary.json`, `paired_predictions.csv`, `subgroups.json`, `source_audit.json`, and `audit.json`. Baseline source coordinates use the raw provider's transverse x and attack-normalized longitudinal y, internally checked against penalty-area tags. Same-frame duplicates and opponent actor IDs are reconciled explicitly. An initial unrun preparation is preserved with an amendment record. Existing source files and pilot results are unchanged.

The local full videos and GSR files are Git LFS pointers. This extension does not claim to have run VLMs on the larger 306-shot label cohort or on independent new matches.
